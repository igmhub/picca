#!/usr/bin/env python3
"""Export auto and cross-correlation for the fitter."""

import argparse
import os.path
import sys

import fitsio
import h5py
import numpy as np
import scipy.interpolate
import scipy.linalg
from numpy.polynomial.legendre import legvander
from scipy.sparse import coo_array

from picca.utils import (
    calculate_xi_ell,
    compute_cov,
    compute_cov_boot,
    get_legendre_bins,
    smooth_cov,
    userprint,
)

# TODO: add tags here when we are allowed to unblind them
UNBLINDABLE_STRATEGIES = ["none", "desi_m2", "desi_y1", "desi_y3"]

# BAO blinding templates, computed on a regular grid of bin centres and interpolated onto
# the binning of each measurement
BLINDING_TEMPLATE_DIR = "/global/cfs/projectdirs/desi/science/lya/lya_blinding/bao/"
BLINDING_TEMPLATES = {
    "desi_dr3": "dr3_blinding_v4_regular_grid_28_05_2026.h5",
}

# Binning of the template grid in h^-1 Mpc: r_trans in [0, 200] and r_par in [0, 200] for
# auto-correlations or [-200, 200] for cross-correlations, in bins of 4 h^-1 Mpc
TEMPLATE_BIN_SIZE = 4.0
TEMPLATE_R_MAX = 200.0


def get_template_bin_centres(is_x_correlation):
    """Return the (r_par, r_trans) bin centres of the blinding-template binning.

    Parameters
    ----------
    is_x_correlation : bool
        True for cross-correlations, whose r_par extends to negative values.

    Returns
    -------
    r_par_centres : numpy.ndarray
        Bin centres along the line of sight in h^-1 Mpc, shape (num_bins_r_par,).
    r_trans_centres : numpy.ndarray
        Bin centres across the line of sight in h^-1 Mpc, shape (num_bins_r_trans,).
    """
    r_par_min = -TEMPLATE_R_MAX if is_x_correlation else 0.0
    num_bins_r_par = int(round((TEMPLATE_R_MAX - r_par_min) / TEMPLATE_BIN_SIZE))
    num_bins_r_trans = int(round(TEMPLATE_R_MAX / TEMPLATE_BIN_SIZE))

    r_par_centres = r_par_min + (np.arange(num_bins_r_par) + 0.5) * TEMPLATE_BIN_SIZE
    r_trans_centres = (np.arange(num_bins_r_trans) + 0.5) * TEMPLATE_BIN_SIZE
    return r_par_centres, r_trans_centres


def read_blinding_template(filename, blind_corr_type):
    """Read one correlation of a BAO blinding template.

    Parameters
    ----------
    filename : str
        HDF5 blinding template.
    blind_corr_type : str
        Correlation key in the ``blinding`` group, e.g. ``'lyaxlya'`` or ``'lyaxqso'``.

    Returns
    -------
    numpy.ndarray
        Template Delta xi, stored as float-hex strings, shape (num_bins_r_par * num_bins_r_trans,),
        with r_trans varying fastest.
    """
    with h5py.File(filename, "r") as blinding_file:
        hex_template = np.array(blinding_file["blinding"][blind_corr_type]).astype(str)
    return np.array([float.fromhex(value) for value in hex_template])


def read_template_redshift_evolution(filename, blind_corr_type):
    """Read the fiducial redshift evolution of a BAO blinding template.

    Parameters
    ----------
    filename : str
        HDF5 blinding template, whose ``blinding`` group stores the attributes ``z_eff``,
        ``gamma_lya`` and ``gamma_qso`` written by the template script.
    blind_corr_type : str
        Correlation key, e.g. ``'lyaxlya'`` or ``'lyaxqso'``.

    Returns
    -------
    z_eff : float
        Redshift at which the template was computed.
    gamma_sum : float
        Sum of the bias-evolution exponents of the two tracers of the correlation
        (Lya for both legs of the auto-correlations, Lya and QSO for the crosses).

    Raises
    ------
    KeyError
        If the template does not store its redshift evolution.
    """
    with h5py.File(filename, "r") as blinding_file:
        attributes = blinding_file["blinding"].attrs
        missing = [key for key in ["z_eff", "gamma_lya", "gamma_qso"] if key not in attributes]
        if missing:
            raise KeyError(
                f"Blinding template {filename} does not store {missing}; it cannot be "
                "rescaled to the redshift of the measurement."
            )
        z_eff = float(attributes["z_eff"])
        gamma_lya = float(attributes["gamma_lya"])
        gamma_qso = float(attributes["gamma_qso"])

    if blind_corr_type in ["lyaxqso", "lybxqso"]:
        return z_eff, gamma_lya + gamma_qso
    return z_eff, 2 * gamma_lya


def compute_template_amplitude_ratio(z, z_eff, gamma_sum):
    """Amplitude of the correlation function at z relative to z_eff.

    The bias of each tracer evolves as (1 + z)^gamma and the growth factor follows the
    Einstein-de Sitter solution, D(z) proportional to 1 / (1 + z), so that

        A(z) / A(z_eff) = [(1 + z) / (1 + z_eff)]^(gamma_1 + gamma_2 - 2).

    This is the weighting picca uses for each tracer, ((1 + z) / (1 + z_ref))^(gamma - 1).
    Compared with the LCDM growth used in Vega, the difference is below 1% for 2 < z < 3.

    Parameters
    ----------
    z : float
        Effective redshift of the measurement.
    z_eff : float
        Redshift at which the template was computed.
    gamma_sum : float
        Sum of the bias-evolution exponents of the two tracers.

    Returns
    -------
    float
        A(z) / A(z_eff).
    """
    return ((1.0 + z) / (1.0 + z_eff)) ** (gamma_sum - 2.0)


def interpolate_blinding_template(template_grid, is_x_correlation, r_par, r_trans):
    """Interpolate the regular-grid blinding template onto arbitrary separations.

    A bicubic interpolating spline is built on the template bin centres. Separations outside
    the grid are evaluated at the nearest grid edge (constant extrapolation), which matches
    the FITPACK evaluation used previously; the template is negligible at these separations.

    Parameters
    ----------
    template_grid : numpy.ndarray
        Regular-grid template, shape (num_bins_r_par * num_bins_r_trans,), r_trans fastest.
    is_x_correlation : bool
        True for cross-correlations.
    r_par, r_trans : numpy.ndarray
        Separations along and across the line of sight in h^-1 Mpc, same shape.

    Returns
    -------
    numpy.ndarray
        Template evaluated at (r_par, r_trans), same shape as r_par.

    Raises
    ------
    ValueError
        If the template size does not match the regular grid.
    """
    r_par_centres, r_trans_centres = get_template_bin_centres(is_x_correlation)
    if template_grid.size != r_par_centres.size * r_trans_centres.size:
        raise ValueError(
            f"Regular-grid template has {template_grid.size} bins, expected "
            f"{r_par_centres.size} x {r_trans_centres.size}."
        )

    spline = scipy.interpolate.RectBivariateSpline(
        r_par_centres,
        r_trans_centres,
        template_grid.reshape(r_par_centres.size, r_trans_centres.size),
        kx=3,
        ky=3,
    )

    outside_grid = (
        (r_par < r_par_centres[0] - TEMPLATE_BIN_SIZE / 2)
        | (r_par > r_par_centres[-1] + TEMPLATE_BIN_SIZE / 2)
        | (r_trans > r_trans_centres[-1] + TEMPLATE_BIN_SIZE / 2)
    )
    if np.any(outside_grid):
        userprint(
            f"WARNING: {outside_grid.sum()} bins lie outside the blinding template grid; "
            "the template is evaluated at the nearest grid edge there."
        )

    r_par_clipped = np.clip(r_par, r_par_centres[0], r_par_centres[-1])
    r_trans_clipped = np.clip(r_trans, r_trans_centres[0], r_trans_centres[-1])
    return spline.ev(r_par_clipped, r_trans_clipped)


def main(cmdargs=None):
    """Export auto and cross-correlation for the fitter.

    Parameters
    ----------
    cmdargs : list of str or None, optional
        Command-line arguments. The default, None, reads the process arguments.

    Returns
    -------
    None
        Write the correlation, covariance, and distortion matrix to a FITS file.

    Notes
    -----
    The ``desi_dr3_civ`` strategy propagates unchanged for parameter blinding in
    Vega, without adding a correlation-blinding template during export.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export auto and cross-correlation for the fitter.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        required=True,
        help="Correlation produced via picca_cf.py, picca_xcf.py, ...",
    )

    parser.add_argument(
        "--out", type=str, default=None, required=True, help="Output file name"
    )

    parser.add_argument(
        "--dmat",
        type=str,
        default=None,
        required=False,
        help=(
            "Distortion matrix produced via picca_dmat.py, picca_xdmat.py... "
            "(if not provided will be identity)"
        ),
    )

    parser.add_argument(
        "--multipoles", action="store_true", help="Use multipole extension"
    )
    parser.add_argument(
        "--lmax-model",
        type=int,
        default=6,
        help=(
            "Max multipole ell for the model if rmu-binning"
            ". Note odd multipoles are removed from auto."
        ),
    )
    parser.add_argument(
        "--lmax-data",
        type=int,
        default=10,
        help=(
            "Max multipole ell for the output if rmu-binning"
            ". Note odd multipoles are removed from auto."
        ),
    )

    parser.add_argument(
        "--cov",
        type=str,
        default=None,
        required=False,
        help=(
            "Covariance matrix (if not provided will be calculated by " "subsampling)"
        ),
    )

    parser.add_argument(
        "--cor",
        type=str,
        default=None,
        required=False,
        help=(
            "Correlation matrix (if not provided will be calculated by " "subsampling)"
        ),
    )

    parser.add_argument(
        "--remove-shuffled-correlation",
        type=str,
        default=None,
        required=False,
        help="Remove a correlation from shuffling the distribution of los",
    )

    parser.add_argument(
        "--num-boot-cov",
        type=int,
        default=0,
        help="Number of bootstrap realizations. <=0 turns it off.",
    )
    parser.add_argument(
        "--do-not-smooth-cov",
        action="store_true",
        default=False,
        help="Do not smooth the covariance matrix",
    )
    parser.add_argument(
        "--smooth-per-r-par",
        action="store_true",
        default=False,
        help="Consider different correlation coefficients per r_par",
    )

    parser.add_argument(
        "--blind-corr-type",
        default=None,
        choices=["lyaxlya", "lyaxlyb", "qsoxlya", "qsoxlyb", "lyaxqso", "lybxqso"],
        help="Type of correlation. Required to apply blinding in DESI",
    )

    args = parser.parse_args(cmdargs)

    hdul = fitsio.FITS(args.data)

    r_par = np.array(hdul[1]["RP"][:])
    r_trans = np.array(hdul[1]["RT"][:])
    z = np.array(hdul[1]["Z"][:])
    num_pairs = np.array(hdul[1]["NB"][:])
    weights = np.array(hdul[2]["WE"][:])

    if "DA_BLIND" in hdul[2].get_colnames():
        xi = np.array(hdul[2]["DA_BLIND"][:])
        data_name = "DA_BLIND"
    else:
        xi = np.array(hdul[2]["DA"][:])
        data_name = "DA"

    head = hdul[1].read_header()
    num_bins_r_par = head["NP"]
    num_bins_r_trans = head["NT"]
    r_trans_max = head["RTMAX"]
    r_par_min = head["RPMIN"]
    r_par_max = head["RPMAX"]
    nsamples = xi.shape[0]

    is_x_correlation = r_par_min < 0

    # (r_par, r_trans) of the 2D bins for the blinding template, kept before r_par and r_trans
    # are replaced by multipole coordinates. In (r, mu) binning RP holds mu and RT holds r.
    rmu_binning = bool(head.get("RMU_BIN", False))
    if rmu_binning:
        r_par_template = r_trans * r_par
        r_trans_template = r_trans * np.sqrt(np.clip(1.0 - r_par**2, 0.0, None))
    else:
        r_par_template = r_par.copy()
        r_trans_template = r_trans.copy()

    # Effective redshift of the measurement for rescaling the blinding template: average of the
    # bin redshifts weighted by the summed pair weights (empty bins have zero weight and Z = 0)
    bin_weights = weights.sum(axis=0)
    z_measurement = None
    if np.sum(bin_weights) > 0:
        z_measurement = np.sum(bin_weights * z) / np.sum(bin_weights)

    if "BLINDING" in head:
        blinding = head["BLINDING"]
        if blinding == "minimal":
            blinding = "corr_yshift"
            userprint(
                "The minimal strategy is no longer supported."
                "Automatically switch to corr_yshift."
            )
    else:
        # if BLINDING keyword not present (old file), ignore blinding
        blinding = "none"
    hdul.close()

    if args.remove_shuffled_correlation is not None:
        hdul = fitsio.FITS(args.remove_shuffled_correlation)
        xi_shuffled = hdul["COR"][data_name][:]
        weight_shuffled = hdul["COR"]["WE"][:]
        xi_shuffled = (xi_shuffled * weight_shuffled).sum(axis=1)
        weight_shuffled = weight_shuffled.sum(axis=1)
        w = weight_shuffled > 0.0
        xi_shuffled[w] /= weight_shuffled[w]
        hdul.close()
        xi -= xi_shuffled[:, None]

    if args.multipoles:
        userprint("Smoothing is turned off for multipoles")
        args.do_not_smooth_cov = True
        assert head["RMU_BIN"]

        ells_out = np.arange(args.lmax_data + 1)
        if not is_x_correlation:
            ells_out = ells_out[ells_out % 2 == 0]

        nell_out = ells_out.size
        sum_weights = weights.sum(axis=0)
        sum_weights = sum_weights.reshape(num_bins_r_par, num_bins_r_trans)
        sum_sum_weights = sum_weights.sum(0)
        # The following assumes the same effective r and z.
        # Is there a multipole dependent weighting for effective r and z?
        r_trans = r_trans.reshape(num_bins_r_par, num_bins_r_trans)
        r_trans = np.sum(sum_weights * r_trans, 0) / sum_sum_weights
        r_trans = np.tile(r_trans, nell_out)

        z = z.reshape(num_bins_r_par, num_bins_r_trans)
        z = np.sum(sum_weights * z, 0) / sum_sum_weights
        z = np.tile(z, nell_out)

        num_pairs = num_pairs.reshape(num_bins_r_par, num_bins_r_trans).sum(0)
        num_pairs = np.tile(num_pairs, nell_out)
        r_par = np.repeat(ells_out, num_bins_r_trans)

        xi, weights = calculate_xi_ell(
            ells_out, xi, weights, num_bins_r_par, is_x_correlation
        )
        header_multipole = [
            {
                "name": "MLTPOLE",
                "value": True,
                "comment": "Specifying if in multipoles.",
            },
            {
                "name": "NL_DATA",
                "value": nell_out,
                "comment": "Number of output multipoles",
            },
        ]
    else:
        ells_out = None
        header_multipole = [
            {
                "name": "MLTPOLE",
                "value": False,
                "comment": "Specifying if in multipoles.",
            }
        ]

    if args.cov is not None:
        userprint(
            ("INFO: The covariance-matrix will be read from file: " "{}").format(
                args.cov
            )
        )
        hdul = fitsio.FITS(args.cov)
        covariance = hdul[1]["CO"][:]
        hdul.close()
    elif args.cor is not None:
        userprint(
            ("INFO: The correlation-matrix will be read from file: " "{}").format(
                args.cor
            )
        )
        hdul = fitsio.FITS(args.cor)
        correlation = hdul[1]["CO"][:]
        hdul.close()
        if (
            (correlation.min() < -1.0)
            or (correlation.min() > 1.0)
            or (correlation.max() < -1.0)
            or (correlation.max() > 1.0)
            or np.any(np.diag(correlation) != 1.0)
        ):
            userprint(("WARNING: The correlation-matrix has some incorrect " "values"))
        var = np.diagonal(correlation)
        correlation = correlation / np.sqrt(var * var[:, None])
        covariance = compute_cov(xi, weights)
        var = np.diagonal(covariance)
        covariance = correlation * np.sqrt(var * var[:, None])
    else:
        delta_r_par = (r_par_max - r_par_min) / num_bins_r_par
        delta_r_trans = (r_trans_max - 0.0) / num_bins_r_trans

        if args.num_boot_cov > 0:
            userprint(
                f"INFO: Covariance with {args.num_boot_cov} bootstrap realizations."
            )
            covariance = compute_cov_boot(xi, weights, nboots=args.num_boot_cov)
        else:
            covariance = compute_cov(xi, weights)

        if args.do_not_smooth_cov:
            userprint("INFO: The covariance will not be smoothed")
        else:
            userprint("INFO: The covariance will be smoothed")
            if args.smooth_per_r_par:
                userprint("INFO: with different correlation coefficients per r_par")
            covariance = smooth_cov(
                None,
                None,
                r_par,
                r_trans,
                delta_r_trans=delta_r_trans,
                delta_r_par=delta_r_par,
                covariance=covariance,
                per_r_par=args.smooth_per_r_par,
            )

    xi = (xi * weights).sum(axis=0)
    weights = weights.sum(axis=0)
    w = weights > 0
    xi[w] /= weights[w]

    try:
        scipy.linalg.cholesky(covariance)
    except scipy.linalg.LinAlgError:
        userprint("WARNING: Covariance matrix is not positive definite")

    if args.dmat is not None:
        hdul = fitsio.FITS(args.dmat)
        head_dmat = hdul[1].read_header()

        if data_name == "DA_BLIND" and "DM_BLIND" in hdul[1].get_colnames():
            dmat = np.array(hdul[1]["DM_BLIND"][:])
            dmat_name = "DM_BLIND"
        elif data_name == "DA_BLIND":
            userprint(
                "Blinded correlations were given but distortion matrix "
                "is unblinded. These files should not mix. Exiting..."
            )
            sys.exit(1)
        elif "DM_BLIND" in hdul[1].get_colnames():
            userprint(
                "Non-blinded correlations were given but distortion matrix "
                "is blinded. These files should not mix. Exiting..."
            )
            sys.exit(1)
        else:
            dmat = hdul[1]["DM"][:]
            dmat_name = "DM"

        try:
            r_par_dmat = hdul[2]["RP"][:]
            r_trans_dmat = hdul[2]["RT"][:]
            z_dmat = hdul[2]["Z"][:]
        except IOError:
            r_par_dmat = r_par.copy()
            r_trans_dmat = r_trans.copy()
            z_dmat = z.copy()
        if dmat.shape == (xi.size, xi.size):
            r_par_dmat = r_par.copy()
            r_trans_dmat = r_trans.copy()
            z_dmat = z.copy()
        hdul.close()

        if args.multipoles:
            assert head_dmat["RMU_BIN"]
            nmu_dmat = head_dmat["NP"] * head_dmat["COEFMOD"]
            nr_dmat = head_dmat["NT"] * head_dmat["COEFMOD"]
            dr_dmat = head_dmat["RTMAX"] / nr_dmat
            ells_model = np.arange(args.lmax_model + 1)
            if not is_x_correlation:
                ells_model = ells_model[ells_model % 2 == 0]
            nell_model = ells_model.size

            # From model multipoles to transverse-radial interpolation matrix.
            ell_to_tr_matrix = legvander(r_par_dmat, args.lmax_model)[:, ells_model]
            cols = np.floor(r_trans_dmat / dr_dmat).astype(int)
            cols = np.repeat(cols, nell_model) + np.tile(
                np.arange(nell_model) * nr_dmat, cols.size
            )
            rows = np.repeat(np.arange(dmat.shape[1]), nell_model)
            ell_to_tr_matrix = coo_array(
                (ell_to_tr_matrix.ravel(), (rows, cols)),
                shape=(dmat.shape[1], nell_model * nr_dmat),
            )

            r_par_dmat = np.repeat(ells_model, nr_dmat)
            r_trans_dmat = np.tile(
                r_trans_dmat.reshape(nmu_dmat, nr_dmat).mean(0), nell_model
            )
            z_dmat = np.tile(z_dmat.reshape(nmu_dmat, nr_dmat).mean(0), nell_model)
            dmat = ell_to_tr_matrix.T.dot(dmat.T).T

            del ell_to_tr_matrix, rows, cols

            # From transverse-radial binning to multipoles matrix
            assert xi.size == nell_out * num_bins_r_trans
            tr_to_ell_matrix = np.zeros((xi.size, dmat.shape[0]))

            leg_ells = get_legendre_bins(ells_out, num_bins_r_par, is_x_correlation)

            for i in range(xi.size):
                ell = i // num_bins_r_trans
                j1 = i % num_bins_r_trans
                tr_to_ell_matrix[i, j1::num_bins_r_trans] = leg_ells[ell]

            dmat = tr_to_ell_matrix.dot(dmat)
            del tr_to_ell_matrix

            header_multipole.append(
                {
                    "name": "NL_MODEL",
                    "value": nell_model,
                    "comment": "Number of model multipoles",
                }
            )
    else:
        dmat = np.eye(len(xi))
        r_par_dmat = r_par.copy()
        r_trans_dmat = r_trans.copy()
        z_dmat = z.copy()
        dmat_name = "DM_EMPTY"

    results = fitsio.FITS(args.out, "rw", clobber=True)
    header = [
        {
            "name": "BLINDING",
            "value": blinding,
            "comment": "String specifying the blinding strategy",
        },
        {"name": "RPMIN", "value": r_par_min, "comment": "Minimum r-parallel"},
        {"name": "RPMAX", "value": r_par_max, "comment": "Maximum r-parallel"},
        {"name": "RTMAX", "value": r_trans_max, "comment": "Maximum r-transverse"},
        {
            "name": "NP",
            "value": num_bins_r_par,
            "comment": "Number of bins in r-parallel",
        },
        {
            "name": "NT",
            "value": num_bins_r_trans,
            "comment": "Number of bins in r-transverse",
        },
        {"name": "ZMIN", "value": head["ZMIN"], "comment": "Minimum redshift of pairs"},
        {"name": "ZMAX", "value": head["ZMAX"], "comment": "Maximum redshift of pairs"},
        {
            "name": "OMEGAM",
            "value": head["OMEGAM"],
            "comment": "Omega_matter(z=0) of fiducial LambdaCDM cosmology",
        },
        {
            "name": "OMEGAR",
            "value": head["OMEGAR"],
            "comment": "Omega_radiation(z=0) of fiducial LambdaCDM cosmology",
        },
        {
            "name": "OMEGAK",
            "value": head["OMEGAK"],
            "comment": "Omega_k(z=0) of fiducial LambdaCDM cosmology",
        },
        {
            "name": "WL",
            "value": head["WL"],
            "comment": "Equation of state of dark energy of fiducial LambdaCDM cosmology",
        },
        {
            "name": "RMU_BIN",
            "value": "RMU_BIN" in head and head["RMU_BIN"] and not args.multipoles,
            "comment": "True if binned in r, mu",
        },
        {"name": "NSAMPLES", "value": nsamples, "comment": "Number of samples"},
    ] + header_multipole
    comment = [
        "R-parallel",
        "R-transverse",
        "Redshift",
        "Correlation",
        "Covariance matrix",
        "Distortion matrix",
        "Number of pairs",
    ]

    # Check if we need to unblind
    if blinding in UNBLINDABLE_STRATEGIES:
        userprint(f"'{blinding}' correlations are not blinded.")
        blinding = "none"
        data_name = "DA"
        dmat_name = "DM"

    requires_blinding = "BLIND" in data_name or blinding != "none"
    uses_parameter_blinding = blinding == "desi_dr3_civ"

    # CIV uses parameter blinding in Vega instead of correlation templates.
    if requires_blinding and not uses_parameter_blinding:
        if blinding in BLINDING_TEMPLATES:
            userprint(f"Blinding using seed for {blinding}")
        else:
            raise ValueError(
                f"Expected blinding to be one of {BLINDING_TEMPLATES.keys()}."
                f" Found {blinding}."
            )

        if args.blind_corr_type is None:
            raise ValueError("Blinding requires argument --blind-corr-type.")
        else:
            blind_corr_type = args.blind_corr_type
            # match name expected in blinding template file
            if blind_corr_type == "qsoxlya":
                blind_corr_type = "lyaxqso"
            if blind_corr_type == "qsoxlyb":
                blind_corr_type = "lybxqso"

        # Check that the correlation type agrees with the measured correlation
        if blind_corr_type in ["lyaxlya", "lyaxlyb"]:
            template_is_x_correlation = False
        elif blind_corr_type in ["lyaxqso", "lybxqso"]:
            template_is_x_correlation = True
        else:
            raise ValueError("Unknown correlation type: {}".format(blind_corr_type))
        if template_is_x_correlation != is_x_correlation:
            raise ValueError(
                f"--blind-corr-type {args.blind_corr_type} does not match the data, which is "
                f"{'a cross' if is_x_correlation else 'an auto'}-correlation."
            )

        blinding_filename = os.path.join(BLINDING_TEMPLATE_DIR, BLINDING_TEMPLATES[blinding])
        if not os.path.isfile(blinding_filename):
            raise RuntimeError(
                "Missing blinding file. Make sure you are running at"
                " NERSC or contact picca developers"
            )
        template = read_blinding_template(blinding_filename, blind_corr_type)

        # Interpolate the regular-grid template onto the bin centres of the measurement
        diff = interpolate_blinding_template(
            template, is_x_correlation, r_par_template, r_trans_template
        )

        # Rescale the template amplitude from its redshift to that of the measurement, with
        # the fiducial bias evolution and Einstein-de Sitter growth, so that the same BAO
        # shift is injected at every redshift
        template_z_eff, gamma_sum = read_template_redshift_evolution(
            blinding_filename, blind_corr_type
        )
        if z_measurement is None:
            raise ValueError("The correlation has no pairs; cannot blind it.")
        amplitude_ratio = compute_template_amplitude_ratio(
            z_measurement, template_z_eff, gamma_sum
        )
        userprint(
            f"Rescaling the blinding template from z = {template_z_eff:.3f} to "
            f"z = {z_measurement:.3f}"
        )
        diff *= amplitude_ratio
        header += [
            {
                "name": "BLINDZ",
                "value": z_measurement,
                "comment": "Redshift the blinding template is rescaled to",
            },
            {
                "name": "BLINDAMP",
                "value": amplitude_ratio,
                "comment": "Amplitude factor applied to the blinding template",
            },
        ]

        if args.multipoles:
            # Project the 2D (mu, r) template onto the multipoles of the data
            diff = calculate_xi_ell(
                ells_out,
                diff[None, :],
                np.ones_like(diff)[None, :],
                num_bins_r_par,
                is_x_correlation,
            )[0][0]

        # Check that the shapes match
        if np.shape(xi) != np.shape(diff):
            raise RuntimeError(
                "Unknown binning or wrong correlation type. Cannot blind."
                " Please raise an issue or contact picca developers."
            )

        # Add blinding
        xi = xi + diff

    results.write(
        [xi, r_par, r_trans, z, covariance, dmat, num_pairs],
        names=[data_name, "RP", "RT", "Z", "CO", dmat_name, "NB"],
        comment=comment,
        header=header,
        extname="COR",
    )
    comment = ["R-parallel model", "R-transverse model", "Redshift model"]
    results.write(
        [r_par_dmat, r_trans_dmat, z_dmat],
        names=["DMRP", "DMRT", "DMZ"],
        comment=comment,
        extname="DMATTRI",
    )
    results.close()


if __name__ == "__main__":
    cmdargs = sys.argv[1:]
    main(cmdargs)

if __name__ == "__main__":
    cmdargs = sys.argv[1:]
    main(cmdargs)
