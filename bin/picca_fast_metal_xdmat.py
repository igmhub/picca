#!/usr/bin/env python
"""Compute the metal matrices
"""
import sys
import os
import time
import argparse
import numpy as np
import fitsio

from picca import constants, xcf, io
from picca.utils import userprint


def read_stack_deltas_table(filename):
    """
    Read stack.

    Args:
        filename : std , path
    Returns:
        table as numpy.ndarray
    """
    return fitsio.read(filename, "STACK_DELTAS")


def calc_fast_metal_dmat(in_lambda_abs,
                         out_lambda_abs,
                         stack_table,
                         z_qso,
                         weight_qso,
                         rebin_factor=None):
    """Computes the metal distortion matrix.

    Args:
        in_lambda_abs : str
            Name of absorption in picca.constants in forest pixels from stack
            (input, i.e. 'true' absorber)
        out_lambda_abs : str
            Name of absorption in picca.constants in forest pixels from stack
            (output, i.e. 'assumed' absorber, usually LYA)
        stack_table: table
            table with cols LOGLAM and WEIGHT for first series of deltas
        z_qso : float 1D array
            QSO redshifts
        weight_qso : float 1D array
            QSO weights (as computed in picca.io.read_objects)

    Optionnal : rebin_factor
            rebin loglam and weights

    Returns:
        The distortion matrix data
        Note the global picca.xcf contains the cosmology, the rp grid, and the QSO catalog
    """

    loglam = stack_table["LOGLAM"]
    weight_forest = stack_table["WEIGHT"]
    if rebin_factor is not None:
        size = loglam.size
        loglam = loglam[:(size // rebin_factor) * rebin_factor].reshape(
            (size // rebin_factor), rebin_factor).mean(-1)
        weight_forest = weight_forest[:(size // rebin_factor) *
                                      rebin_factor].reshape(
                                          (size // rebin_factor),
                                          rebin_factor).mean(-1)

    # input
    input_zf = (10**loglam) / constants.ABSORBER_IGM[
        in_lambda_abs] - 1.  # redshift in the forest
    input_rf = xcf.cosmo.get_r_comov(input_zf)

    r_qso = xcf.cosmo.get_r_comov(z_qso)  # comoving distance for qso

    # all pairs
    input_rp = (
        input_rf[:, None] -
        r_qso[None, :]).ravel()  # same sign as line 528 of xcf.py (forest-qso)

    # output
    output_zf = (10**loglam) / constants.ABSORBER_IGM[out_lambda_abs] - 1.
    output_rf = xcf.cosmo.get_r_comov(output_zf)

    # all pairs
    output_rp = (
        output_rf[:, None] -
        r_qso[None, :]).ravel()  # same sign as line 528 of xcf.py (forest-qso)

    # weights
    # alpha_in: in (1+z)^(alpha_in-1) is a scaling used to model how the metal contribution
    # evolves with redshift (by default alpha=1 so that this has no effect)
    alpha_in = xcf.alpha_abs[in_lambda_abs]
    # alpha_out: (1+z)^(alpha_out-1) is applied to the delta weights in io.read_deltas and
    # used for the correlation function. It also has to be applied here.
    alpha_out = xcf.alpha_abs[out_lambda_abs]
    # so here we have to apply both scalings (in the original code : alpha_in is applied in
    # xcf.calc_metal_dmat and alpha_out in io.read_deltas)
    # qso weights have already been scaled with (1+z)^alpha_obj
    weights = ((weight_forest *
                ((1 + input_zf)**(alpha_in + alpha_out - 2)))[:, None] *
               weight_qso[None, :]).ravel()

    # distortion matrix
    rpbins = xcf.r_par_min + (
        xcf.r_par_max -
        xcf.r_par_min) / xcf.num_bins_r_par * np.arange(xcf.num_bins_r_par + 1)

    # I checked the orientation of the matrix
    dmat, _, _ = np.histogram2d(output_rp,
                                input_rp,
                                bins=(rpbins, rpbins),
                                weights=weights)

    # normalize (sum of weight should be one for each input rp,rt)
    sum_in_weight, _ = np.histogram(input_rp, bins=rpbins, weights=weights)
    dmat *= ((sum_in_weight > 0) / (sum_in_weight +
                                    (sum_in_weight == 0)))[None, :]

    # mean outputs
    sum_out_weight, _ = np.histogram(output_rp, bins=rpbins, weights=weights)
    sum_out_weight_rp, _ = np.histogram(output_rp,
                                        bins=rpbins,
                                        weights=weights *
                                        (output_rp[None, :].ravel()))

    # return the redshift of the actual absorber, which is the average of input_zf
    # and z_qso
    sum_out_weight_z, _ = np.histogram(
        output_rp,
        bins=rpbins,
        weights=weights *
        (((input_zf[:, None] + z_qso[None, :]) / 2.).ravel()))
    r_par_eff = sum_out_weight_rp / (sum_out_weight + (sum_out_weight == 0))
    z_eff = sum_out_weight_z / (sum_out_weight + (sum_out_weight == 0))

    # we could return the quantities computed as a function of rp only (and not rt):
    # return dmat, r_par_eff, r_trans_eff, z_eff
    # but for now we will return the full dmat to be consistent with the other computation
    # it consists in duplicating the result found to all rt, with output_rt = input_rt
    num_bins_total = xcf.num_bins_r_par * xcf.num_bins_r_trans

    full_dmat = np.zeros((num_bins_total, num_bins_total))
    full_r_par_eff = np.zeros(num_bins_total)
    full_r_trans_eff = np.zeros(num_bins_total)
    full_z_eff = np.zeros(num_bins_total)
    r_par_indices = np.arange(xcf.num_bins_r_par)
    r_trans = (0.5 + np.arange(
        xcf.num_bins_r_trans)) * xcf.r_trans_max / xcf.num_bins_r_trans
    for j in range(xcf.num_bins_r_trans):
        indices = j + xcf.num_bins_r_trans * r_par_indices
        for k, i in zip(indices, r_par_indices):
            full_dmat[indices, k] = dmat[r_par_indices, i]
        full_r_par_eff[indices] = r_par_eff
        full_z_eff[indices] = z_eff
        full_r_trans_eff[indices] = r_trans[j]

    return full_dmat, full_r_par_eff, full_r_trans_eff, full_z_eff


def main():
    """Compute the metal matrix of the cross-correlation delta x object for
     a list of IGM absorption."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Computes metal matrices for the cross-correlation '
                     'delta x object for a list of IGM absorption.'))

    parser.add_argument('--out',
                        type=str,
                        default=None,
                        required=True,
                        help='Output file name')

    parser.add_argument(
        '-i',
        '--in-attributes',
        type=str,
        default=None,
        required=True,
        help='Path to delta_attributes.fits.gz file with hdu STACK_DELTAS'
        ' containing table with at least rows "LOGLAM" and "WEIGHT"')

    parser.add_argument(
        '--delta-dir',
        type=str,
        default=None,
        required=False,
        help='Path to directory with delta*.gz to get the blinding info'
        ' (default is trying to guess from attributes file)')

    parser.add_argument('--drq',
                        type=str,
                        default=None,
                        required=True,
                        help='Catalog of objects in DRQ format')

    parser.add_argument('--mode',
                        type=str,
                        default='sdss',
                        choices=['sdss', 'desi', 'desi_mocks', 'desi_healpix'],
                        required=False,
                        help='type of catalog supplied, default sdss')

    parser.add_argument('--rp-min',
                        type=float,
                        default=-200.,
                        required=False,
                        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument('--rp-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument('--rt-max',
                        type=float,
                        default=200.,
                        required=False,
                        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument('--np',
                        type=int,
                        default=100,
                        required=False,
                        help='Number of r-parallel bins')

    parser.add_argument('--nt',
                        type=int,
                        default=50,
                        required=False,
                        help='Number of r-transverse bins')

    parser.add_argument(
        '--coef-binning-model',
        type=int,
        default=1,
        required=False,
        help=('Coefficient multiplying np and nt to get finner binning for the '
              'model'))

    parser.add_argument('--z-min-obj',
                        type=float,
                        default=0,
                        required=False,
                        help='Min redshift for object field')

    parser.add_argument('--z-max-obj',
                        type=float,
                        default=10,
                        required=False,
                        help='Max redshift for object field')

    parser.add_argument(
        '--z-cut-min',
        type=float,
        default=0.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift larger than '
              'z-cut-min'))

    parser.add_argument(
        '--z-cut-max',
        type=float,
        default=10.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift smaller than '
              'z-cut-max'))

    parser.add_argument('--z-min-sources',
                        type=float,
                        default=0.,
                        required=False,
                        help=('Limit the minimum redshift of the quasars '
                              'used as sources for spectra'))

    parser.add_argument('--z-max-sources',
                        type=float,
                        default=10.,
                        required=False,
                        help=('Limit the maximum redshift of the quasars '
                              'used as sources for spectra'))

    parser.add_argument(
        '--lambda-abs',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the delta'))

    parser.add_argument('--obj-name',
                        type=str,
                        default='QSO',
                        required=False,
                        help='Name of the object tracer')

    parser.add_argument(
        '--abs-igm',
        type=str,
        default=None,
        required=False,
        nargs='*',
        help='List of names of metal absorption in picca.constants')

    parser.add_argument('--z-ref',
                        type=float,
                        default=2.25,
                        required=False,
                        help='Reference redshift')

    parser.add_argument(
        '--z-evol-del',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol-obj',
        type=float,
        default=1.44,
        required=False,
        help='Exponent of the redshift evolution of the object field')

    parser.add_argument(
        '--metal-alpha',
        type=float,
        default=1.,
        required=False,
        help='Exponent of the redshift evolution of the metal delta field')

    parser.add_argument(
        '--fid-Om',
        type=float,
        default=0.315,
        required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-Or',
        type=float,
        default=0.,
        required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument('--fid-Ok',
                        type=float,
                        default=0.,
                        required=False,
                        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-wl',
        type=float,
        default=-1.,
        required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--rebin-factor',
        type=int,
        default=None,
        required=False,
        help='Rebin factor for deltas. If not None, deltas will '
        'be rebinned by that factor')
    parser.add_argument('--qso-z-bins',
                        type=int,
                        default=1000,
                        required=False,
                        help='Bins for the distribution of QSO redshifts')

    args = parser.parse_args()

    # setup variables in module xcf
    xcf.r_par_max = args.rp_max
    xcf.r_par_min = args.rp_min
    xcf.r_trans_max = args.rt_max
    xcf.z_cut_max = args.z_cut_max
    xcf.z_cut_min = args.z_cut_min
    xcf.num_bins_r_par = args.np * args.coef_binning_model
    xcf.num_bins_r_trans = args.nt * args.coef_binning_model
    xcf.num_model_bins_r_par = args.np * args.coef_binning_model
    xcf.num_model_bins_r_trans = args.nt * args.coef_binning_model
    xcf.z_ref = args.z_ref
    xcf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]

    xcf.alpha_abs = {}
    xcf.alpha_abs[args.lambda_abs] = args.z_evol_del
    for metal in args.abs_igm:
        xcf.alpha_abs[metal] = args.metal_alpha

    # read blinding keyword
    if args.delta_dir is None:
        args.delta_dir = os.path.dirname(args.in_attributes) + "/../Delta/"
        if not os.path.isdir(args.delta_dir):
            userprint(
                "Tried to guess the delta directory (containing the delta*.gz "
                f"files) but '{args.delta_dir}' is not valid"
            )
            userprint("Please specify the directory with option --delta-dir")
            sys.exit(1)
    blinding = io.read_blinding(args.in_dir, args.lambda_abs, args.obj_name)

    # load fiducial cosmology
    cosmo = constants.Cosmo(Om=args.fid_Om,
                            Or=args.fid_Or,
                            Ok=args.fid_Ok,
                            wl=args.fid_wl,
                            blinding=blinding)
    xcf.cosmo = cosmo

    t0 = time.time()

    ### Read data
    stack_table = read_stack_deltas_table(args.in_attributes)

    # read objets
    catalog = io.read_drq(args.drq,
                          z_min=args.z_min_obj,
                          z_max=args.z_max_obj,
                          keep_bal=True,
                          mode=args.mode)
    z_qso = catalog['Z']
    weight_qso = ((1. + z_qso) / (1. + args.z_ref))**(args.z_evol_obj - 1.)

    zbins = args.qso_z_bins
    userprint(f"Use histogram of QSO redshifts with {zbins} bins")
    histo_w, zbins = np.histogram(z_qso, bins=zbins, weights=weight_qso)
    histo_wz, _ = np.histogram(z_qso, bins=zbins, weights=weight_qso * z_qso)
    selection = histo_w > 0
    z_qso = histo_wz[selection] / histo_w[selection]  # weighted mean in bins
    weight_qso = histo_w[selection]

    t1 = time.time()
    userprint(
        f'picca_metal_xdmat.py - Time reading data: {(t1-t0)/60:.3f} minutes')

    # intitialize arrays to store the results for the different metal absorption
    dmat_all = []
    r_par_all = []
    r_trans_all = []
    z_all = []
    names = []

    # loop over metals
    for index, abs_igm in enumerate(args.abs_igm):

        userprint("Computing", abs_igm)

        # this a matrix as a function of rp only
        dmat, r_par_eff, r_trans_eff, z_eff = calc_fast_metal_dmat(
            abs_igm,
            args.lambda_abs,
            stack_table,
            z_qso,
            weight_qso,
            rebin_factor=args.rebin_factor)

        # add these results to the list ofor the different metal absorption
        dmat_all.append(dmat)
        r_par_all.append(r_par_eff)
        r_trans_all.append(r_trans_eff)
        z_all.append(z_eff)
        names.append(abs_igm)

    t2 = time.time()
    userprint(
        f'picca_metal_xdmat.py - Time computing all metal matrix: {(t2-t1)/60:.3f} minutes'
    )

    # save the results
    results = fitsio.FITS(args.out, 'rw', clobber=True)
    header = [{
        'name': 'RPMIN',
        'value': xcf.r_par_min,
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RPMAX',
        'value': xcf.r_par_max,
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'RTMAX',
        'value': xcf.r_trans_max,
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    }, {
        'name': 'NP',
        'value': xcf.num_bins_r_par,
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NT',
        'value': xcf.num_bins_r_trans,
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'COEFMOD',
        'value': args.coef_binning_model,
        'comment': 'Coefficient for model binning'
    }, {
        'name': 'ZCUTMIN',
        'value': xcf.z_cut_min,
        'comment': 'Minimum redshift of pairs'
    }, {
        'name': 'ZCUTMAX',
        'value': xcf.z_cut_max,
        'comment': 'Maximum redshift of pairs'
    }, {
        'name': 'OMEGAM',
        'value': args.fid_Om,
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAR',
        'value': args.fid_Or,
        'comment': 'Omega_radiation(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': 'OMEGAK',
        'value': args.fid_Ok,
        'comment': 'Omega_k(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name':
            'WL',
        'value':
            args.fid_wl,
        'comment':
            'Equation of state of dark energy of fiducial LambdaCDM cosmology'
    }, {
        'name': "BLINDING",
        'value': blinding,
        'comment': 'String specifying the blinding strategy'
    }]
    len_names = np.array([len(name) for name in names]).max()
    names = np.array(names, dtype='S' + str(len_names))
    results.write(
        [np.array(names)],
        names=['ABS_IGM'],
        header=header,
        comment=['Number of pairs', 'Number of used pairs', 'Absorption name'],
        extname='ATTRI')

    dmat_name = "DM_"
    if blinding != "none":
        dmat_name += "BLIND_"
    names = names.astype(str)
    out_list = []
    out_names = []
    out_comment = []
    out_units = []
    for index, name in enumerate(names):
        out_names += ['RP_' + args.obj_name + '_' + name]
        out_list += [r_par_all[index]]
        out_comment += ['R-parallel']
        out_units += ['h^-1 Mpc']

        out_names += ['RT_' + args.obj_name + '_' + name]
        out_list += [r_trans_all[index]]
        out_comment += ['R-transverse']
        out_units += ['h^-1 Mpc']

        out_names += ['Z_' + args.obj_name + '_' + name]
        out_list += [z_all[index]]
        out_comment += ['Redshift']
        out_units += ['']

        out_names += [dmat_name + args.obj_name + '_' + name]
        out_list += [dmat_all[index]]
        out_comment += ['Distortion matrix']
        out_units += ['']

    results.write(out_list,
                  names=out_names,
                  comment=out_comment,
                  units=out_units,
                  extname='MDMAT')
    results.close()

    t3 = time.time()
    userprint(f'picca_metal_xdmat.py - Time total: {(t3-t0)/60:.3f} minutes')


if __name__ == '__main__':
    main()
