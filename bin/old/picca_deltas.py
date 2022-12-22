#!/usr/bin/env python
"""Computes delta field from a list of spectra.

Computes the mean transmission fluctuation field (delta field) for a list of
spectra for the specified absorption line. Follow the procedure described in
section 2.4 of du Mas des Bourboux et al. 2020 (In prep).
"""
import sys
import os
import time
import multiprocessing
from multiprocessing import Pool
import argparse
import fitsio
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

from picca.data import Forest, Delta
from picca import prep_del, io, constants, bal_tools
from picca.utils import userprint
from picca.constants import ACCEPTED_BLINDING_STRATEGIES
import warnings


def cont_fit(forests):
    """ Computes the quasar continua for all the forests in data

    Args:
        forests: a list of forest instances
    Returns:
        the list of forests after having computed their quasar continua
    """
    for forest in forests:
        forest.cont_fit()
    return forests


def get_metadata(data):
    ''' Constructs an astropy.table from all forests' metadata
    '''
    tab = Table()
    # TODO: change mean_snr_save to mean_snr once this is properly treated
    # in data.py
    for field in [
            'ra', 'dec', 'z_qso', 'thingid', 'plate', 'mjd', 'fiberid',
            'mean_snr_save', 'p0', 'p1'
    ]:
        column_values = []
        for healpix in data:
            for forest in data[healpix]:
                if field in forest.__dict__ and not forest.__dict__[
                        field] is None:
                    column_values.append(forest.__dict__[field])
                else:
                    column_values.append(0)
        tab[field] = np.array(column_values)

    mean_delta2_values = []
    for healpix in data:
        for forest in data[healpix]:
            if "delta" in forest.__dict__ and not forest.delta is None:
                mean_delta2 = np.average(forest.delta*forest.delta, weights=forest.ivar)
                mean_delta2_values.append(mean_delta2)
            else:
                mean_delta2_values.append(-100)
    tab["mean_delta2"] = np.array(mean_delta2_values)

    npix = []
    for healpix in data:
        for forest in data[healpix]:
            if forest.log_lambda is None:
                npix.append(0)
            else:
                npix.append(forest.log_lambda.size)
    tab['npixels'] = np.array(npix)

    return tab


def get_delta_from_forest(forest,
                          get_stack_delta,
                          get_var_lss,
                          get_eta,
                          get_fudge,
                          use_mock_cont=False):
    """Computes delta field from forest

    Args:
        forest: Forest
            A forest instance from which to initialize the deltas
        get_stack_delta: function
            Interpolates the stacked delta field for a given redshift.
        get_var_lss: Interpolates the pixel variance due to the Large Scale
            Strucure on the wavelength array.
        get_eta: Interpolates the correction factor to the contribution of the
            pipeline estimate of the instrumental noise to the variance on the
            wavelength array.
        get_fudge: Interpolates the fudge contribution to the variance on the
            wavelength array.
        use_mock_cont: bool - default: False
            Flag to use the mock continuum to compute the mean expected
            flux fraction
    """
    log_lambda = forest.log_lambda
    stack_delta = get_stack_delta(log_lambda)
    var_lss = get_var_lss(log_lambda)
    eta = get_eta(log_lambda)
    fudge = get_fudge(log_lambda)

    #-- if use_mock_cont is True use the mock continuum to compute the mean
    #-- expected flux fraction
    if use_mock_cont:
        mean_expected_flux_frac = forest.mean_expected_flux_frac
    else:
        mean_expected_flux_frac = forest.cont * stack_delta
    delta = forest.flux / mean_expected_flux_frac - 1.
    var_pipe = 1. / forest.ivar / mean_expected_flux_frac**2
    variance = eta * var_pipe + var_lss + fudge / var_pipe
    weights = 1. / variance
    exposures_diff = forest.exposures_diff
    if forest.exposures_diff is not None:
        exposures_diff /= mean_expected_flux_frac
    ivar = forest.ivar / (eta + (eta == 0)) * (mean_expected_flux_frac**2)

    forest.delta = delta
    forest.weights = weights
    forest.exposures_diff = exposures_diff
    forest.ivar = ivar


def main(cmdargs):
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Computes delta field"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the delta field '
                     'from a list of spectra'))

    parser.add_argument('--out-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Output directory')

    parser.add_argument('--drq',
                        type=str,
                        default=None,
                        required=True,
                        help='Catalog of objects in DRQ format')

    parser.add_argument('--in-dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Directory to spectra files')

    parser.add_argument('--log',
                        type=str,
                        default='input.log',
                        required=False,
                        help='Log input data')

    parser.add_argument('--iter-out-prefix',
                        type=str,
                        default='iter',
                        required=False,
                        help='Prefix of the iteration file')

    parser.add_argument('--mode',
                        type=str,
                        choices=['pix', 'spec','spcframe','spplate','desi',
                                 'desi_healpix','desi_survey_tilebased',
                                 'desi_sv_no_coadd','desi_mocks','desiminisv'],
                        default='pix',
                        required=False,
                        help=('''Open mode of the spectra files: pix, spec,
                              spcframe, spplate, desi_mocks (formerly known as desi),
                              desi_healpix (for healpix based coadded data),
                              desi_survey_tilebased (for tilebased data with coadding),
                              desi_sv_no_coadd (without coadding across tiles, will output in tile format)'''))

    parser.add_argument('--best-obs',
                        action='store_true',
                        required=False,
                        help=('If mode == spcframe, then use only the best '
                              'observation'))

    parser.add_argument('--single-exp',
                        action='store_true',
                        required=False,
                        help=('If mode == spcframe, then use only one of the '
                              'available exposures. If best-obs then choose it '
                              'among those contributing to the best obs'))

    parser.add_argument('--zqso-min',
                        type=float,
                        default=None,
                        required=False,
                        help='Lower limit on quasar redshift from drq')

    parser.add_argument('--zqso-max',
                        type=float,
                        default=None,
                        required=False,
                        help='Upper limit on quasar redshift from drq')

    parser.add_argument('--keep-bal',
                        action='store_true',
                        required=False,
                        help='Do not reject BALs in drq')

    parser.add_argument('--bi-max',
                        type=float,
                        required=False,
                        default=None,
                        help=('Maximum CIV balnicity index in drq (overrides '
                              '--keep-bal)'))

    parser.add_argument('--lambda-min',
                        type=float,
                        default=3600.,
                        required=False,
                        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max',
                        type=float,
                        default=5500.,
                        required=False,
                        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min',
                        type=float,
                        default=1040.,
                        required=False,
                        help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max',
                        type=float,
                        default=1200.,
                        required=False,
                        help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--rebin',
                        type=int,
                        default=3,
                        required=False,
                        help=('Rebin wavelength grid by combining this number '
                              'of adjacent pixels (ivar weight)'))

    parser.add_argument('--npix-min',
                        type=int,
                        default=50,
                        required=False,
                        help='Minimum of rebined pixels')

    parser.add_argument('--dla-vac',
                        type=str,
                        default=None,
                        required=False,
                        help='DLA catalog file')

    parser.add_argument('--dla-mask',
                        type=float,
                        default=0.8,
                        required=False,
                        help=('Lower limit on the DLA transmission. '
                              'Transmissions below this number are masked'))

    parser.add_argument('--absorber-vac',
                        type=str,
                        default=None,
                        required=False,
                        help='Absorber catalog file')

    parser.add_argument('--absorber-mask',
                        type=float,
                        default=2.5,
                        required=False,
                        help=('Mask width on each side of the absorber central '
                              'observed wavelength in units of '
                              '1e4*dlog10(lambda)'))

    parser.add_argument('--mask-file',
                        type=str,
                        default=None,
                        required=False,
                        help=('Path to file to mask regions in lambda_OBS and '
                              'lambda_RF. In file each line is: region_name '
                              'region_min region_max (OBS or RF) [Angstrom]'))

    parser.add_argument('--optical-depth',
                        type=str,
                        default=None,
                        required=False,
                        nargs='*',
                        help=('Correct for the optical depth: tau_1 gamma_1 '
                              'absorber_1 tau_2 gamma_2 absorber_2 ...'))

    parser.add_argument('--dust-map',
                        type=str,
                        default=None,
                        required=False,
                        help=('Path to DRQ catalog of objects for dust map to '
                              'apply the Schlegel correction'))

    parser.add_argument('--flux-calib',
                        type=str,
                        default=None,
                        required=False,
                        help=('Path to previously produced picca_delta.py file '
                              'to correct for multiplicative errors in the '
                              'pipeline flux calibration'))

    parser.add_argument('--ivar-calib',
                        type=str,
                        default=None,
                        required=False,
                        help=('Path to previously produced picca_delta.py file '
                              'to correct for multiplicative errors in the '
                              'pipeline inverse variance calibration'))

    parser.add_argument('--eta-min',
                        type=float,
                        default=0.5,
                        required=False,
                        help='Lower limit for eta')

    parser.add_argument('--eta-max',
                        type=float,
                        default=1.5,
                        required=False,
                        help='Upper limit for eta')

    parser.add_argument('--vlss-min',
                        type=float,
                        default=0.,
                        required=False,
                        help='Lower limit for variance LSS')

    parser.add_argument('--vlss-max',
                        type=float,
                        default=0.3,
                        required=False,
                        help='Upper limit for variance LSS')

    parser.add_argument('--delta-format',
                        type=str,
                        default=None,
                        required=False,
                        help='Format for Pk 1D: Pk1D')

    parser.add_argument('--use-ivar-as-weight',
                        action='store_true',
                        default=False,
                        help=('Use ivar as weights (implemented as eta = 1, '
                              'sigma_lss = fudge = 0)'))

    parser.add_argument('--use-constant-weight',
                        action='store_true',
                        default=False,
                        help=('Set all the delta weights to one (implemented '
                              'as eta = 0, sigma_lss = 1, fudge = 0)'))

    parser.add_argument('--order',
                        type=int,
                        default=1,
                        required=False,
                        help=('Order of the log10(lambda) polynomial for the '
                              'continuum fit, by default 1.'))

    parser.add_argument('--nit',
                        type=int,
                        default=5,
                        required=False,
                        help=('Number of iterations to determine the mean '
                              'continuum shape, LSS variances, etc.'))

    parser.add_argument('--nproc',
                        type=int,
                        default=None,
                        required=False,
                        help='Number of processors')

    parser.add_argument('--nspec',
                        type=int,
                        default=None,
                        required=False,
                        help='Maximum number of spectra to read')

    parser.add_argument('--use-mock-continuum',
                        action='store_true',
                        default=False,
                        help='use the mock continuum for computing the deltas')

    parser.add_argument('--spall',
                        type=str,
                        default=None,
                        required=False,
                        help=('Path to spAll file'))

    parser.add_argument('--bal-catalog',
                        type=str,
                        default=None,
                        required=False,
                        help=('BAL catalog location, used if BAL information is'
                            ' not included in the quasar catalog.  Use with '
                            '--keep-bal to mask BAL features'))

    parser.add_argument('--metadata',
                        type=str,
                        default='metadata.fits',
                        required=False,
                        help=('Name for table containing forests metadata'))

    parser.add_argument('--survey',
                        type=str.lower,
                        choices=('desi','eboss'),
                        default='desi',
                        required=False,
                        help=('Survey the catalog comes from. Defines which '
                            'naming conventions to use when masking BALs.'))

    parser.add_argument('--use-single-nights',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Use individual night for input spectra (DESI SV)')

    parser.add_argument('--use-all',
                        action='store_true',
                        default=False,
                        required=False,
                        help=('Use all dir for input spectra (DESI SV)'))

    parser.add_argument('--blinding-desi',
                        type=str,
                        default="corr_yshift",
                        required=False,
                        help='Blinding strategy. "none" for no blinding')

    t0 = time.time()

    args = parser.parse_args(cmdargs)

    assert (args.blinding_desi in ACCEPTED_BLINDING_STRATEGIES)

    # comment this when ready to unblind
    if args.blinding_desi == "none":
        userprint("WARINING: --blinding-desi is being ignored. 'corr_yshift' blinding engaged")
        args.blinding_desi = "corr_yshift"

    # setup forest class variables
    Forest.log_lambda_min = np.log10(args.lambda_min)
    Forest.log_lambda_max = np.log10(args.lambda_max)
    Forest.log_lambda_min_rest_frame = np.log10(args.lambda_rest_min)
    Forest.log_lambda_max_rest_frame = np.log10(args.lambda_rest_max)
    Forest.rebin = args.rebin
    Forest.delta_log_lambda = args.rebin * 1e-4
    # minumum dla transmission
    Forest.dla_mask_limit = args.dla_mask
    Forest.absorber_mask_width = args.absorber_mask

    # Find the redshift range
    if args.zqso_min is None:
        args.zqso_min = max(0., args.lambda_min / args.lambda_rest_max - 1.)
        userprint("zqso_min = {}".format(args.zqso_min))
    if args.zqso_max is None:
        args.zqso_max = max(0., args.lambda_max / args.lambda_rest_min - 1.)
        userprint("zqso_max = {}".format(args.zqso_max))

    #-- Create interpolators for mean quantities, such as
    #-- Large-scale structure variance : var_lss
    #-- Pipeline ivar correction error: eta
    #-- Pipeline ivar correction term : fudge
    #-- Mean continuum : mean_cont
    log_lambda_temp = (Forest.log_lambda_min + np.arange(2) *
                       (Forest.log_lambda_max - Forest.log_lambda_min))
    log_lambda_rest_frame_temp = (
        Forest.log_lambda_min_rest_frame + np.arange(2) *
        (Forest.log_lambda_max_rest_frame - Forest.log_lambda_min_rest_frame))
    Forest.get_var_lss = interp1d(log_lambda_temp,
                                  0.2 + np.zeros(2),
                                  fill_value="extrapolate",
                                  kind="nearest")
    Forest.get_eta = interp1d(log_lambda_temp,
                              np.ones(2),
                              fill_value="extrapolate",
                              kind="nearest")
    Forest.get_fudge = interp1d(log_lambda_temp,
                                np.zeros(2),
                                fill_value="extrapolate",
                                kind="nearest")
    Forest.get_mean_cont = interp1d(log_lambda_rest_frame_temp, 1 + np.zeros(2))


    #-- Check that the order of the continuum fit is 0 (constant) or 1 (linear).
    if args.order:
        if (args.order != 0) and (args.order != 1):
            userprint(("ERROR : invalid value for order, must be eqal to 0 or"
                       "1. Here order = {:d}").format(args.order))
            sys.exit(12)

    #-- Correct multiplicative pipeline flux calibration
    if args.flux_calib is not None:
        hdu = fitsio.read(args.flux_calib, ext=1)
        stack_log_lambda = hdu['loglam']
        stack_delta = hdu['stack']
        w = (stack_delta != 0.)
        Forest.correct_flux = interp1d(stack_log_lambda[w],
                                       stack_delta[w],
                                       fill_value="extrapolate",
                                       kind="nearest")

    #-- Correct multiplicative pipeline inverse variance calibration
    if args.ivar_calib is not None:
        hdu = fitsio.read(args.ivar_calib, ext=2)
        log_lambda = hdu['loglam']
        eta = hdu['eta']
        Forest.correct_ivar = interp1d(log_lambda,
                                       eta,
                                       fill_value="extrapolate",
                                       kind="nearest")

    ### Apply dust correction
    if not args.dust_map is None:
        userprint("applying dust correction")
        Forest.extinction_bv_map = io.read_dust_map(args.dust_map)

    log_file = open(os.path.expandvars(args.log), 'w')

    # Read data
    (data, num_data, nside,
     healpy_pix_ordering,
     blinding) = io.read_data(os.path.expandvars(args.in_dir),
                              args.drq,
                              args.mode,
                              z_min=args.zqso_min,
                              z_max=args.zqso_max,
                              max_num_spec=args.nspec,
                              log_file=log_file,
                              keep_bal=args.keep_bal,
                              bi_max=args.bi_max,
                              best_obs=args.best_obs,
                              single_exp=args.single_exp,
                              pk1d=args.delta_format,
                              spall=args.spall,
                              useall=args.use_all,
                              usesinglenights=args.use_single_nights,
                              blinding_desi=args.blinding_desi)

     #-- Add order info
    for pix in data:
        for forest in data[pix]:
            if not forest is None:
                forest.order = args.order

    ### Read masks
    if args.mask_file is not None:
        args.mask_file = os.path.expandvars(args.mask_file)
        try:
            mask = Table.read(args.mask_file,
                              names=('type', 'wave_min', 'wave_max', 'frame'),
                              format='ascii')
            mask['log_wave_min'] = np.log10(mask['wave_min'])
            mask['log_wave_max'] = np.log10(mask['wave_max'])
        except (OSError, ValueError):
            userprint(("ERROR: Error while reading mask_file "
                       "file {}").format(args.mask_file))
            sys.exit(1)
    else:
        mask = Table(names=('type', 'wave_min', 'wave_max', 'frame',
                            'log_wave_min', 'log_wave_max'))

    ### Mask lines
    for healpix in data:
        for forest in data[healpix]:
            forest.mask(mask)

    ### Mask absorbers
    if not args.absorber_vac is None:
        userprint("INFO: Adding absorbers")
        absorbers = io.read_absorbers(args.absorber_vac)
        num_absorbers = 0
        for healpix in data:
            for forest in data[healpix]:
                if forest.thingid in absorbers:
                    for lambda_absorber in absorbers[forest.thingid]:
                        forest.add_absorber(lambda_absorber)
                        num_absorbers += 1
        log_file.write("Found {} absorbers in forests\n".format(num_absorbers))

    ### Add optical depth contribution
    if not args.optical_depth is None:
        userprint(("INFO: Adding {} optical"
                   "depths").format(len(args.optical_depth) // 3))
        assert len(args.optical_depth) % 3 == 0
        for index in range(len(args.optical_depth) // 3):
            tau = float(args.optical_depth[3 * index])
            gamma = float(args.optical_depth[3 * index + 1])
            lambda_rest_frame = constants.ABSORBER_IGM[args.optical_depth[
                3 * index + 2]]
            userprint(
                ("INFO: Adding optical depth for tau = {}, gamma = {}, "
                 "lambda_rest_frame = {} A").format(tau, gamma,
                                                    lambda_rest_frame))
            for healpix in data:
                for forest in data[healpix]:
                    forest.add_optical_depth(tau, gamma, lambda_rest_frame)

    ### Mask DLAs
    if not args.dla_vac is None:
        userprint("INFO: Adding DLAs")
        if 'desi' in args.mode:
            dlas= io.read_dlas(args.dla_vac, obj_id_name='TARGETID')
        else:
            dlas = io.read_dlas(args.dla_vac)
        num_dlas = 0
        for healpix in data:
            for forest in data[healpix]:
                if forest.thingid in dlas:
                    for dla in dlas[forest.thingid]:
                        forest.add_dla(dla[0], dla[1], mask)
                        num_dlas += 1
        log_file.write("Found {} DLAs in forests\n".format(num_dlas))

    ### Mask BALs
    if 'desi' in args.mode:
            bal_catalog_to_read = args.drq
    else:
            bal_catalog_to_read = args.bal_catalog
    if args.keep_bal is True:
        userprint("INFO: Masking BALs")
        bal_cat = bal_tools.read_bal(bal_catalog_to_read,args.mode)
        num_bal = 0
        for healpix in data:
            for forest in data[healpix]:
                bal_mask = bal_tools.add_bal_mask(bal_cat, forest.thingid,
                        args.mode)
                forest.mask(bal_mask)
            if len(bal_mask) > 0:
                    num_bal += 1
        log_file.write("Found {} BAL quasars in forests\n".format(num_bal))

    ## Apply cuts
    log_file.write(
        ("INFO: Input sample has {} "
         "forests\n").format(np.sum([len(forest) for forest in data.values()])))
    remove_keys = []
    for healpix in data:
        forests = []
        for forest in data[healpix]:
            if ((forest.log_lambda is None) or
                    len(forest.log_lambda) < args.npix_min):
                if forest.log_lambda is None:
                    forest_size = 0
                else:
                    forest_size = len(forest.log_lambda)
                log_file.write(("INFO: Rejected {} due to forest too "
                                "short ({})\n").format(forest.thingid,
                                                       forest_size))
                continue

            if np.isnan((forest.flux * forest.ivar).sum()):
                log_file.write(("INFO: Rejected {} due to nan "
                                "found\n").format(forest.thingid))
                continue

            if (args.use_constant_weight and
                (forest.flux.mean() <= 0.0 or forest.mean_snr <= 1.0)):
                log_file.write(("INFO: Rejected {} due to negative mean or "
                                "too low SNR found\n").format(forest.thingid))
                continue

            forests.append(forest)
            log_file.write("{} {}-{}-{} accepted\n".format(
                forest.thingid, forest.plate, forest.mjd, forest.fiberid))
        data[healpix][:] = forests
        if len(data[healpix]) == 0:
            remove_keys += [healpix]

    for healpix in remove_keys:
        del data[healpix]

    num_forests = np.sum([len(forest) for forest in data.values()])
    log_file.write(("INFO: Remaining sample has {} "
                    "forests\n").format(num_forests))
    userprint(f"Remaining sample has {num_forests} forests")

    # Sanity check: all forests must have the attribute log_lambda
    for healpix in data:
        for forest in data[healpix]:
            assert forest.log_lambda is not None

    t1 = time.time()
    tmin = (t1 - t0) / 60
    userprint('INFO: time elapsed to read data', tmin, 'minutes')

    # compute fits to the forests iteratively
    # (see equations 2 to 4 in du Mas des Bourboux et al. 2020)
    num_iterations = args.nit
    for iteration in range(num_iterations):
        context = multiprocessing.get_context('fork')
        pool = context.Pool(processes=args.nproc)
        userprint(
            f"Continuum fitting: starting iteration {iteration} of {num_iterations}"
        )

        #-- Sorting healpix pixels before giving to pool (for some reason)
        pixels = np.array([k for k in data])
        sort = pixels.argsort()
        sorted_data = [data[k] for k in pixels[sort]]
        data_fit_cont = pool.map(cont_fit, sorted_data)
        for index, healpix in enumerate(pixels[sort]):
            data[healpix] = data_fit_cont[index]

        userprint(
            f"Continuum fitting: ending iteration {iteration} of {num_iterations}"
        )

        pool.close()

        if iteration < num_iterations - 1:
            #-- Compute mean continuum (stack in rest-frame)
            (log_lambda_rest_frame, mean_cont,
             mean_cont_weight) = prep_del.compute_mean_cont(data)
            w = mean_cont_weight > 0.
            log_lambda_cont = log_lambda_rest_frame[w]
            new_cont = Forest.get_mean_cont(log_lambda_cont) * mean_cont[w]
            Forest.get_mean_cont = interp1d(log_lambda_cont,
                                            new_cont,
                                            fill_value="extrapolate")

            #-- Compute observer-frame mean quantities (var_lss, eta, fudge)
            if not (args.use_ivar_as_weight or args.use_constant_weight):
                (log_lambda, eta, var_lss, fudge, num_pixels, var_pipe_values,
                 var_delta, var2_delta, count, num_qso, chi2_in_bin, error_eta,
                 error_var_lss, error_fudge) = prep_del.compute_var_stats(
                     data, (args.eta_min, args.eta_max),
                     (args.vlss_min, args.vlss_max))
                w = num_pixels > 0
                Forest.get_eta = interp1d(log_lambda[w],
                                          eta[w],
                                          fill_value="extrapolate",
                                          kind="nearest")
                Forest.get_var_lss = interp1d(log_lambda[w],
                                              var_lss[w],
                                              fill_value="extrapolate",
                                              kind="nearest")
                Forest.get_fudge = interp1d(log_lambda[w],
                                            fudge[w],
                                            fill_value="extrapolate",
                                            kind="nearest")
            else:
                num_bins = 10  # this value is arbitrary
                log_lambda = (
                    Forest.log_lambda_min + (np.arange(num_bins) + .5) *
                    (Forest.log_lambda_max - Forest.log_lambda_min) / num_bins)

                if args.use_ivar_as_weight:
                    userprint(("INFO: using ivar as weights, skipping eta, "
                               "var_lss, fudge fits"))
                    eta = np.ones(num_bins)
                    var_lss = np.zeros(num_bins)
                    fudge = np.zeros(num_bins)
                else:
                    userprint(("INFO: using constant weights, skipping eta, "
                               "var_lss, fudge fits"))
                    eta = np.zeros(num_bins)
                    var_lss = np.ones(num_bins)
                    fudge = np.zeros(num_bins)

                error_eta = np.zeros(num_bins)
                error_var_lss = np.zeros(num_bins)
                error_fudge = np.zeros(num_bins)
                chi2_in_bin = np.zeros(num_bins)

                num_pixels = np.zeros(num_bins)
                var_pipe_values = np.zeros(num_bins)
                var_delta = np.zeros((num_bins, num_bins))
                var2_delta = np.zeros((num_bins, num_bins))
                count = np.zeros((num_bins, num_bins))
                num_qso = np.zeros((num_bins, num_bins))

                Forest.get_eta = interp1d(log_lambda,
                                          eta,
                                          fill_value='extrapolate',
                                          kind='nearest')
                Forest.get_var_lss = interp1d(log_lambda,
                                              var_lss,
                                              fill_value='extrapolate',
                                              kind='nearest')
                Forest.get_fudge = interp1d(log_lambda,
                                            fudge,
                                            fill_value='extrapolate',
                                            kind='nearest')

        stack_log_lambda, stack_delta, stack_weight = prep_del.stack(data)
        get_stack_delta = interp1d(stack_log_lambda[stack_weight > 0.],
                                   stack_delta[stack_weight > 0.],
                                   kind="nearest",
                                   fill_value="extrapolate")
        get_stack_delta_weights = interp1d(stack_log_lambda[stack_weight > 0.],
                                           stack_weight[stack_weight > 0.],
                                           kind="nearest",
                                           fill_value=0.0,
                                           bounds_error=False)

        ### Save iter_out_prefix
        delta_attrib_name = args.iter_out_prefix
        if iteration == num_iterations - 1:
            delta_attrib_name += ".fits.gz"
        else:
            delta_attrib_name += "_iteration{}.fits.gz".format(iteration + 1)
        with fitsio.FITS(delta_attrib_name, 'rw', clobber=True) as results:
            header = {}
            header["NSIDE"] = nside
            header["PIXORDER"] = healpy_pix_ordering
            header["FITORDER"] = args.order
            results.write([stack_log_lambda, get_stack_delta(stack_log_lambda),
                           get_stack_delta_weights(stack_log_lambda)],
                          names=['loglam', 'stack', 'weight'],
                          header=header,
                          extname='STACK')
            results.write(
                [log_lambda,
                 Forest.get_eta(log_lambda),
                 Forest.get_var_lss(log_lambda),
                 Forest.get_fudge(log_lambda),
                 num_pixels],
                names=['loglam', 'eta', 'var_lss', 'fudge', 'nb_pixels'],
                extname='WEIGHT')
            results.write([
                log_lambda_rest_frame,
                Forest.get_mean_cont(log_lambda_rest_frame), mean_cont_weight
            ],
                          names=['loglam_rest', 'mean_cont', 'weight'],
                          extname='CONT')
            var_pipe_values_out = np.broadcast_to(var_pipe_values.reshape(1, -1),
                                              var_delta.shape)
            results.write([
                var_pipe_values_out, var_delta, var2_delta, count, num_qso,
                chi2_in_bin
            ],
                          names=[
                              'var_pipe', 'var_del', 'var2_del', 'count',
                              'nqsos', 'chi2'
                          ],
                          extname='VAR')

    ### Compute deltas and format them
    deltas = {}
    data_bad_cont = []
    for healpix in sorted(data.keys()):
        for forest in data[healpix]:
            if not forest.bad_cont is None:
                continue
            #-- Compute delta field from flux, continuum and various quantites
            get_delta_from_forest(forest, get_stack_delta, Forest.get_var_lss,
                                  Forest.get_eta, Forest.get_fudge,
                                  args.use_mock_continuum)
            if healpix in deltas:
                deltas[healpix].append(forest)
            else:
                deltas[healpix] = [forest]
        data_bad_cont = data_bad_cont + [
            forest for forest in data[healpix] if forest.bad_cont is not None
        ]

    for forest in data_bad_cont:
        log_file.write("INFO: Rejected {} due to {}\n".format(
            forest.thingid, forest.bad_cont))

    log_file.write(
        ("INFO: Accepted sample has {}"
         "forests\n").format(np.sum([len(p) for p in deltas.values()])))

    t2 = time.time()
    tmin = (t2 - t1) / 60
    userprint('INFO: time elapsed to fit continuum', tmin, 'minutes')

    ### Read metadata from forests and export it
    if not args.metadata is None:
        tab_cont = get_metadata(data)
        tab_cont.write(os.path.expandvars(args.metadata), format="fits", overwrite=True)

    ### Save delta
    for healpix in sorted(deltas.keys()):

        if args.delta_format == 'Pk1D_ascii':
            results = open(args.out_dir + "/delta-{}".format(healpix) + ".txt",
                           'w')
            for delta in deltas[healpix]:
                num_pixels = len(delta.delta)
                if args.mode == 'desi':
                    delta_log_lambda = (
                        (delta.log_lambda[-1] - delta.log_lambda[0]) /
                        float(len(delta.log_lambda) - 1))
                else:
                    delta_log_lambda = delta.delta_log_lambda
                line = '{} {} {} '.format(delta.plate, delta.mjd, delta.fiberid)
                line += '{} {} {} '.format(delta.ra, delta.dec, delta.z_qso)
                line += '{} {} {} {} {} '.format(delta.mean_z, delta.mean_snr,
                                                 delta.mean_reso,
                                                 delta_log_lambda, num_pixels)
                for index in range(num_pixels):
                    line += '{} '.format(delta.delta[index])
                for index in range(num_pixels):
                    line += '{} '.format(delta.log_lambda[index])
                for index in range(num_pixels):
                    line += '{} '.format(delta.ivar[index])
                for index in range(num_pixels):
                    line += '{} '.format(delta.exposures_diff[index])
                line += ' \n'
                results.write(line)

            results.close()

        else:
            results = fitsio.FITS(args.out_dir + "/delta-{}".format(healpix) +
                                  ".fits.gz",
                                  'rw',
                                  clobber=True)
            for delta in deltas[healpix]:
                header = [
                    {
                        'name': 'RA',
                        'value': delta.ra,
                        'comment': 'Right Ascension [rad]'
                    },
                    {
                        'name': 'DEC',
                        'value': delta.dec,
                        'comment': 'Declination [rad]'
                    },
                    {
                        'name': 'Z',
                        'value': delta.z_qso,
                        'comment': 'Redshift'
                    },
                    {
                        'name':
                            'PMF',
                        'value':
                            '{}-{}-{}'.format(delta.plate, delta.mjd,
                                              delta.fiberid)
                    },
                    {
                        'name': 'THING_ID',
                        'value': delta.thingid,
                        'comment': 'Object identification'
                    },
                    {
                        'name': 'PLATE',
                        'value': delta.plate
                    },
                    {
                        'name': 'MJD',
                        'value': delta.mjd,
                        'comment': 'Modified Julian date'
                    },
                    {
                        'name': 'FIBERID',
                        'value': delta.fiberid
                    },
                    {
                        'name': 'ORDER',
                        'value': delta.order,
                        'comment': 'Order of the continuum fit'
                    },
                    {
                        'name': "BLINDING",
                        'value': blinding,
                        'comment': 'String specifying the blinding strategy'
                    },
                ]

                if blinding != "none":
                    delta_name = "DELTA_BLIND"
                else:
                    delta_name = "DELTA"
                if args.delta_format == 'Pk1D':
                    header += [
                        {
                            'name': 'MEANZ',
                            'value': delta.mean_z,
                            'comment': 'Mean redshift'
                        },
                        {
                            'name': 'MEANRESO',
                            'value': delta.mean_reso,
                            'comment': 'Mean resolution'
                        },
                        {
                            'name': 'MEANSNR',
                            'value': delta.mean_snr,
                            'comment': 'Mean SNR'
                        },
                    ]
                    if args.mode == 'desi':
                        delta_log_lambda = (
                            (delta.log_lambda[-1] - delta.log_lambda[0]) /
                            float(len(delta.log_lambda) - 1))
                    else:
                        delta_log_lambda = delta.delta_log_lambda
                    header += [{
                        'name': 'DLL',
                        'value': delta_log_lambda,
                        'comment': 'Loglam bin size [log Angstrom]'
                    }]
                    exposures_diff = delta.exposures_diff
                    if exposures_diff is None:
                        exposures_diff = delta.log_lambda * 0

                    cols = [
                        delta.log_lambda, delta.delta, delta.ivar,
                        exposures_diff
                    ]
                    names = ['LOGLAM', delta_name, 'IVAR', 'DIFF']
                    units = ['log Angstrom', '', '', '']
                    comments = [
                        'Log lambda', 'Delta field', 'Inverse variance',
                        'Difference'
                    ]
                else:
                    cols = [
                        delta.log_lambda, delta.delta, delta.ivar, delta.weights, delta.cont
                    ]
                    names = ['LOGLAM', delta_name, 'IVAR', 'WEIGHT', 'CONT']
                    units = ['log Angstrom', '', '', '', '']
                    comments = [
                        'Log lambda', 'Delta field', 'Inverse variance', 'Pixel weights',
                        'Continuum'
                    ]

                results.write(cols,
                              names=names,
                              header=header,
                              comment=comments,
                              units=units,
                              extname=str(delta.thingid))

            results.close()

    t3 = time.time()
    tmin = (t3 - t2) / 60
    userprint('INFO: time elapsed to write deltas', tmin, 'minutes')
    ttot = (t3 - t0) / 60
    userprint('INFO: total elapsed time', ttot, 'minutes')

    log_file.close()


if __name__ == '__main__':
    cmdargs=sys.argv[1:]
    warnings.warn("Note that the picca_deltas routines will be removed with the next picca release, please use picca_delta_extraction instead", DeprecationWarning)

    main(cmdargs)
