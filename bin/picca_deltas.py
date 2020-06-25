#!/usr/bin/env python3
"""Computes delta field from a list of spectra.

Computes the mean transmission fluctuation field (delta field) for a list of
spectra for the specified absorption line. Follow the procedure described in
section 2.4 of du Mas des Bourboux et al. 2020 (In prep).
"""
import sys
import os
from multiprocessing import Pool
import argparse
import fitsio
import numpy as np
from scipy.interpolate import interp1d

from picca.data import Forest, Delta
from picca import prep_del, io, constants
from picca.utils import userprint


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


def main():
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
                        default='pix',
                        required=False,
                        help=('Open mode of the spectra files: pix, spec, '
                              'spcframe, spplate, desi'))

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

    args = parser.parse_args()

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

    Forest.get_var_lss = interp1d(
        (Forest.log_lambda_min + np.arange(2) *
         (Forest.log_lambda_max - Forest.log_lambda_min)),
        0.2 + np.zeros(2),
        fill_value="extrapolate",
        kind="nearest")
    Forest.get_eta = interp1d((Forest.log_lambda_min + np.arange(2) *
                               (Forest.log_lambda_max - Forest.log_lambda_min)),
                              np.ones(2),
                              fill_value="extrapolate",
                              kind="nearest")
    Forest.get_fudge = interp1d(
        (Forest.log_lambda_min + np.arange(2) *
         (Forest.log_lambda_max - Forest.log_lambda_min)),
        np.zeros(2),
        fill_value="extrapolate",
        kind="nearest")
    Forest.get_mean_cont = interp1d(
        (Forest.log_lambda_min_rest_frame + np.arange(2) *
         (Forest.log_lambda_max_rest_frame - Forest.log_lambda_min_rest_frame)),
        1 + np.zeros(2))
    # end of setup forest class variables

    ### check that the order of the continuum fit is 0 (constant) or 1 (linear).
    if args.order:
        if (args.order != 0) and (args.order != 1):
            userprint(("ERROR : invalid value for order, must be eqal to 0 or"
                       "1. Here order = {:d}").format(args.order))
            sys.exit(12)

    ### Correct multiplicative pipeline flux calibration
    if args.flux_calib is not None:
        try:
            hdul = fitsio.FITS(args.flux_calib)
            stack_log_lambda = hdul[1]['loglam'][:]
            stack_delta = hdul[1]['stack'][:]
            w = (stack_delta != 0.)
            Forest.correct_flux = interp1d(stack_log_lambda[w],
                                           stack_delta[w],
                                           fill_value="extrapolate",
                                           kind="nearest")
            hdul.close()
        except (OSError, ValueError):
            userprint(("ERROR: Error while reading flux_calib"
                       "file {}".format(args.flux_calib)))
            sys.exit(1)

    ### Correct multiplicative pipeline inverse variance calibration
    if args.ivar_calib is not None:
        try:
            hdul = fitsio.FITS(args.ivar_calib)
            log_lambda = hdul[2]['LOGLAM'][:]
            eta = hdul[2]['ETA'][:]
            Forest.correct_ivar = interp1d(log_lambda,
                                           eta,
                                           fill_value="extrapolate",
                                           kind="nearest")
            hdul.close()
        except (OSError, ValueError):
            userprint(("ERROR: Error while reading ivar_calib"
                       "file {}".format(args.ivar_calib)))
            sys.exit(1)

    ### Apply dust correction
    if not args.dust_map is None:
        userprint("applying dust correction")
        Forest.extinction_bv_map = io.read_dust_map(args.dust_map)

    log_file = open(os.path.expandvars(args.log), 'w')

    # Read data
    (data, num_data, nside,
     healpy_pix_ordering) = io.read_data(os.path.expandvars(args.in_dir),
                                         args.drq,
                                         args.mode,
                                         z_min=args.zqso_min,
                                         z_max=args.zqso_max,
                                         max_num_spec=args.nspec,
                                         log_file=log_file,
                                         keep_bal=args.keep_bal,
                                         bi_max=args.bi_max,
                                         order=args.order,
                                         best_obs=args.best_obs,
                                         single_exp=args.single_exp,
                                         pk1d=args.delta_format)

    ### Read masks
    mask_obs_frame = None
    mask_rest_frame = None
    mask_rest_frame_dla = None
    if args.mask_file is not None:
        args.mask_file = os.path.expandvars(args.mask_file)
        try:
            mask_obs_frame = []
            mask_rest_frame = []
            mask_rest_frame_dla = []
            with open(args.mask_file, 'r') as file:
                loop = True
                for line in file:
                    if line[0] == '#':
                        continue
                    cols = line.split()
                    if cols[3] == 'OBS':
                        mask_obs_frame += [[float(cols[1]), float(cols[2])]]
                    elif cols[3] == 'RF':
                        mask_rest_frame += [[float(cols[1]), float(cols[2])]]
                    elif cols[3] == 'RF_DLA':
                        mask_rest_frame_dla += [[
                            float(cols[1]), float(cols[2])
                        ]]
                    else:
                        raise ValueError("Invalid value found in mask")
            mask_obs_frame = np.log10(np.asarray(mask_obs_frame))
            mask_rest_frame = np.log10(np.asarray(mask_rest_frame))
            mask_rest_frame_dla = np.log10(np.asarray(mask_rest_frame_dla))
            if mask_rest_frame_dla.size == 0:
                mask_rest_frame_dla = None

        except (OSError, ValueError):
            userprint(("ERROR: Error while reading mask_file "
                       "file {}").format(args.mask_file))
            sys.exit(1)

    ### Mask lines
    if not mask_obs_frame is None:
        if mask_obs_frame.size + mask_rest_frame.size != 0:
            for healpix in data:
                for forest in data[healpix]:
                    forest.mask(mask_obs_frame, mask_rest_frame)

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
        np.random.seed(0)
        dlas = io.read_dlas(args.dla_vac)
        num_dlas = 0
        for healpix in data:
            for forest in data[healpix]:
                if forest.thingid in dlas:
                    for dla in dlas[forest.thingid]:
                        forest.add_dla(dla[0], dla[1], mask_rest_frame_dla)
                        num_dlas += 1
        log_file.write("Found {} DLAs in forests\n".format(num_dlas))

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
                log_file.write(("INFO: Rejected {} due to forest too "
                                "short\n").format(forest.thingid))
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

    log_file.write(
        ("INFO: Remaining sample has {} "
         "forests\n").format(np.sum([len(forest) for forest in data.values()])))

    # Sanity check: all forests must have the attribute log_lambda
    for healpix in data:
        for forest in data[healpix]:
            assert forest.log_lambda is not None

    # compute fits to the forests iteratively
    # (see equations 2 to 4 in du Mas des Bourboux et al. 2020)
    num_iterations = args.nit
    for iteration in range(num_iterations):
        pool = Pool(processes=args.nproc)
        userprint("iteration: ", iteration)
        nfit = 0
        sort = np.array(list(data.keys())).argsort()
        data_fit_cont = pool.map(cont_fit, np.array(list(data.values()))[sort])
        for index, healpix in enumerate(sorted(list(data.keys()))):
            data[healpix] = data_fit_cont[index]

        userprint("done")

        pool.close()

        if iteration < num_iterations - 1:
            (log_lambda_rest_frame, mean_cont,
             mean_cont_weight) = prep_del.compute_mean_cont(data)
            Forest.get_mean_cont = interp1d(
                log_lambda_rest_frame[mean_cont_weight > 0.],
                Forest.get_mean_cont(
                    log_lambda_rest_frame[mean_cont_weight > 0.]) *
                mean_cont[mean_cont_weight > 0.],
                fill_value="extrapolate")
            if not (args.use_ivar_as_weight or args.use_constant_weight):
                (log_lambda, eta, var_lss, fudge, num_pixels, var_pipe_values,
                 var_delta, var2_delta, count, num_qso, chi2_in_bin, error_eta,
                 error_var_lss, err_fudge) = prep_del.compute_var_stats(
                     data, (args.eta_min, args.eta_max),
                     (args.vlss_min, args.vlss_max))
                Forest.get_eta = interp1d(log_lambda[num_pixels > 0],
                                          eta[num_pixels > 0],
                                          fill_value="extrapolate",
                                          kind="nearest")
                Forest.get_var_lss = interp1d(log_lambda[num_pixels > 0],
                                              var_lss[num_pixels > 0.],
                                              fill_value="extrapolate",
                                              kind="nearest")
                Forest.get_fudge = interp1d(log_lambda[num_pixels > 0],
                                            fudge[num_pixels > 0],
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
                err_fudge = np.zeros(num_bins)
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

    ### Save iter_out_prefix
    results = fitsio.FITS(args.iter_out_prefix + ".fits.gz", 'rw', clobber=True)
    header = {}
    header["NSIDE"] = nside
    header["PIXORDER"] = healpy_pix_ordering
    header["FITORDER"] = args.order
    results.write([stack_log_lambda, stack_delta, stack_weight],
                  names=['loglam', 'stack', 'weight'],
                  header=header,
                  extname='STACK')
    results.write([log_lambda, eta, var_lss, fudge, num_pixels],
                  names=['loglam', 'eta', 'var_lss', 'fudge', 'nb_pixels'],
                  extname='WEIGHT')
    results.write([
        log_lambda_rest_frame,
        Forest.get_mean_cont(log_lambda_rest_frame), mean_cont_weight
    ],
                  names=['loglam_rest', 'mean_cont', 'weight'],
                  extname='CONT')
    var_pipe_values = np.broadcast_to(var_pipe_values.reshape(1, -1),
                                      var_delta.shape)
    results.write(
        [var_pipe_values, var_delta, var2_delta, count, num_qso, chi2_in_bin],
        names=['var_pipe', 'var_del', 'var2_del', 'count', 'nqsos', 'chi2'],
        extname='VAR')
    results.close()

    ### Compute deltas and format them
    get_stack_delta = interp1d(stack_log_lambda[stack_weight > 0.],
                               stack_delta[stack_weight > 0.],
                               kind="nearest",
                               fill_value="extrapolate")
    deltas = {}
    data_bad_cont = []
    for healpix in sorted(data.keys()):
        deltas[healpix] = [
            Delta.from_forest(forest, get_stack_delta, Forest.get_var_lss,
                              Forest.get_eta, Forest.get_fudge,
                              args.use_mock_continuum)
            for forest in data[healpix]
            if forest.bad_cont is None
        ]
        data_bad_cont = data_bad_cont + [
            forest for forest in data[healpix] if forest.bad_cont is not None
        ]

    for forest in data_bad_cont:
        log_file.write("INFO: Rejected {} due to {}\n".format(
            forest.thingid, forest.bad_cont))

    log_file.write(
        ("INFO: Accepted sample has {}"
         "forests\n").format(np.sum([len(p) for p in deltas.values()])))

    log_file.close()

    ### Save delta
    for healpix in sorted(deltas.keys()):

        if len(deltas[healpix]) == 0:
            continue
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
                ]

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
                    names = ['LOGLAM', 'DELTA', 'IVAR', 'DIFF']
                    units = ['log Angstrom', '', '', '']
                    comments = [
                        'Log lambda', 'Delta field', 'Inverse variance',
                        'Difference'
                    ]
                else:
                    cols = [
                        delta.log_lambda, delta.delta, delta.weights, delta.cont
                    ]
                    names = ['LOGLAM', 'DELTA', 'WEIGHT', 'CONT']
                    units = ['log Angstrom', '', '', '']
                    comments = [
                        'Log lambda', 'Delta field', 'Pixel weights',
                        'Continuum'
                    ]

                results.write(cols,
                              names=names,
                              header=header,
                              comment=comments,
                              units=units,
                              extname=str(delta.thingid))

            results.close()


if __name__ == '__main__':
    main()
