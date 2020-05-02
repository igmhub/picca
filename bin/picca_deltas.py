#!/usr/bin/python3
"""Computes delta field from a list of spectra.

Computes the mean transmission fluctuation field (delta field) for a list of
spectra for the specified absorption line. Follow the procedure describe in
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


def cont_fit(data):
    """ Computes the quasar continua for all the forests in data

    Args:
        data: a list of forest instances
    Returns:
        the list of forests after having computed their quasar continua
    """
    for d in data:
        d.cont_fit()
    return data


def main():
    # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    """Computes delta field"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Compute the delta field from a list of spectra')

    parser.add_argument('--out-dir', type=str, default=None, required=True,
                        help='Output directory')

    parser.add_argument('--drq', type=str, default=None, required=True,
                        help='Catalog of objects in DRQ format')

    parser.add_argument('--in-dir', type=str, default=None, required=True,
                        help='Directory to spectra files')

    parser.add_argument('--log', type=str, default='input.log', required=False,
                        help='Log input data')

    parser.add_argument('--iter-out-prefix', type=str, default='iter', required=False,
                        help='Prefix of the iteration file')

    parser.add_argument('--mode', type=str, default='pix', required=False,
                        help='Open mode of the spectra files: pix, spec, spcframe, spplate, desi')

    parser.add_argument('--best-obs', action='store_true', required=False,
                        help='If mode == spcframe, then use only the best observation')

    parser.add_argument('--single-exp', action='store_true', required=False,
                        help='If mode == spcframe, then use only one of the available exposures. If best-obs then choose it among those contributing to the best obs')

    parser.add_argument('--zqso-min', type=float, default=None, required=False,
                        help='Lower limit on quasar redshift from drq')

    parser.add_argument('--zqso-max', type=float, default=None, required=False,
                        help='Upper limit on quasar redshift from drq')

    parser.add_argument('--keep-bal', action='store_true', required=False,
                        help='Do not reject BALs in drq')

    parser.add_argument('--bi-max', type=float, required=False, default=None,
                        help='Maximum CIV balnicity index in drq (overrides --keep-bal)')

    parser.add_argument('--lambda-min', type=float, default=3600., required=False,
                        help='Lower limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-max', type=float, default=5500., required=False,
                        help='Upper limit on observed wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-min', type=float, default=1040., required=False,
                        help='Lower limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--lambda-rest-max', type=float, default=1200., required=False,
                        help='Upper limit on rest frame wavelength [Angstrom]')

    parser.add_argument('--rebin', type=int, default=3, required=False,
                        help='Rebin wavelength grid by combining this number of adjacent pixels (ivar weight)')

    parser.add_argument('--npix-min', type=int, default=50, required=False,
                        help='Minimum of rebined pixels')

    parser.add_argument('--dla-vac', type=str, default=None, required=False,
                        help='DLA catalog file')

    parser.add_argument('--dla-mask', type=float, default=0.8, required=False,
                        help='Lower limit on the DLA transmission. Transmissions below this number are masked')

    parser.add_argument('--absorber-vac', type=str, default=None, required=False,
                        help='Absorber catalog file')

    parser.add_argument('--absorber-mask', type=float, default=2.5, required=False,
                        help='Mask width on each side of the absorber central observed wavelength in units of 1e4*dlog10(lambda)')

    parser.add_argument('--mask-file', type=str, default=None, required=False,
                        help='Path to file to mask regions in lambda_OBS and lambda_RF. In file each line is: region_name region_min region_max (OBS or RF) [Angstrom]')

    parser.add_argument('--optical-depth', type=str, default=None, required=False,
                        help='Correct for the optical depth: tau_1 gamma_1 absorber_1 tau_2 gamma_2 absorber_2 ...', nargs='*')

    parser.add_argument('--dust-map', type=str, default=None, required=False,
                        help='Path to DRQ catalog of objects for dust map to apply the Schlegel correction')

    parser.add_argument('--flux-calib', type=str, default=None, required=False,
                        help='Path to previously produced picca_delta.py file to correct for multiplicative errors in the pipeline flux calibration')

    parser.add_argument('--ivar-calib', type=str, default=None, required=False,
                        help='Path to previously produced picca_delta.py file to correct for multiplicative errors in the pipeline inverse variance calibration')

    parser.add_argument('--eta-min', type=float, default=0.5, required=False,
                        help='Lower limit for eta')

    parser.add_argument('--eta-max', type=float, default=1.5, required=False,
                        help='Upper limit for eta')

    parser.add_argument('--vlss-min', type=float, default=0., required=False,
                        help='Lower limit for variance LSS')

    parser.add_argument('--vlss-max', type=float, default=0.3, required=False,
                        help='Upper limit for variance LSS')

    parser.add_argument('--delta-format', type=str, default=None, required=False,
                        help='Format for Pk 1D: Pk1D')

    parser.add_argument('--use-ivar-as-weight', action='store_true', default=False,
                        help='Use ivar as weights (implemented as eta = 1, sigma_lss = fudge = 0)')

    parser.add_argument('--use-constant-weight', action='store_true', default=False,
                        help='Set all the delta weights to one (implemented as eta = 0, sigma_lss = 1, fudge = 0)')

    parser.add_argument('--order', type=int, default=1, required=False,
                        help='Order of the log10(lambda) polynomial for the continuum fit, by default 1.')

    parser.add_argument('--nit', type=int, default=5, required=False,
                        help='Number of iterations to determine the mean continuum shape, LSS variances, etc.')

    parser.add_argument('--nproc', type=int, default=None, required=False,
                        help='Number of processors')

    parser.add_argument('--nspec', type=int, default=None, required=False,
                        help='Maximum number of spectra to read')

    parser.add_argument('--use-mock-continuum', action='store_true', default=False,
                        help='use the mock continuum for computing the deltas')

    args = parser.parse_args()

    # setup forest class variables
    Forest.log_lambda_min = np.log10(args.lambda_min)
    Forest.log_lambda_max = np.log10(args.lambda_max)
    Forest.log_lambda_min_rest_frame = np.log10(args.lambda_rest_min)
    Forest.log_lambda_max_rest_frame = np.log10(args.lambda_rest_max)
    Forest.rebin = args.rebin
    Forest.delta_log_lambda = args.rebin*1e-4
    # minumum dla transmission
    Forest.dla_mask_limit = args.dla_mask
    Forest.absorber_mask_width = args.absorber_mask

    # Find the redshift range
    if args.zqso_min is None:
        args.zqso_min = max(0., args.lambda_min/args.lambda_rest_max - 1.)
        userprint("zqso_min = {}".format(args.zqso_min))
    if args.zqso_max is None:
        args.zqso_max = max(0., args.lambda_max/args.lambda_rest_min - 1.)
        userprint("zqso_max = {}".format(args.zqso_max))

    Forest.get_var_lss = interp1d(Forest.log_lambda_min + np.arange(2)*(Forest.log_lambda_max - Forest.log_lambda_min), 0.2 + np.zeros(2), fill_value="extrapolate", kind="nearest")
    Forest.get_eta = interp1d(Forest.log_lambda_min + np.arange(2)*(Forest.log_lambda_max - Forest.log_lambda_min), np.ones(2), fill_value="extrapolate", kind="nearest")
    Forest.get_fudge = interp1d(Forest.log_lambda_min + np.arange(2)*(Forest.log_lambda_max - Forest.log_lambda_min), np.zeros(2), fill_value="extrapolate", kind="nearest")
    Forest.get_mean_cont = interp1d(Forest.log_lambda_min_rest_frame + np.arange(2)*(Forest.log_lambda_max_rest_frame - Forest.log_lambda_min_rest_frame), 1 + np.zeros(2))
    # end of setup forest class variables

    ### check that the order of the continuum fit is 0 (constant) or 1 (linear).
    if args.order:
        if (args.order != 0) and (args.order != 1):
            userprint("ERROR : invalid value for order, must be eqal to 0 or 1. Here order = %i"%(args.order))
            sys.exit(12)

    ### Correct multiplicative pipeline flux calibration
    if args.flux_calib is not None:
        try:
            hdul = fitsio.FITS(args.flux_calib)
            ll_st = hdul[1]['loglam'][:]
            mean_delta = hdul[1]['stack'][:]
            w = (mean_delta != 0.)
            Forest.correct_flux = interp1d(ll_st[w], mean_delta[w], fill_value="extrapolate", kind="nearest")
            hdul.close()
        except (OSError, ValueError):
            userprint("ERROR: Error while reading flux_calib file {}".format(args.flux_calib))
            sys.exit(1)

    ### Correct multiplicative pipeline inverse variance calibration
    if args.ivar_calib is not None:
        try:
            hdul = fitsio.FITS(args.ivar_calib)
            log_lambda = hdul[2]['LOGLAM'][:]
            eta = hdul[2]['ETA'][:]
            Forest.correct_ivar = interp1d(log_lambda, eta, fill_value="extrapolate", kind="nearest")
            hdul.close()
        except (OSError, ValueError):
            userprint("ERROR: Error while reading ivar_calib file {}".format(args.ivar_calib))
            sys.exit(1)

    ### Apply dust correction
    if not args.dust_map is None:
        userprint("applying dust correction")
        Forest.extinction_bv_map = io.read_dust_map(args.dust_map)

    nit = args.nit

    log = open(os.path.expandvars(args.log), 'w')

    data, ndata, healpy_nside, healpy_pix_ordering = io.read_data(os.path.expandvars(args.in_dir), args.drq, args.mode,\
        zmin=args.zqso_min, zmax=args.zqso_max, nspec=args.nspec, log=log,\
        keep_bal=args.keep_bal, bi_max=args.bi_max, order=args.order,\
        best_obs=args.best_obs, single_exp=args.single_exp, pk1d=args.delta_format)

    ### Get the lines to veto
    mask_obs_frame = None
    mask_rest_frame = None
    usr_mask_RF_DLA = None
    if args.mask_file is not None:
        args.mask_file = os.path.expandvars(args.mask_file)
        try:
            mask_obs_frame = []
            mask_rest_frame = []
            usr_mask_RF_DLA = []
            with open(args.mask_file, 'r') as f:
                loop = True
                for l in f:
                    if l[0] == '#':
                        continue
                    l = l.split()
                    if l[3] == 'OBS':
                        mask_obs_frame += [[float(l[1]), float(l[2])]]
                    elif l[3] == 'RF':
                        mask_rest_frame += [[float(l[1]), float(l[2])]]
                    elif l[3] == 'RF_DLA':
                        usr_mask_RF_DLA += [[float(l[1]), float(l[2])]]
                    else:
                        raise ValueError("Invalid value found in mask")
            mask_obs_frame = np.log10(np.asarray(mask_obs_frame))
            mask_rest_frame = np.log10(np.asarray(mask_rest_frame))
            usr_mask_RF_DLA = np.log10(np.asarray(usr_mask_RF_DLA))
            if usr_mask_RF_DLA.size == 0:
                usr_mask_RF_DLA = None

        except (OSError, ValueError):
            userprint("ERROR: Error while reading mask_file file {}".format(args.mask_file))
            sys.exit(1)

    ### Veto lines
    if not mask_obs_frame is None:
        if mask_obs_frame.size + mask_rest_frame.size != 0:
            for p in data:
                for d in data[p]:
                    d.mask(mask_obs_frame, mask_rest_frame)

    ### Veto absorbers
    if not args.absorber_vac is None:
        userprint("INFO: Adding absorbers")
        absorbers = io.read_absorbers(args.absorber_vac)
        nb_absorbers_in_forest = 0
        for p in data:
            for d in data[p]:
                if d.thingid in absorbers:
                    for lambda_absorber in absorbers[d.thingid]:
                        d.add_absorber(lambda_absorber)
                        nb_absorbers_in_forest += 1
        log.write("Found {} absorbers in forests\n".format(nb_absorbers_in_forest))

    ### Apply optical depth
    if not args.optical_depth is None:
        userprint("INFO: Adding {} optical depths".format(len(args.optical_depth)//3))
        assert len(args.optical_depth)%3 == 0
        for idxop in range(len(args.optical_depth)//3):
            tau = float(args.optical_depth[3*idxop])
            gamma = float(args.optical_depth[3*idxop+1])
            lambda_rest_frame = constants.absorber_IGM[args.optical_depth[3*idxop+2]]
            userprint("INFO: Adding optical depth for tau = {}, gamma = {}, lambda_rest_frame = {} A".format(tau, gamma, lambda_rest_frame))
            for p in data:
                for d in data[p]:
                    d.add_optical_depth(tau, gamma, lambda_rest_frame)

    ### Correct for DLAs
    if not args.dla_vac is None:
        userprint("INFO: Adding DLAs")
        np.random.seed(0)
        dlas = io.read_dlas(args.dla_vac)
        nb_dla_in_forest = 0
        for p in data:
            for d in data[p]:
                if d.thingid in dlas:
                    for dla in dlas[d.thingid]:
                        d.add_dla(dla[0], dla[1], usr_mask_RF_DLA)
                        nb_dla_in_forest += 1
        log.write("Found {} DLAs in forests\n".format(nb_dla_in_forest))

    ## cuts
    log.write("INFO: Input sample has {} forests\n".format(np.sum([len(p) for p in data.values()])))
    lstKeysToDel = []
    for p in data:
        l = []
        for d in data[p]:
            if not hasattr(d, 'log_lambda') or len(d.log_lambda) < args.npix_min:
                log.write("INFO: Rejected {} due to forest too short\n".format(d.thingid))
                continue

            if np.isnan((d.flux*d.ivar).sum()):
                log.write("INFO: Rejected {} due to nan found\n".format(d.thingid))
                continue

            if(args.use_constant_weight and (d.flux.mean() <= 0.0 or d.mean_snr <= 1.0)):
                log.write("INFO: Rejected {} due to negative mean or too low SNR found\n".format(d.thingid))
                continue

            l.append(d)
            log.write("{} {}-{}-{} accepted\n".format(d.thingid, d.plate, d.mjd, d.fiberid))
        data[p][:] = l
        if len(data[p]) == 0:
            lstKeysToDel += [p]

    for p in lstKeysToDel:
        del data[p]

    log.write("INFO: Remaining sample has {} forests\n".format(np.sum([len(p) for p in data.values()])))

    for p in data:
        for d in data[p]:
            assert hasattr(d, 'log_lambda')

    for it in range(nit):
        pool = Pool(processes=args.nproc)
        userprint("iteration: ", it)
        nfit = 0
        sort = np.array(list(data.keys())).argsort()
        data_fit_cont = pool.map(cont_fit, np.array(list(data.values()))[sort])
        for i, p in enumerate(sorted(list(data.keys()))):
            data[p] = data_fit_cont[i]

        userprint("done")

        pool.close()

        if it < nit-1:
            ll_rest, mean_cont, wmc = prep_del.compute_mean_cont(data)
            Forest.get_mean_cont = interp1d(ll_rest[wmc > 0.], Forest.get_mean_cont(ll_rest[wmc > 0.])*mean_cont[wmc > 0.], fill_value="extrapolate")
            if not (args.use_ivar_as_weight or args.use_constant_weight):
                log_lambda, eta, vlss, fudge, nb_pixels, var, var_del, var2_del,\
                    count, nqsos, chi2, err_eta, err_vlss, err_fudge = \
                        prep_del.var_lss(data, (args.eta_min, args.eta_max), (args.vlss_min, args.vlss_max))
                Forest.get_eta = interp1d(log_lambda[nb_pixels > 0], eta[nb_pixels > 0],
                                      fill_value="extrapolate", kind="nearest")
                Forest.get_var_lss = interp1d(log_lambda[nb_pixels > 0], vlss[nb_pixels > 0.],
                                          fill_value="extrapolate", kind="nearest")
                Forest.get_fudge = interp1d(log_lambda[nb_pixels > 0], fudge[nb_pixels > 0],
                                        fill_value="extrapolate", kind="nearest")
            else:

                nlss = 10 # this value is arbitrary
                log_lambda = Forest.log_lambda_min + (np.arange(nlss)+.5)*(Forest.log_lambda_max-Forest.log_lambda_min)/nlss

                if args.use_ivar_as_weight:
                    userprint('INFO: using ivar as weights, skipping eta, var_lss, fudge fits')
                    eta = np.ones(nlss)
                    vlss = np.zeros(nlss)
                    fudge = np.zeros(nlss)
                else:
                    userprint('INFO: using constant weights, skipping eta, var_lss, fudge fits')
                    eta = np.zeros(nlss)
                    vlss = np.ones(nlss)
                    fudge = np.zeros(nlss)

                err_eta = np.zeros(nlss)
                err_vlss = np.zeros(nlss)
                err_fudge = np.zeros(nlss)
                chi2 = np.zeros(nlss)

                nb_pixels = np.zeros(nlss)
                var = np.zeros(nlss)
                var_del = np.zeros((nlss, nlss))
                var2_del = np.zeros((nlss, nlss))
                count = np.zeros((nlss, nlss))
                nqsos = np.zeros((nlss, nlss))

                Forest.get_eta = interp1d(log_lambda, eta, fill_value='extrapolate', kind='nearest')
                Forest.get_var_lss = interp1d(log_lambda, vlss, fill_value='extrapolate', kind='nearest')
                Forest.get_fudge = interp1d(log_lambda, fudge, fill_value='extrapolate', kind='nearest')


    ll_st, mean_delta, wst = prep_del.stack(data)

    ### Save iter_out_prefix
    res = fitsio.FITS(args.iter_out_prefix + ".fits.gz", 'rw', clobber=True)
    hd = {}
    hd["NSIDE"] = healpy_nside
    hd["PIXORDER"] = healpy_pix_ordering
    hd["FITORDER"] = args.order
    res.write([ll_st, mean_delta, wst], names=['loglam', 'stack', 'weight'], header=hd, extname='STACK')
    res.write([log_lambda, eta, vlss, fudge, nb_pixels], names=['loglam', 'eta', 'var_lss', 'fudge', 'nb_pixels'], extname='WEIGHT')
    res.write([ll_rest, Forest.get_mean_cont(ll_rest), wmc], names=['loglam_rest', 'mean_cont', 'weight'], extname='CONT')
    var = np.broadcast_to(var.reshape(1, -1), var_del.shape)
    res.write([var, var_del, var2_del, count, nqsos, chi2], names=['var_pipe', 'var_del', 'var2_del', 'count', 'nqsos', 'chi2'], extname='VAR')
    res.close()

    ### Save delta
    get_mean_delta = interp1d(ll_st[wst > 0.], mean_delta[wst > 0.], kind="nearest", fill_value="extrapolate")
    deltas = {}
    data_bad_cont = []
    for p in sorted(data.keys()):
        deltas[p] = [Delta.from_forest(d, get_mean_delta, Forest.get_var_lss, Forest.get_eta, Forest.get_fudge, args.use_mock_continuum) for d in data[p] if d.bad_cont is None]
        data_bad_cont = data_bad_cont + [d for d in data[p] if d.bad_cont is not None]

    for d in data_bad_cont:
        log.write("INFO: Rejected {} due to {}\n".format(d.thingid, d.bad_cont))

    log.write("INFO: Accepted sample has {} forests\n".format(np.sum([len(p) for p in deltas.values()])))

    log.close()

    ###
    for p in sorted(deltas.keys()):

        if len(deltas[p]) == 0:
            continue
        if args.delta_format == 'Pk1D_ascii':
            out_ascii = open(args.out_dir + "/delta-{}".format(p) + ".txt", 'w')
            for d in deltas[p]:
                nbpixel = len(d.delta)
                delta_log_lambda = d.delta_log_lambda
                if args.mode == 'desi':
                    delta_log_lambda = (d.log_lambda[-1] - d.log_lambda[0])/float(len(d.log_lambda) - 1)
                line = '{} {} {} '.format(d.plate, d.mjd, d.fiberid)
                line += '{} {} {} '.format(d.ra, d.dec, d.z_qso)
                line += '{} {} {} {} {} '.format(d.mean_z, d.mean_snr, d.mean_reso, delta_log_lambda, nbpixel)
                for i in range(nbpixel):
                    line += '{} '.format(d.delta[i])
                for i in range(nbpixel):
                    line += '{} '.format(d.log_lambda[i])
                for i in range(nbpixel):
                    line += '{} '.format(d.ivar[i])
                for i in range(nbpixel):
                    line += '{} '.format(d.exposures_diff[i])
                line += ' \n'
                out_ascii.write(line)

            out_ascii.close()

        else:
            out = fitsio.FITS(args.out_dir + "/delta-{}".format(p) + ".fits.gz", 'rw', clobber=True)
            for d in deltas[p]:
                hd = [{'name':'RA', 'value':d.ra, 'comment':'Right Ascension [rad]'},
                      {'name':'DEC', 'value':d.dec, 'comment':'Declination [rad]'},
                      {'name':'Z', 'value':d.z_qso, 'comment':'Redshift'},
                      {'name':'PMF', 'value':'{}-{}-{}'.format(d.plate, d.mjd, d.fiberid)},
                      {'name':'THING_ID', 'value':d.thingid, 'comment':'Object identification'},
                      {'name':'PLATE', 'value':d.plate},
                      {'name':'MJD', 'value':d.mjd, 'comment':'Modified Julian date'},
                      {'name':'FIBERID', 'value':d.fiberid},
                      {'name':'ORDER', 'value':d.order, 'comment':'Order of the continuum fit'},
                      ]

                if args.delta_format == 'Pk1D':
                    hd += [{'name':'MEANZ', 'value':d.mean_z, 'comment':'Mean redshift'},
                           {'name':'MEANRESO', 'value':d.mean_reso, 'comment':'Mean resolution'},
                           {'name':'MEANSNR', 'value':d.mean_snr, 'comment':'Mean SNR'},
                           ]
                    delta_log_lambda = d.delta_log_lambda
                    if args.mode == 'desi':
                        delta_log_lambda = (d.log_lambda[-1] - d.log_lambda[0])/float(len(d.log_lambda) - 1)
                    hd += [{'name':'DLL', 'value':delta_log_lambda, 'comment':'Loglam bin size [log Angstrom]'}]
                    exposures_diff = d.exposures_diff
                    if exposures_diff is None:
                        exposures_diff = d.log_lambda*0

                    cols = [d.log_lambda, d.delta, d.ivar, exposures_diff]
                    names = ['LOGLAM', 'DELTA', 'IVAR', 'DIFF']
                    units = ['log Angstrom', '', '', '']
                    comments = ['Log lambda', 'Delta field', 'Inverse variance', 'Difference']
                else:
                    cols = [d.log_lambda, d.delta, d.weights, d.continuum]
                    names = ['LOGLAM', 'DELTA', 'WEIGHT', 'CONT']
                    units = ['log Angstrom', '', '', '']
                    comments = ['Log lambda', 'Delta field', 'Pixel weights', 'Continuum']

                out.write(cols, names=names, header=hd, comment=comments, units=units, extname=str(d.thingid))

            out.close()


if __name__ == '__main__':
    main()
