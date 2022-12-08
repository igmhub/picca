"""This module provides I/O functions for the raw analysis

These are:
    - read_transmission_file
    - write_delta_from_transmission
    - convert_transmission_to_deltas
The first two are meant for paralellization, and should not
be called on their own. Use the convert function for the raw analysis.
See the respective docstrings for more details
"""
import os
import sys
import glob
import numpy as np
import scipy
import fitsio
import healpy
from multiprocessing import Pool

from .data import Delta
from .utils import userprint


def read_transmission_file(filename, num_bins, objs_thingid, tracer='F_LYA', lambda_min=3600.,
                           lambda_max=5500., lambda_min_rest_frame=1040., 
                           lambda_max_rest_frame=1200., delta_log_lambda=None,
                           delta_lambda=None, lin_spaced=False):

    """Make delta objects from all skewers in a transmission file.
    Args:
        filename: str
            Path to the transmission file to read
        num_bins: int
            The number of bins in our wavelength grid.
        objs_thingid: array
            Array of object thing id.
        tracer: string - default: F_LYA.
            Tracer name for HDU in transmission file, usually F_LYA, F_LYB or F_METALS
        lambda_min: float - default: 3600.
            Minimum observed wavelength in Angstrom
        lambda_max: float - default: 5500.
            Maximum observed wavelength in Angstrom
        lambda_min_rest_frame: float - default: 1040.
            Minimum Rest Frame wavelength in Angstrom
        lambda_max_rest_frame: float - default: 1200.
            Maximum Rest Frame wavelength in Angstrom
        delta_log_lambda: float - default: None
            Variation of the logarithm of the wavelength between two pixels
        delta_lambda: float - default: None
            Variation of the wavelength between two pixels
        lin_spaced: float - default: False
            Whether to use linear spacing for the wavelength binning
    Returns:
        deltas:
            Dictionary containing delta objects for all the skewers (flux not
            yet normalised). Keys are HEALPix pixels, values are lists of delta
            objects.
        stack_flux:
            The stacked value of flux across all skewers.
        stack_weight:
            The stacked value of weights across all skewers.
    """
    if lin_spaced:
        x_min = lambda_min
        delta_x = delta_lambda if delta_lambda is not None else 3.
    else:
        x_min = np.log10(lambda_min)
        delta_x = delta_log_lambda if delta_log_lambda is not None else 3.e-4

    stack_flux = np.zeros(num_bins)
    stack_weight = np.zeros(num_bins)
    deltas = {}

    hdul = fitsio.FITS(filename)
    thingid = hdul['METADATA']['MOCKID'][:]
    if np.in1d(thingid, objs_thingid).sum() == 0:
        hdul.close()
        return
    ra = hdul['METADATA']['RA'][:].astype(np.float64) * np.pi / 180.
    dec = hdul['METADATA']['DEC'][:].astype(np.float64) * np.pi / 180.
    z = hdul['METADATA']['Z'][:]

    # Use "lambda_array" to store either lambda or log lambda
    if lin_spaced:
        lambda_array = hdul['WAVELENGTH'].read()
    else:
        lambda_array = np.log10(hdul['WAVELENGTH'].read())

    if tracer in hdul:
        trans = hdul[tracer].read()
    else:
        raise ValueError(f'Tracer {tracer} could not be found in the transmission files.')

    num_obj = z.size
    healpix = filename.split('-')[-1].split('.')[0]

    if trans.shape[0] != num_obj:
        trans = trans.transpose()

    bins = np.floor((lambda_array - x_min) / delta_x + 0.5).astype(int)
    aux_lambda = x_min + bins * delta_x
    if not lin_spaced:
        aux_lambda = 10**aux_lambda
    lambda_obs_frame = aux_lambda * np.ones(num_obj)[:, None]
    lambda_rest_frame = aux_lambda / (1. + z[:, None])
    valid_pixels = np.zeros_like(trans).astype(int)
    valid_pixels[(lambda_obs_frame >= lambda_min) &
                 (lambda_obs_frame < lambda_max) &
                 (lambda_rest_frame > lambda_min_rest_frame) &
                 (lambda_rest_frame < lambda_max_rest_frame)] = 1
    num_pixels = np.sum(valid_pixels, axis=1)
    w = num_pixels >= 50
    w &= np.in1d(thingid, objs_thingid)
    if w.sum() == 0:
        hdul.close()
        return

    ra = ra[w]
    dec = dec[w]
    z = z[w]
    thingid = thingid[w]
    trans = trans[w, :]
    valid_pixels = valid_pixels[w, :]
    num_obj = z.size
    hdul.close()

    deltas[healpix] = []
    for index2 in range(num_obj):
        aux_lambda = lambda_array[valid_pixels[index2, :] > 0]
        aux_trans = trans[index2, :][valid_pixels[index2, :] > 0]

        norm_lambda = (aux_lambda - x_min) / delta_x + 0.5
        bins = np.floor(np.around(norm_lambda, decimals=3)).astype(int)
        # bins = np.floor((aux_lambda - x_min) / delta_x + 0.5).astype(int)
        rebin_log_lambda = (x_min + np.arange(num_bins) * delta_x)
        if lin_spaced:
            rebin_log_lambda = np.log10(rebin_log_lambda)
        rebin_flux = np.bincount(bins, weights=aux_trans, minlength=num_bins)
        rebin_ivar = np.bincount(bins, minlength=num_bins).astype(float)

        w = rebin_ivar > 0.
        if w.sum() < 50:
            continue
        stack_flux += rebin_flux
        stack_weight += rebin_ivar
        rebin_log_lambda = rebin_log_lambda[w]
        rebin_flux = rebin_flux[w] / rebin_ivar[w]
        rebin_ivar = rebin_ivar[w]
        deltas[healpix].append(
            Delta(thingid[index2], ra[index2], dec[index2], z[index2],
                  thingid[index2], thingid[index2], thingid[index2],
                  rebin_log_lambda, rebin_ivar, None, rebin_flux, 1, None,
                  None, None, None, None, None))

    return deltas, stack_flux, stack_weight


def write_delta_from_transmission(deltas, mean_flux, flux_variance, healpix, out_filename,
                                  x_min, delta_x, use_old_weights=False, lin_spaced=False):

    """Write deltas to file for a given HEALPix pixel.
    Args:
        deltas: list
            List of delta objects contained in the given HEALPix pixel.
        mean_flux: array
            Average flux over all skewers.
        flux_variance: array
            Flux variance over all skewers.
        healpix: int
            The HEALPix pixel under consideration.
        out_filename: str
            Output filename.
        x_min: float
            Minimum observed wavelength or log wavelength in Angstrom
        delta_x: float
            Variation of the wavelength (or log wavelength) between two pixels
        use_old_weights: boolean - default: False
            Whether to use the old weights based only on the bin size
        lin_spaced: float - default: False
            Whether to use linear spacing for the wavelength binning
    """

    if len(deltas) == 0:
        userprint('No data in {}'.format(healpix))
        return

    sigma_lss_sq = None
    if flux_variance is not None:
        sigma_lss_sq = flux_variance / mean_flux**2

    results = fitsio.FITS(out_filename, 'rw', clobber=True)
    for delta in deltas:
        lambda_array = delta.log_lambda
        if lin_spaced:
            lambda_array = 10**(lambda_array)

        norm_lambda = (lambda_array - x_min) / delta_x + 0.5
        bins = np.floor(np.around(norm_lambda, decimals=3)).astype(int)
        # bins = np.floor((lambda_array - x_min) / delta_x + 0.5).astype(int)

        delta.delta = delta.delta / mean_flux[bins] - 1.

        if use_old_weights:
            delta.weights *= mean_flux[bins]**2
        else:
            assert sigma_lss_sq is not None
            delta.weights = 1 / sigma_lss_sq[bins]

        header = {}
        header['RA'] = delta.ra
        header['DEC'] = delta.dec
        header['Z'] = delta.z_qso
        header['PMF'] = '{}-{}-{}'.format(delta.plate, delta.mjd, delta.fiberid)
        header['THING_ID'] = delta.thingid
        header['PLATE'] = delta.plate
        header['MJD'] = delta.mjd
        header['FIBERID'] = delta.fiberid
        header['ORDER'] = delta.order

        cols = [
            delta.log_lambda, delta.delta, delta.weights,
            np.ones(delta.log_lambda.size)
        ]
        names = ['LOGLAM', 'DELTA', 'WEIGHT', 'CONT']
        results.write(cols,
                      names=names,
                      header=header,
                      extname=str(delta.thingid))
    results.close()

    return


def convert_transmission_to_deltas(obj_path, out_dir, in_dir=None, in_filenames=None,
                                   tracer='F_LYA', lambda_min=3600., lambda_max=5500.,
                                   lambda_min_rest_frame=1040.,
                                   lambda_max_rest_frame=1200.,
                                   delta_log_lambda=None, delta_lambda=None, lin_spaced=False,
                                   max_num_spec=None, nproc=None, use_old_weights=False,
                                   use_splines=False, out_healpix_order='RING'):
    """Convert transmission files to picca delta files

    Args:
        obj_path: str
            Path to the catalog of object to extract the transmission from
        out_dir: str
            Path to the directory where delta files will be written
        in_dir: str or None - default: None
            Path to the directory containing the transmission files directory.
            If 'None', then in_files must be a non-empty array
        in_filenames: array of str or None - default: None
            List of the filenames for the transmission files. Ignored if in_dir
            is not 'None'
        tracer: string - default: F_LYA.
            Tracer name for HDU in transmission file, usually F_LYA, F_LYB or F_METALS
        lambda_min: float - default: 3600.
            Minimum observed wavelength in Angstrom
        lambda_max: float - default: 5500.
            Maximum observed wavelength in Angstrom
        lambda_min_rest_frame: float - default: 1040.
            Minimum Rest Frame wavelength in Angstrom
        lambda_max_rest_frame: float - default: 1200.
            Maximum Rest Frame wavelength in Angstrom
        delta_log_lambda: float - default: None
            Variation of the logarithm of the wavelength between two pixels
        delta_lambda: float - default: None
            Variation of the wavelength between two pixels
        lin_spaced: float - default: False
            Whether to use linear spacing for the wavelength binning
        max_num_spec: int or None - default: None
            Maximum number of spectra to read. 'None' for no maximum
        nproc: int or None - default: None
            Number of cpus to use for I/O operations. If None, defaults to os.cpu_count().
        use_old_weights: boolean - default: False
            Whether to use the old weights based only on the bin size
        use_splines: boolean - default: False
            Whether to use spline interpolations for mean flux and sigma_lss (helps with stability)
        out_healpix_order: string: 'RING' or 'NEST' - default: 'RING'
            Healpix numbering scheme for output files.
    """
    # read catalog of objects
    hdul = fitsio.FITS(obj_path)
    hdu = hdul[1]
    key_val = set(hdu.get_colnames())

    # Get object id in HDU
    accepted_obj_ids = ['TARGETID', 'THING_ID', 'MOCKID']
    obj_keyid = key_val.intersection(accepted_obj_ids)
    if not obj_keyid:
        err_msg = f"Object ID has to be one of {', '.join(accepted_obj_ids)}"
        userprint(f"ERROR: {err_msg}")
        raise KeyError(err_msg)

    objs_thingid = hdu[obj_keyid.pop()][:]

    # moved hpx values here to read from master catalog header
    # rather than transmission files for ohio-p1d mocks
    hdr = hdu.read_header()
    if 'HPXNSIDE' in hdr and 'HPXNEST' in hdr:
        in_nside  = hdr['HPXNSIDE']
        is_nested = hdr['HPXNEST'] # true or false
    else:
        in_nside = None
        is_nested = None

    accepted_z_keys = ['Z', 'Z_QSO_RSD']
    z_key = key_val.intersection(accepted_z_keys).pop()
    if not z_key:
        err_msg = f"Z key has to be one of {', '.join(accepted_z_keys)}"
        userprint(f"ERROR: {err_msg}")
        raise KeyError(err_msg)

    w = hdu['Z'][:] > max(0., lambda_min / lambda_max_rest_frame - 1.)
    w &= hdu['Z'][:] < max(0., lambda_max / lambda_min_rest_frame - 1.)
    objs_ra = hdu['RA'][:][w].astype('float64') * np.pi / 180.
    objs_dec = hdu['DEC'][:][w].astype('float64') * np.pi / 180.
    objs_thingid = objs_thingid[w]
    hdul.close()
    userprint('INFO: Found {} quasars'.format(objs_ra.size))

    # Load list of transmission files
    if ((in_dir is None and in_filenames is None) or
            (in_dir is not None and in_filenames is not None)):
        userprint(("ERROR: No transmisson input files or both 'in_dir' and "
                   "'in_filenames' given"))
        sys.exit()
    elif in_dir is not None:
        # ohio-p1d transmissions are named lya-transmissions*
        files = np.array(sorted(glob.glob(in_dir + '/*/*/*transmission*.fits*')))

        if in_nside is None:
            hdul = fitsio.FITS(files[0])
            in_nside = hdul['METADATA'].read_header()['HPXNSIDE']
            is_nested = hdul['METADATA'].read_header()['HPXNEST']
            hdul.close()

        in_healpixs = healpy.ang2pix(in_nside,
                                     np.pi / 2. - objs_dec,
                                     objs_ra,
                                     nest=is_nested)
        if files[0].endswith('.gz'):
            end_of_file = '.gz'
        else:
            end_of_file = ''
    else:
        files = np.sort(np.array(in_filenames))
        is_nested = None
    userprint('INFO: Found {} files'.format(files.size))

    # Check if we should compute linear or log spaced deltas
    # Use the x_min/x_max/delta_x variables to stand in for either
    # linear of log spaced parameters
    if lin_spaced:
        x_min = lambda_min
        x_max = lambda_max
        delta_x = delta_lambda if delta_lambda is not None else 3.
    else:
        x_min = np.log10(lambda_min)
        x_max = np.log10(lambda_max)
        delta_x = delta_log_lambda if delta_lambda is not None else 3.e-4
    num_bins = int((x_max - x_min) / delta_x) + 1

    # Read the transmission files in parallel
    arguments = [(f, num_bins, objs_thingid, tracer, lambda_min, lambda_max,
                  lambda_min_rest_frame, lambda_max_rest_frame,
                  delta_log_lambda, delta_lambda, lin_spaced) for f in files]

    userprint("Reading transmission files...")
    with Pool(processes=nproc) as pool:
        read_results = pool.starmap(read_transmission_file, arguments)

    userprint("Done reading transmission files")

    # Read and merge the results
    stack_flux = np.zeros(num_bins)
    stack_weight = np.zeros(num_bins)
    deltas = {}
    for res in read_results:
        if res is not None:
            healpix_deltas = res[0]
            healpix_stack_flux = res[1]
            healpix_stack_weight = res[2]

            for key in healpix_deltas.keys():
                if key not in deltas.keys():
                    deltas[key] = []
                deltas[key] += healpix_deltas[key]

            stack_flux += healpix_stack_flux
            stack_weight += healpix_stack_weight

            num_spec = np.sum([len(deltas[healpix]) for healpix in deltas])
            if (max_num_spec is not None and num_spec >= max_num_spec):
                break

    userprint('\n')

    # normalize stacked transmission
    w = stack_weight > 0.
    mean_flux = stack_flux
    mean_flux[w] /= stack_weight[w]

    rebin_lambda = (x_min + np.arange(num_bins) * delta_x)
    mask = (mean_flux < 1) & (mean_flux > 0)
    mean_flux_spline = scipy.interpolate.UnivariateSpline(rebin_lambda[mask][10:-10],
                                                          mean_flux[mask][10:-10], k=5)
    mean_flux_from_spline = mean_flux_spline(rebin_lambda)

    # set mapping between input healpix and output healpix
    out_healpix_method = lambda in_nside, healpix: healpix

    # The first case doesn't make sense.
    # Why even convert from nested to ring at all?
    if is_nested is None and out_healpix_order is not None:
        raise ValueError('Input HEALPix scheme not known, cannot'
                         'convert to scheme {}'.format(out_healpix_order))
    elif is_nested and out_healpix_order.lower() == 'ring':
        out_healpix_method = lambda in_nside, healpix: healpy.nest2ring(int(in_nside), int(healpix))
    elif not is_nested and out_healpix_order.lower() == 'nest':
        out_healpix_method = lambda in_nside, healpix: healpy.ring2nest(int(in_nside), int(healpix))
    else:
        raise ValueError('HEALPix scheme {} not recognised'.format(
            out_healpix_order))

    #  save results
    out_filenames = {}
    for healpix in sorted(deltas):
        out_healpix = out_healpix_method(in_nside, healpix)

        print(f'Input nested? {is_nested} // in_healpix={healpix} // out_healpix={out_healpix}')
        out_filenames[healpix] = out_dir + '/delta-{}'.format(out_healpix) + '.fits.gz'

    # Compute variance
    stack_variance = np.zeros(len(mean_flux))
    var_weights = np.zeros(len(mean_flux))
    for hpix_deltas in deltas.values():
        for delta in hpix_deltas:
            lambda_array = delta.log_lambda
            if lin_spaced:
                lambda_array = 10**(lambda_array)

            norm_lambda = (lambda_array - x_min) / delta_x + 0.5
            bins = np.floor(np.around(norm_lambda, decimals=3)).astype(int)

            stack_variance[bins] += (delta.delta - mean_flux[bins])**2
            var_weights[bins] += np.ones(len(bins))

    w = var_weights > 0.
    flux_variance = stack_variance
    flux_variance[w] /= var_weights[w]

    mask = (flux_variance > 0)
    flux_variance_spline = scipy.interpolate.UnivariateSpline(rebin_lambda[mask][10:-10],
                                                              flux_variance[mask][10:-10], k=5)
    flux_variance_from_spline = flux_variance_spline(rebin_lambda)

    if use_splines:
        arguments = [(deltas[hpix], mean_flux_from_spline, flux_variance_from_spline, hpix,
                      out_filenames[hpix], x_min, delta_x, use_old_weights,
                      lin_spaced) for hpix in deltas.keys()]
    else:
        arguments = [(deltas[hpix], mean_flux, flux_variance, hpix, out_filenames[hpix],
                      x_min, delta_x, use_old_weights, lin_spaced) for hpix in deltas.keys()]

    userprint("Writing deltas...")
    with Pool(processes=nproc) as pool:
        pool.starmap(write_delta_from_transmission, arguments)

    # Output the mean flux and other info
    dir_name = os.path.basename(os.path.normpath(out_dir))
    filename = out_dir + f'/../{dir_name}-stats.fits.gz'
    userprint(f"Writing statistics to {filename}")

    results = fitsio.FITS(filename, 'rw', clobber=True)
    cols = [rebin_lambda, mean_flux, mean_flux_from_spline, stack_weight,
            flux_variance, flux_variance_from_spline, var_weights]
    names = ['LAMBDA', 'MEANFLUX', 'MEANFLUX_SPLINE', 'WEIGHTS', 'VAR', 'VAR_SPLINE', 'VARWEIGHTS']
    header = {}
    header['L_MIN'] = lambda_min
    header['L_MAX'] = lambda_max
    header['LR_MIN'] = lambda_min_rest_frame
    header['LR_MAX'] = lambda_max_rest_frame
    header['DEL_LL'] = delta_log_lambda
    header['DEL_L'] = delta_lambda
    header['LINEAR'] = lin_spaced
    results.write(cols, names=names, header=header, extname='STATS')
    results.close()

    userprint("")
