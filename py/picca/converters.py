"""This module provides functions to format several catalogues

These are:
    - eboss_convert_dla
    - desi_convert_dla
    - desi_from_truth_to_drq
    - desi_from_ztarget_to_drq
    - desi_convert_transmission_to_delta_files
See the respective docstrings for more details
"""
import os
import sys
import glob
import numpy as np
import fitsio
import healpy

from picca.data import Delta
from picca.utils import userprint


def eboss_convert_dla(in_path, drq_filename, out_path, drq_z_key='Z'):
    """Converts Pasquier Noterdaeme ASCII DLA catalog to a fits file

    Args:
        in_path: string
            Full path filename containing the ASCII DLA catalogue
        drq_filename: string
            Filename of the DRQ catalogue
        out_path: string
            Full path filename where the fits DLA catalogue will be written to
        drq_z_key: string
            Name of the column of DRQ containing the quasrs redshift
    """
    # Read catalogue
    filename = open(os.path.expandvars(in_path), 'r')
    for line in filename:
        cols = line.split()
        if (len(cols) == 0) or (cols[0][0] == '#') or (cols[0][0] == '-'):
            continue
        if cols[0] == 'ThingID':
            from_key_to_index = {key: index for index, key in enumerate(cols)}
            dla_cat = {key: [] for key in from_key_to_index}
            for key in 'MJD-plate-fiber'.split('-'):
                dla_cat[key] = []
            continue
        for key in from_key_to_index.keys():
            value = cols[from_key_to_index[key]]
            if key == 'MJD-plate-fiber':
                for key2, value2 in zip('MJD-plate-fiber'.split('-'),
                                        value.split('-')):
                    dla_cat[key2] += [value2]
            dla_cat[key] += [value]
    filename.close()
    userprint(("INFO: Found {} DLA from {} "
               "quasars").format(len(dla_cat['ThingID']),
                                 np.unique(dla_cat['ThingID']).size))

    # convert Noterdaemem keys to picca keys
    from_noterdaeme_key_to_picca_key = {
        'ThingID': 'THING_ID',
        'z_abs': 'Z',
        'zqso': 'ZQSO',
        'NHI': 'NHI',
        'plate': 'PLATE',
        'MJD': 'MJD',
        'fiber': 'FIBERID',
        'RA': 'RA',
        'Dec': 'DEC'
    }
    # define types
    from_picca_key_to_type = {
        'THING_ID': np.int64,
        'Z': np.float64,
        'ZQSO': np.float64,
        'NHI': np.float64,
        'PLATE': np.int64,
        'MJD': np.int64,
        'FIBERID': np.int64,
        'RA': np.float64,
        'DEC': np.float64
    }

    # format catalogue
    cat = {
        value: np.array(dla_cat[key], dtype=from_picca_key_to_type[value])
        for key, value in from_noterdaeme_key_to_picca_key.items()
    }

    # apply cuts
    w = cat['THING_ID'] > 0
    userprint(("INFO: Removed {} DLA, because "
               "THING_ID<=0").format((cat['THING_ID'] <= 0).sum()))
    w &= cat['Z'] > 0.
    userprint(("INFO: Removed {} DLA, because "
               "Z<=0.").format((cat['Z'] <= 0.).sum()))
    for key in cat:
        cat[key] = cat[key][w]

    # update RA, DEC, and Z_QSO from DRQ catalogue
    hdul = fitsio.FITS(drq_filename)
    thingid = hdul[1]['THING_ID'][:]
    ra = hdul[1]['RA'][:]
    dec = hdul[1]['DEC'][:]
    z_qso = hdul[1][drq_z_key][:]
    hdul.close()
    from_thingid_to_index = {t: index for index, t in enumerate(thingid)}
    cat['RA'] = np.array(
        [ra[from_thingid_to_index[t]] for t in cat['THING_ID']])
    cat['DEC'] = np.array(
        [dec[from_thingid_to_index[t]] for t in cat['THING_ID']])
    cat['ZQSO'] = np.array(
        [z_qso[from_thingid_to_index[t]] for t in cat['THING_ID']])

    # apply cuts
    w = cat['RA'] != cat['DEC']
    userprint(("INFO: Removed {} DLA, because "
               "RA==DEC").format((cat['RA'] == cat['DEC']).sum()))
    w &= cat['RA'] != 0.
    userprint(("INFO: Removed {} DLA, because "
               "RA==0").format((cat['RA'] == 0.).sum()))
    w &= cat['DEC'] != 0.
    userprint(("INFO: Removed {} DLA, because "
               "DEC==0").format((cat['DEC'] == 0.).sum()))
    w &= cat['ZQSO'] > 0.
    userprint(("INFO: Removed {} DLA, because "
               "ZQSO<=0.").format((cat['ZQSO'] <= 0.).sum()))
    for key in cat:
        cat[key] = cat[key][w]

    # sort first by redshift
    w = np.argsort(cat['Z'])
    for key in cat.keys():
        cat[key] = cat[key][w]
    # then by thingid
    w = np.argsort(cat['THING_ID'])
    for key in cat:
        cat[key] = cat[key][w]
    # add DLA ID
    cat['DLAID'] = np.arange(1, cat['Z'].size + 1, dtype=np.int64)

    for key in ['RA', 'DEC']:
        cat[key] = cat[key].astype('float64')

    # Save catalogue
    results = fitsio.FITS(out_path, 'rw', clobber=True)
    cols = list(cat.values())
    names = list(cat)
    results.write(cols, names=names, extname='DLACAT')
    results.close()


def desi_convert_dla(in_path, out_path):
    """Convert a catalog of DLA from a DESI format to the format used by picca

    Args:
        in_path: string
            Full path filename containing the ASCII DLA catalogue
        out_path: string
            Full path filename where the fits DLA catalogue will be written to
    """
    from_desi_key_to_picca_key = {
        'RA': 'RA',
        'DEC': 'DEC',
        'Z': 'Z_DLA_RSD',
        'ZQSO': 'Z_QSO_RSD',
        'NHI': 'N_HI_DLA',
        'THING_ID': 'MOCKID',
        'DLAID': 'DLAID',
        'PLATE': 'MOCKID',
        'MJD': 'MOCKID',
        'FIBERID': 'MOCKID',
    }
    # read catalogue
    cat = {}
    hdul = fitsio.FITS(in_path)
    for key, value in from_desi_key_to_picca_key.items():
        cat[key] = hdul['DLACAT'][value][:]
    hdul.close()
    userprint(("INFO: Found {} DLA from {} "
               "quasars").format(cat['Z'].size,
                                 np.unique(cat['THING_ID']).size))
    # sort by THING_ID
    w = np.argsort(cat['THING_ID'])
    for key in cat:
        cat[key] = cat[key][w]

    for key in ['RA', 'DEC']:
        cat[key] = cat[key].astype('float64')

    # save results
    results = fitsio.FITS(out_path, 'rw', clobber=True)
    cols = list(cat.values())
    names = list(cat)
    results.write(cols, names=names, extname='DLACAT')
    results.close()


def desi_from_truth_to_drq(truth_filename,
                           targets_filename,
                           out_path,
                           spec_type="QSO"):
    """Transform a desi truth.fits file and a desi targets.fits into a drq
    like file

    Args:
        truth_filename: string
            Filename of the truth.fits file
        targets_filename: string
            Filename of the desi targets.fits file
        out_path: string
            Full path filename where the fits catalogue will be written to
        spec_type: string
            Spectral type of the objects to include in the catalogue
    """
    # read truth table
    hdul = fitsio.FITS(truth_filename)

    # apply cuts
    w = np.ones(hdul[1]['TARGETID'][:].size).astype(bool)
    userprint(" start                 : nb object in cat = {}".format(w.sum()))
    w &= np.char.strip(hdul[1]['TRUESPECTYPE'][:].astype(str)) == spec_type
    userprint(" and TRUESPECTYPE=={}  : nb object in cat = {}".format(
        spec_type, w.sum()))
    # load the arrays
    thingid = hdul[1]['TARGETID'][:][w]
    z_qso = hdul[1]['TRUEZ'][:][w]
    hdul.close()
    ra = np.zeros(thingid.size)
    dec = np.zeros(thingid.size)
    plate = thingid
    mjd = thingid
    fiberid = thingid

    ### Get RA and DEC from targets
    hdul = fitsio.FITS(targets_filename)
    thingid_targets = hdul[1]['TARGETID'][:]
    ra_targets = hdul[1]['RA'][:].astype('float64')
    dec_targets = hdul[1]['DEC'][:].astype('float64')
    hdul.close()

    from_targetid_to_index = {}
    for index, t in enumerate(thingid_targets):
        from_targetid_to_index[t] = index
    keys_from_targetid_to_index = from_targetid_to_index.keys()

    for index, t in enumerate(thingid):
        if t not in keys_from_targetid_to_index:
            continue
        index2 = from_targetid_to_index[t]
        ra[index] = ra_targets[index2]
        dec[index] = dec_targets[index2]

    # apply cuts
    if (ra == 0.).sum() != 0 or (dec == 0.).sum() != 0:
        w = ra != 0.
        w &= dec != 0.
        userprint((" and RA and DEC        : nb object in cat = "
                   "{}").format(w.sum()))

        ra = ra[w]
        dec = dec[w]
        z_qso = z_qso[w]
        thingid = thingid[w]
        plate = plate[w]
        mjd = mjd[w]
        fiberid = fiberid[w]

    # save catalogue
    results = fitsio.FITS(out_path, 'rw', clobber=True)
    cols = [ra, dec, thingid, plate, mjd, fiberid, z_qso]
    names = ['RA', 'DEC', 'THING_ID', 'PLATE', 'MJD', 'FIBERID', 'Z']
    results.write(cols, names=names, extname='CAT')
    results.close()


def desi_from_ztarget_to_drq(in_path,
                             out_path,
                             spec_type='QSO',
                             downsampling_z_cut=None,
                             downsampling_num=None):
    """Transforms a catalog of object in desi format to a catalog in DRQ format

    Args:
        in_path: string
            Full path filename containing the catalogue of objects
        out_path: string
            Full path filename where the fits DLA catalogue will be written to
        spec_type: string
            Spectral type of the objects to include in the catalogue
        downsampling_z_cut: float or None - default: None
            Minimum redshift to downsample the data. 'None' for no downsampling
        downsampling_num: int
            Target number of object above redshift downsampling-z-cut.
            'None' for no downsampling
    """

    ## Info of the primary observation
    hdul = fitsio.FITS(in_path)
    spec_type_list = np.char.strip(hdul[1]['SPECTYPE'][:].astype(str))

    # apply cuts
    userprint((" start               : nb object in cat = "
               "{}").format(spec_type_list.size))
    w = hdul[1]['ZWARN'][:] == 0.
    userprint(' and zwarn==0        : nb object in cat = {}'.format(w.sum()))
    w &= spec_type_list == spec_type
    userprint(' and spectype=={}    : nb object in cat = {}'.format(
        spec_type, w.sum()))
    # load the arrays
    cat = {}
    from_desi_key_to_picca_key = {
        'RA': 'RA',
        'DEC': 'DEC',
        'Z': 'Z',
        'THING_ID': 'TARGETID',
        'PLATE': 'TARGETID',
        'MJD': 'TARGETID',
        'FIBERID': 'TARGETID'
    }
    for key, value in from_desi_key_to_picca_key.items():
        cat[key] = hdul[1][value][:][w]
    hdul.close()

    for key in ['RA', 'DEC']:
        cat[key] = cat[key].astype('float64')

    # apply downsampling
    if downsampling_z_cut is not None and downsampling_num is not None:
        if cat['RA'].size < downsampling_num:
            userprint(("WARNING:: Trying to downsample, when nb cat = {} and "
                       "nb downsampling = {}").format(cat['RA'].size,
                                                      downsampling_num))
        else:
            z_cut_num = (cat['Z'] > downsampling_z_cut).sum()
            select_fraction = (downsampling_num / z_cut_num)
            if select_fraction < 1.0:
                np.random.seed(0)
                w = np.random.choice(np.arange(cat['RA'].size),
                                 size=int(cat['RA'].size * select_fraction),
                                 replace=False)
                for key in cat:
                    cat[key] = cat[key][w]
                userprint((" and downsampling : nb object in cat = {}, nb z > "
                       "{} = {}").format(cat['RA'].size, downsampling_z_cut,
                                        z_cut_num))
            else:
                userprint(("WARNING:: Trying to downsample, when nb QSOs with "
                           "z > {} = {} and downsampling = {}").format
                           (downsampling_z_cut, z_cut_num, downsampling_num))

    # sort by THING_ID
    w = np.argsort(cat['THING_ID'])
    for key in cat:
        cat[key] = cat[key][w]

    # save catalogue
    results = fitsio.FITS(out_path, 'rw', clobber=True)
    cols = list(cat.values())
    names = list(cat)
    results.write(cols, names=names, extname='CAT')
    results.close()


def desi_convert_transmission_to_delta_files(obj_path,
                                             out_dir,
                                             in_dir=None,
                                             in_filenames=None,
                                             lambda_min=3600.,
                                             lambda_max=5500.,
                                             lambda_min_rest_frame=1040.,
                                             lambda_max_rest_frame=1200.,
                                             delta_log_lambda=3.e-4,
                                             max_num_spec=None):
    """Convert desi transmission files to picca delta files

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
        lambda_min: float - default: 3600.
            Minimum observed wavelength in Angstrom
        lambda_max: float - default: 5500.
            Maximum observed wavelength in Angstrom
        lambda_min_rest_frame: float - default: 1040.
            Minimum Rest Frame wavelength in Angstrom
        lambda_max_rest_frame: float - default: 1200.
            Maximum Rest Frame wavelength in Angstrom
        delta_log_lambda: float - default: 3.e-4
            Variation of the logarithm of the wavelength between two pixels
        max_num_spec: int or None - default: None
            Maximum number of spectra to read. 'None' for no maximum
    """
    # read catalog of objects
    hdul = fitsio.FITS(obj_path)
    key_val = np.char.strip(
        np.array([
            hdul[1].read_header()[key] for key in hdul[1].read_header().keys()
        ]).astype(str))
    if 'TARGETID' in key_val:
        objs_thingid = hdul[1]['TARGETID'][:]
    elif 'THING_ID' in key_val:
        objs_thingid = hdul[1]['THING_ID'][:]
    w = hdul[1]['Z'][:] > max(0., lambda_min / lambda_max_rest_frame - 1.)
    w &= hdul[1]['Z'][:] < max(0., lambda_max / lambda_min_rest_frame - 1.)
    objs_ra = hdul[1]['RA'][:][w].astype('float64') * np.pi / 180.
    objs_dec = hdul[1]['DEC'][:][w].astype('float64') * np.pi / 180.
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
        files = glob.glob(in_dir + '/*/*/transmission*.fits*')
        files = np.sort(np.array(files))
        hdul = fitsio.FITS(files[0])
        in_nside = hdul['METADATA'].read_header()['HPXNSIDE']
        nest = hdul['METADATA'].read_header()['HPXNEST']
        hdul.close()
        in_healpixs = healpy.ang2pix(in_nside,
                                     np.pi / 2. - objs_dec,
                                     objs_ra,
                                     nest=nest)
        if files[0].endswith('.gz'):
            end_of_file = '.gz'
        else:
            end_of_file = ''
        files = np.sort(
            np.array([("{}/{}/{healpix}/transmission-{}-{healpix}"
                       ".fits{}").format(in_dir,
                                         int(healpix // 100),
                                         in_nside,
                                         end_of_file,
                                         healpix=healpix)
                      for healpix in np.unique(in_healpixs)]))
    else:
        files = np.sort(np.array(in_filenames))
    userprint('INFO: Found {} files'.format(files.size))

    # Stack the deltas transmission
    log_lambda_min = np.log10(lambda_min)
    log_lambda_max = np.log10(lambda_max)
    num_bins = int((log_lambda_max - log_lambda_min) / delta_log_lambda) + 1
    stack_delta = np.zeros(num_bins)
    stack_weight = np.zeros(num_bins)

    deltas = {}

    # read deltas
    for index, filename in enumerate(files):
        userprint("\rread {} of {} {}".format(
            index, files.size,
            np.sum([len(deltas[healpix]) for healpix in deltas])),
                  end="")
        hdul = fitsio.FITS(filename)
        thingid = hdul['METADATA']['MOCKID'][:]
        if np.in1d(thingid, objs_thingid).sum() == 0:
            hdul.close()
            continue
        ra = hdul['METADATA']['RA'][:].astype(np.float64) * np.pi / 180.
        dec = hdul['METADATA']['DEC'][:].astype(np.float64) * np.pi / 180.
        z = hdul['METADATA']['Z'][:]
        log_lambda = np.log10(hdul['WAVELENGTH'].read())
        if 'F_LYA' in hdul:
            trans = hdul['F_LYA'].read()
        else:
            trans = hdul['TRANSMISSION'].read()

        num_obj = z.size
        healpix = filename.split('-')[-1].split('.')[0]

        if trans.shape[0] != num_obj:
            trans = trans.transpose()

        bins = np.floor((log_lambda - log_lambda_min) / delta_log_lambda +
                        0.5).astype(int)
        aux_log_lambda = log_lambda_min + bins * delta_log_lambda
        lambda_obs_frame =  (10**aux_log_lambda) * np.ones(num_obj)[:, None]
        lambda_rest_frame = (10**aux_log_lambda) / (1. + z[:, None])
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
            continue

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
            aux_log_lambda = log_lambda[valid_pixels[index2, :] > 0]
            aux_trans = trans[index2, :][valid_pixels[index2, :] > 0]

            bins = np.floor((aux_log_lambda - log_lambda_min) /
                            delta_log_lambda + 0.5).astype(int)
            rebin_log_lambda = (log_lambda_min +
                                np.arange(num_bins) * delta_log_lambda)
            rebin_flux = np.bincount(bins,
                                     weights=aux_trans,
                                     minlength=num_bins)
            rebin_ivar = np.bincount(bins, minlength=num_bins).astype(float)

            w = rebin_ivar > 0.
            if w.sum() < 50:
                continue
            stack_delta += rebin_flux
            stack_weight += rebin_ivar
            rebin_log_lambda = rebin_log_lambda[w]
            rebin_flux = rebin_flux[w] / rebin_ivar[w]
            rebin_ivar = rebin_ivar[w]
            deltas[healpix].append(
                Delta(thingid[index2], ra[index2], dec[index2], z[index2],
                      thingid[index2], thingid[index2], thingid[index2],
                      rebin_log_lambda, rebin_ivar, None, rebin_flux, 1, None,
                      None, None, None, None, None))
        if (max_num_spec is not None and
                np.sum([len(deltas[healpix])
                        for healpix in deltas]) >= max_num_spec):
            break

    userprint('\n')

    # normalize stacked transmission
    w = stack_weight > 0.
    stack_delta[w] /= stack_weight[w]

    #  save results
    for index, healpix in enumerate(sorted(deltas)):
        if len(deltas[healpix]) == 0:
            userprint('No data in {}'.format(healpix))
            continue
        results = fitsio.FITS(out_dir + '/delta-{}'.format(healpix) +
                              '.fits.gz',
                              'rw',
                              clobber=True)
        for delta in deltas[healpix]:
            bins = np.floor((delta.log_lambda - log_lambda_min) /
                            delta_log_lambda + 0.5).astype(int)
            delta.delta = delta.delta / stack_delta[bins] - 1.
            delta.weights *= stack_delta[bins]**2

            header = {}
            header['RA'] = delta.ra
            header['DEC'] = delta.dec
            header['Z'] = delta.z_qso
            header['PMF'] = '{}-{}-{}'.format(delta.plate, delta.mjd,
                                              delta.fiberid)
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
        userprint("\rwrite {} of {}: {} quasars".format(index, len(deltas),
                                                        len(deltas[healpix])),
                  end="")

    userprint("")
