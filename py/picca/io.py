"""This module defines a set of functions to manage reading of data.

This module several functions to read different types of data:
    - read_dlas
    - read_drq
    - read_blinding
    - read_delta_file
    - read_deltas
    - read_objects
See the respective documentation for details
"""
import glob
import sys
import time
import os.path
import copy
import numpy as np
import healpy
import fitsio
from astropy.table import Table
import warnings
from multiprocessing import Pool

from .utils import userprint
from .data import Delta, QSO
from .pk1d.prep_pk1d import exp_diff, spectral_resolution
from .pk1d.prep_pk1d import spectral_resolution_desi


def read_dlas(filename,obj_id_name='THING_ID'):
    """Reads the DLA catalog from a fits file.

    ASCII or DESI files can be converted using:
        utils.eBOSS_convert_DLA()
        utils.desi_convert_DLA()

    Args:
        filename: str
            File containing the DLAs

    Returns:
        A dictionary with the DLA's information. Keys are the THING_ID
        associated with the DLA. Values are a tuple with its redshift and
        column density.
    """
    userprint('Reading DLA catalog from:', filename)

    columns_list = [obj_id_name, 'Z', 'NHI']
    hdul = fitsio.FITS(filename)
    cat = {col: hdul['DLACAT'][col][:] for col in columns_list}
    hdul.close()

    # sort the items in the dictionary according to THING_ID and redshift
    w = np.argsort(cat['Z'])
    for key in cat.keys():
        cat[key] = cat[key][w]
    w = np.argsort(cat[obj_id_name])
    for key in cat.keys():
        cat[key] = cat[key][w]

    # group DLAs on the same line of sight together
    dlas = {}
    for thingid in np.unique(cat[obj_id_name]):
        w = (thingid == cat[obj_id_name])
        dlas[thingid] = list(zip(cat['Z'][w], cat['NHI'][w]))
    num_dlas = np.sum([len(dla) for dla in dlas.values()])

    userprint(' In catalog: {} DLAs'.format(num_dlas))
    userprint(' In catalog: {} forests have a DLA'.format(len(dlas)))
    userprint('\n')

    return dlas


def read_drq(drq_filename,
             z_min=0,
             z_max=10.,
             keep_bal=False,
             bi_max=None,
             mode='sdss'):
    """Reads the quasars in the DRQ quasar catalog.

    Args:
        drq_filename: str
            Filename of the DRQ catalogue
        z_min: float - default: 0.
            Minimum redshift. Quasars with redshifts lower than z_min will be
            discarded
        z_max: float - default: 10.
            Maximum redshift. Quasars with redshifts higher than or equal to
            z_max will be discarded
        keep_bal: bool - default: False
            If False, remove the quasars flagged as having a Broad Absorption
            Line. Ignored if bi_max is not None
        bi_max: float or None - default: None
            Maximum value allowed for the Balnicity Index to keep the quasar

    Returns:
        catalog: astropy.table.Table
            Table containing the metadata of the selected objects
    """
    userprint('Reading catalog from ', drq_filename)
    catalog = Table(fitsio.read(drq_filename, ext=1))

    keep_columns = ['RA', 'DEC', 'Z']

    if 'desi' in mode and 'TARGETID' in catalog.colnames:
        obj_id_name = 'TARGETID'
        if 'TARGET_RA' in catalog.colnames:
            catalog.rename_column('TARGET_RA', 'RA')
            catalog.rename_column('TARGET_DEC', 'DEC')
        keep_columns += ['TARGETID']
        if 'TILEID' in catalog.colnames:
            keep_columns += ['TILEID', 'PETAL_LOC']
        if 'FIBER' in catalog.colnames:
            keep_columns += ['FIBER']
        if 'SURVEY' in catalog.colnames:
            keep_columns += ['SURVEY']
        if 'DESI_TARGET' in catalog.colnames:
            keep_columns += ['DESI_TARGET']
        if 'SV1_DESI_TARGET' in catalog.colnames:
            keep_columns += ['SV1_DESI_TARGET']
        if 'SV3_DESI_TARGET' in catalog.colnames:
            keep_columns += ['SV3_DESI_TARGET']


    else:
        obj_id_name = 'THING_ID'
        keep_columns += ['THING_ID', 'PLATE', 'MJD', 'FIBERID']

    if mode == "desi_mocks":
        for key in ['RA', 'DEC']:
            catalog[key] = catalog[key].astype('float64')

    ## Redshift
    if 'Z' not in catalog.colnames:
        if 'Z_VI' in catalog.colnames:
            catalog.rename_column('Z_VI', 'Z')
            userprint(
                "Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)"
            )
        else:
            userprint("ERROR: No valid column for redshift found in ",
                      drq_filename)
            return None

    ## Sanity checks
    userprint('')
    w = np.ones(len(catalog), dtype=bool)
    userprint(f" start                 : nb object in cat = {np.sum(w)}")
    w &= catalog[obj_id_name] > 0
    userprint(f" and {obj_id_name} > 0       : nb object in cat = {np.sum(w)}")
    w &= catalog['RA'] != catalog['DEC']
    userprint(f" and ra != dec         : nb object in cat = {np.sum(w)}")
    w &= catalog['RA'] != 0.
    userprint(f" and ra != 0.          : nb object in cat = {np.sum(w)}")
    w &= catalog['DEC'] != 0.
    userprint(f" and dec != 0.         : nb object in cat = {np.sum(w)}")

    ## Redshift range
    w &= catalog['Z'] >= z_min
    userprint(f" and z >= {z_min}        : nb object in cat = {np.sum(w)}")
    w &= catalog['Z'] < z_max
    userprint(f" and z < {z_max}         : nb object in cat = {np.sum(w)}")

    ## BAL visual
    if not keep_bal and bi_max is None:
        if 'BAL_FLAG_VI' in catalog.colnames:
            bal_flag = catalog['BAL_FLAG_VI']
            w &= bal_flag == 0
            userprint(
                f" and BAL_FLAG_VI == 0  : nb object in cat = {np.sum(w)}")
            keep_columns += ['BAL_FLAG_VI']
        else:
            userprint("WARNING: BAL_FLAG_VI not found")

    ## BAL CIV
    if bi_max is not None:
        if 'BI_CIV' in catalog.colnames:
            bi = catalog['BI_CIV']
            w &= bi <= bi_max
            userprint(
                f" and BI_CIV <= {bi_max}  : nb object in cat = {np.sum(w)}")
            keep_columns += ['BI_CIV']
        else:
            userprint("ERROR: --bi-max set but no BI_CIV field in HDU")
            sys.exit(0)

    #-- DLA Column density
    if 'NHI' in catalog.colnames:
        keep_columns += ['NHI']

    if 'LAST_NIGHT' in catalog.colnames:
        keep_columns += ['LAST_NIGHT']
        if 'FIRST_NIGHT' in catalog.colnames:
            keep_columns += ['FIRST_NIGHT']
    elif 'NIGHT' in catalog.colnames:
        keep_columns += ['NIGHT']

    catalog.keep_columns(keep_columns)
    w = np.where(w)[0]
    catalog = catalog[w]

    #-- Convert angles to radians
    catalog['RA'] = np.radians(catalog['RA'])
    catalog['DEC'] = np.radians(catalog['DEC'])


    return catalog


def read_blinding(in_dir):
    """Checks the delta files for blinding settings

    Args:
        in_dir: str
            Directory to spectra files. If mode is "spec-mock-1D", then it is
            the filename of the fits file contianing the mock spectra

    Returns:
        The following variables:
            blinding: True if data is blinded and False otherwise
    """
    files = []
    in_dir = os.path.expandvars(in_dir)
    if len(in_dir) > 8 and in_dir[-8:] == '.fits.gz':
        files += glob.glob(in_dir)
    elif len(in_dir) > 5 and in_dir[-5:] == '.fits':
        files += glob.glob(in_dir)
    else:
        files += glob.glob(in_dir + '/*.fits') + glob.glob(in_dir
                                                           + '/*.fits.gz')
    filename = files[0]
    hdul = fitsio.FITS(filename)
    if "LAMBDA" in hdul: # This is for ImageHDU format
        header = hdul["METADATA"].read_header()
        blinding = header["BLINDING"]
    else: # This is for BinTable format
        header = hdul[1].read_header()
        if "BLINDING" in header:
            blinding = header["BLINDING"]
        else:
            blinding = "none"

    return blinding


def read_delta_file(filename, z_min_qso=0, z_max_qso=10, rebin_factor=None):
    """Extracts deltas from a single file.
    Args:
        filename: str
            Path to the file to read
        z_min_qso: float - default: 0
            Specifies the minimum redshift for QSOs
        z_max_qso: float - default: 10
            Specifies the maximum redshift for QSOs
        rebin_factor: int - default: None
            Factor to rebin the lambda grid by. If None, no rebinning is done.
    Returns:
        deltas:
            A dictionary with the data. Keys are the healpix numbers of each
                spectrum. Values are lists of delta instances.
    """

    hdul = fitsio.FITS(filename)
    # If there is an extension called lambda format is image
    if 'LAMBDA' in hdul:
        deltas = Delta.from_image(hdul, z_min_qso=z_min_qso, z_max_qso=z_max_qso)
    else:
        deltas = [Delta.from_fitsio(hdu) for hdu in hdul[1:] if z_min_qso<hdu.read_header()['Z']<z_max_qso]

# Rebin
    if rebin_factor is not None:
        if 'LAMBDA' in hdul:
            card = 'LAMBDA'
        else:
            card = 1

        if hdul[card].read_header()['WAVE_SOLUTION'] != 'lin':
            raise ValueError('Delta rebinning only implemented for linear \
                    lambda bins')
        
        dwave = hdul[card].read_header()['DELTA_LAMBDA']
            
        for i in range(len(deltas)):
            deltas[i].rebin(rebin_factor, dwave=dwave)
            
    hdul.close()

    return deltas


def read_deltas(in_dir,
                nside,
                lambda_abs,
                alpha,
                z_ref,
                cosmo,
                max_num_spec=None,
                no_project=False,
                nproc=None,
                rebin_factor=None,
                z_min_qso=0,
                z_max_qso=10):
    """Reads deltas and computes their redshifts.

    Fills the fields delta.z and multiplies the weights by
        `(1+z)^(alpha-1)/(1+z_ref)^(alpha-1)`
    (equation 7 of du Mas des Bourboux et al. 2020)

    Args:
        in_dir: str
            Directory to spectra files. If mode is "spec-mock-1D", then it is
            the filename of the fits file contianing the mock spectra
        nside: int
            The healpix nside parameter
        lambda_abs: float
            Wavelength of the absorption (in Angstroms)
        alpha: float
            Redshift evolution coefficient (see equation 7 of du Mas des
            Bourboux et al. 2020)
        z_ref: float
            Redshift of reference
        cosmo: constants.Cosmo
            The fiducial cosmology
        max_num_spec: int or None - default: None
            Maximum number of spectra to read
        no_project: bool - default: False
            If False, project the deltas (see equation 5 of du Mas des Bourboux
            et al. 2020)
        nproc: int - default: None
            Number of cpus for parallelization. If None, uses all available.
        rebin_factor: int - default: None
            Factor to rebin the lambda grid by. If None, no rebinning is done.
        z_min_qso: float - default: 0
            Specifies the minimum redshift for QSOs
        z_max_qso: float - default: 10
            Specifies the maximum redshift for QSOs

    Returns:
        The following variables:
            data: A dictionary with the data. Keys are the healpix numbers of
                each spectrum. Values are lists of delta instances.
            num_data: Number of spectra in data.
            z_min: Minimum redshift of the loaded deltas.
            z_max: Maximum redshift of the loaded deltas.

    Raises:
        AssertionError: if no healpix numbers are found
    """
    files = []
    in_dir = os.path.expandvars(in_dir)

    if len(in_dir) > 8 and in_dir[-8:] == '.fits.gz':
        files += sorted(glob.glob(in_dir))
    elif len(in_dir) > 5 and in_dir[-5:] == '.fits':
        files += sorted(glob.glob(in_dir))
    else:
        files += sorted(glob.glob(in_dir + '/*.fits') + glob.glob(in_dir +
                                                            '/*.fits.gz'))
    files = sorted(files)

    if rebin_factor is not None:
        userprint(f"Rebinning deltas by a factor of {rebin_factor}\n")

    arguments = [(f, z_min_qso, z_max_qso, rebin_factor) for f in files]
    pool = Pool(processes=nproc)
    results = pool.starmap(read_delta_file, arguments)
    pool.close()

    deltas = []
    num_data = 0
    for delta in results:
        if delta is not None:
            deltas += delta
            num_data = len(deltas)
            if (max_num_spec is not None) and (num_data > max_num_spec):
                break

    # truncate the deltas if we load too many lines of sight
    if max_num_spec is not None:
        deltas = deltas[:max_num_spec]
        num_data = len(deltas)

    userprint("\n")

    # compute healpix numbers
    phi = [delta.ra for delta in deltas]
    theta = [np.pi / 2. - delta.dec for delta in deltas]
    healpixs = healpy.ang2pix(nside, theta, phi)
    if healpixs.size == 0:
        raise AssertionError('ERROR: No data in {}'.format(in_dir))

    data = {}
    z_min = 10**deltas[0].log_lambda[0] / lambda_abs - 1.
    z_max = 0.
    for delta, healpix in zip(deltas, healpixs):
        z = 10**delta.log_lambda / lambda_abs - 1.
        z_min = min(z_min, z.min())
        z_max = max(z_max, z.max())
        delta.z = z
        if not cosmo is None:
            delta.r_comov = cosmo.get_r_comov(z)
            delta.dist_m = cosmo.get_dist_m(z)
        delta.weights *= ((1 + z) / (1 + z_ref))**(alpha - 1)

        if not no_project:
            delta.project()

        if not healpix in data:
            data[healpix] = []
        data[healpix].append(delta)

    return data, num_data, z_min, z_max


def read_objects(filename,
                 nside,
                 z_min,
                 z_max,
                 alpha,
                 z_ref,
                 cosmo,
                 mode='sdss',
                 keep_bal=True):
    """Reads objects and computes their redshifts.

    Fills the fields delta.z and multiplies the weights by
        `(1+z)^(alpha-1)/(1+z_ref)^(alpha-1)`
    (equation 7 of du Mas des Bourboux et al. 2020)

    Args:
        filename: str
            Filename of the objects catalogue (must follow DRQ catalogue
            structure)
        nside: int
            The healpix nside parameter
        z_min: float
            Minimum redshift. Quasars with redshifts lower than z_min will be
            discarded
        z_max: float
            Maximum redshift. Quasars with redshifts higher than or equal to
            z_max will be discarded
        alpha: float
            Redshift evolution coefficient (see equation 7 of du Mas des
            Bourboux et al. 2020)
        z_ref: float
            Redshift of reference
        cosmo: constants.Cosmo
            The fiducial cosmology
        mode: str
            Mode to read drq file. Defaults to sdss for backward compatibility
        keep_bal: bool
            If False, remove the quasars flagged as having a Broad Absorption
            Line. Ignored if bi_max is not None

    Returns:
        The following variables:
            objs: A list of QSO instances
            z_min: Minimum redshift of the loaded objects.

    Raises:
        AssertionError: if no healpix numbers are found
    """
    objs = {}

    catalog = read_drq(filename, z_min=z_min, z_max=z_max, keep_bal=keep_bal, mode=mode)

    phi = catalog['RA']
    theta = np.pi / 2. - catalog['DEC']
    healpixs = healpy.ang2pix(nside, theta, phi)
    if healpixs.size == 0:
        raise AssertionError()
    userprint("Reading objects ")

    unique_healpix = np.unique(healpixs)

    if mode == 'desi_mocks':
        nightcol='TARGETID'
    elif 'desi' in mode:
        if 'LAST_NIGHT' in catalog.colnames:
            nightcol='LAST_NIGHT'
        elif 'NIGHT' in catalog.colnames:
            nightcol='NIGHT'
        elif 'SURVEY' in catalog.colnames:
            nightcol='TARGETID'
        else:
            raise Exception("The catalog does not have a NIGHT or LAST_NIGHT entry")

    for index, healpix in enumerate(unique_healpix):
        userprint("{} of {}".format(index, len(unique_healpix)))
        w = healpixs == healpix
        if 'TARGETID' in catalog.colnames:
            if 'FIBER' in catalog.colnames:
                fibercol = "FIBER"
            else:
                fibercol = "TARGETID"
            objs[healpix] = [
                QSO(entry['TARGETID'], entry['RA'], entry['DEC'], entry['Z'],
                entry['TARGETID'], entry[nightcol], entry[fibercol])
                for entry in catalog[w]
            ]
        else:
            objs[healpix] = [
                QSO(entry['THING_ID'], entry['RA'], entry['DEC'], entry['Z'],
                    entry['PLATE'], entry['MJD'], entry['FIBERID'])
                for entry in catalog[w]
            ]

        for obj in objs[healpix]:
            obj.weights = ((1. + obj.z_qso) / (1. + z_ref))**(alpha - 1.)
            if not cosmo is None:
                obj.r_comov = cosmo.get_r_comov(obj.z_qso)
                obj.dist_m = cosmo.get_dist_m(obj.z_qso)

    return objs, catalog['Z'].min()
