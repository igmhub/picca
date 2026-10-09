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
from configparser import ConfigParser
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

from . import constants
from .utils import userprint
from .data import Delta, QSO
from .pk1d.prep_pk1d import exp_diff, spectral_resolution
from .pk1d.prep_pk1d import spectral_resolution_desi

# Blinding strategy of the current DESI data release
CURRENT_DESI_BLINDING = "desi_dr3"


def find_order(in_dir, delta_attributes):
    """Finds the order of the polynomial used for the continuum fitting from the delta_attributes file

    Args:
        in_dir: str
            Directory to spectra files. If mode is "spec-mock-1D", then it is
            the filename of the fits file contianing the mock spectra
        delta_attributes: str or None - default: None
            Filename for the delta attributes file. This will be used to read the
            order of the polynomial used for the continuum fitting, which is needed
            for the projection of the delta field. If None, the code will look for it 
            at the standard position. 

    Returns:
        order: int or None
            Order of the log10(lambda) polynomial for the continuum fit. 
            None will result in an error if the deltas are projected or if the distortion 
            matrix is computed
    """
    if delta_attributes is None:
        expected_fnames = [
            in_dir + "/../Log/delta_attributes.fits.gz",
            in_dir + "/attributes.fits"
        ]
        delta_attributes = next(
            (x for x in expected_fnames if os.path.exists(x)),
            expected_fnames[0]
        )
        userprint(f"WARNING: delta_attributes file not given, setting to {delta_attributes}")
    userprint(f"Reading delta attributes from {delta_attributes}")
    try:
        with fitsio.FITS(delta_attributes) as hdul:
            order = hdul["FIT_METADATA"].read_header()['FITORDER']
            userprint(f"Setting order={order} for the polynomial used for the continuum fitting")
    # this exception clause deals with deprecated delta_attributes files that do not have the FITORDER keyword
    # in the FIT_METADATA header. It attemps to find it elsewhere
    # This should be removed after a while, and simply crash
    except KeyError as e:
        userprint(f"WARNING: KeyError encountered: {str(e)}")
        userprint(F"WARNING: Checking for FITORDER in the STACK_DELTAS extension")
        userprint(f"WARNING: This is deprecated and will lead to an error in the future, please update your delta_attributes file")
        try:
            # first we try to find it in the STACK_DELTAS header, which is where it used to be in older versions of picca
            with fitsio.FITS(delta_attributes) as hdul:
                order = hdul["STACK_DELTAS"].read_header()['FITORDER']
                userprint("WARNING: Found FITORDER in STACK_DELTAS header, continuing the analysis")
                userprint(f"Setting order={order} for the polynomial used for the continuum fitting")
        # otherwise we try to find it in the delta config file
        except KeyError as e:
            userprint(f"WARNING: KeyError encountered: {str(e)}")
            userprint("WARNING: Attempting to find FITORDER from the delta config file")
            config_file = in_dir + "/../.config.ini"
            config = ConfigParser()
            config.read(config_file)
            if "expected flux" in config and "order" in config["expected flux"]:
                order = config["expected flux"].getint("order")
                userprint("WARNING: Found `order` in delta config file, continuing the analysis")
                userprint(f"Setting order={order} for the polynomial used for the continuum fitting")
            else:
                order = None
                userprint("WARNING: `order` not found in delta config file")
                userprint(
                    "WARNING: Setting order=None, this will lead to an error if the deltas are projected" \
                    "or if the distortion matrix is computed")
                userprint(f"Setting order={order} for the polynomial used for the continuum fitting")
    # this exception clause deals with the case where the delta_attributes file is not found at all, 
    # which can happen if the user used non-standard placing of the logs. It attempts to find the order 
    # in the delta config file, but this is deprecated and should be removed after a while
    # This should be removed after a while, and simply crash
    except OSError as e:
        userprint(f"WARNING: OSError encountered: {str(e)}")
        userprint("WARNING: Attempting to find FITORDER from the delta config file")
        userprint(f"WARNING: This is deprecated and will lead to an error in the future, please pass a delta_attributes file")
        config_file = in_dir + "/../.config.ini"
        config = ConfigParser()
        config.read(config_file)
        if "expected flux" in config and "order" in config["expected flux"]:
            order = config["expected flux"].getint("order")
            userprint("WARNING: Found `order` in delta config file, continuing the analysis")
            userprint(f"Setting order={order} for the polynomial used for the continuum fitting")
        else:
            order = None
            userprint("WARNING: `order` not found in delta config file")
            userprint(
                "WARNING: Setting order=None, this will lead to an error if the deltas are projected" \
                "or if the distortion matrix is computed")
            userprint(f"Setting order={order} for the polynomial used for the continuum fitting")

    return order
            

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


def _find_first_delta_file(in_dir):
    """Return the first delta FITS file matching a directory or pattern.

    Parameters
    ----------
    in_dir : str
        Directory containing delta FITS files, or a FITS filename or pattern.
        Environment variables are expanded before matching.

    Returns
    -------
    filename : str
        First matching file, in ``glob`` order.

    Raises
    ------
    IndexError
        If no matching delta FITS file is found.
    """
    files = []
    in_dir = os.path.expandvars(in_dir)
    if len(in_dir) > 8 and in_dir[-8:] == '.fits.gz':
        files += glob.glob(in_dir)
    elif len(in_dir) > 5 and in_dir[-5:] == '.fits':
        files += glob.glob(in_dir)
    else:
        files += (glob.glob(in_dir + '/delta-*.fits')
                  + glob.glob(in_dir + '/delta-*.fits.gz'))
    return files[0]


def _check_lya_region_absorber(filename, lambda_abs):
    """Require LYA redshifts if and only if deltas contain Lya-forest pixels.

    Deltas whose rest-frame window extends blueward of the LYA line contain
    Lya-forest pixels. Assigning them the redshifts of another absorber
    could bypass the Lya blinding, so only LYA (or no absorber) is accepted.
    Conversely, LYA is rejected for windows entirely redward of the LYA line
    (e.g. the SiIV or CIV regions), which contain no Lya absorption; these
    accept any other absorber.

    Parameters
    ----------
    filename : str
        Delta FITS file in ImageHDU or BinTable format.
    lambda_abs : str or None
        Absorber identifier defining the pixel redshifts of these deltas.
        None skips the check without reading the file.

    Raises
    ------
    ValueError
        If ``lambda_abs`` is not LYA and the first forest has pixels blueward
        of the LYA line in the quasar rest frame, or if ``lambda_abs`` is LYA
        and all its pixels lie redward of the LYA line.

    Notes
    -----
    Delta extraction applies one rest-frame window to all forests of a run,
    so only the pixels and redshift of the first forest are read.
    """
    if lambda_abs is None:
        return

    with fitsio.FITS(filename) as hdul:
        if "LAMBDA" in hdul:  # ImageHDU: common grid, forest has weight > 0
            lambda_obs = hdul["LAMBDA"].read().astype(float)
            weights = hdul["WEIGHT"][0:1, :][0]
            lambda_obs = lambda_obs[weights > 0]
            z_qso = hdul["METADATA"]["Z"][0:1][0]
        else:  # BinTable: only forest pixels are stored
            if "LOGLAM" in hdul[1].get_colnames():
                lambda_obs = 10**hdul[1]["LOGLAM"][:].astype(float)
            else:
                lambda_obs = hdul[1]["LAMBDA"][:].astype(float)
            z_qso = hdul[1].read_header()["Z"]

    lambda_rest_min = lambda_obs.min() / (1 + z_qso)
    is_lya_region = lambda_rest_min < constants.ABSORBER_IGM["LYA"]
    if is_lya_region and lambda_abs != "LYA":
        raise ValueError(
            f"Deltas in {filename} extend blueward of the LYA line in the "
            f"quasar rest frame (min {lambda_rest_min:.1f} A), but their "
            f"absorber is {lambda_abs}. Lya-region deltas require LYA.")
    if not is_lya_region and lambda_abs == "LYA":
        raise ValueError(
            f"Deltas in {filename} lie redward of the LYA line in the quasar "
            f"rest frame (min {lambda_rest_min:.1f} A), but their absorber is "
            "LYA. Use the absorber of this region instead.")


def read_blinding(in_dir, lambda_abs=None, lambda_abs2=None, in_dir2=None):
    """Read the delta blinding strategy and select it for the absorbers.

    Parameters
    ----------
    in_dir : str
        Directory containing delta FITS files, or a FITS filename or pattern.
        Environment variables are expanded before selecting the first file.
    lambda_abs : str or None, optional
        Primary absorber identifier. The default, None, supplies no absorber.
    lambda_abs2 : str or None, optional
        Second absorber identifier for a two-forest correlation. The default,
        None, uses only the primary absorber when it is supplied.
    in_dir2 : str or None, optional
        Second delta directory, file or pattern, as passed by the user. Only
        used to check that its absorber, ``lambda_abs2``, or ``lambda_abs``
        if ``lambda_abs2`` is None, matches its rest-frame region (see
        ``_check_lya_region_absorber``). The default, None, skips this check,
        as the second field is then read from ``in_dir``.

    Returns
    -------
    blinding : str
        Stored strategy when no absorbers are supplied. With absorber
        identifiers, return ``none`` if none is LYA, LYB, or a CIV variant,
        or ``desi_dr3_civ`` for DR3 correlations involving only CIV absorbers.
        Other absorber combinations retain the stored strategy.

    Raises
    ------
    IndexError
        If no matching delta FITS file is found.
    KeyError
        If an image-format delta file has no BLINDING keyword.
    ValueError
        If the stored strategy is ``CURRENT_DESI_BLINDING`` and deltas in
        ``in_dir`` (or ``in_dir2``) extend blueward of the LYA line in the
        quasar rest frame but their absorber is not LYA, or lie entirely
        redward of it but their absorber is LYA.

    Notes
    -----
    This selection does not modify delta headers or numerical arrays. Delta
    readers continue to use the original strategy stored in each input file.
    The strategy is read from ``in_dir`` only.
    """
    filename = _find_first_delta_file(in_dir)
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
    hdul.close()

    # For the current DESI release, Lya-region deltas must be assigned LYA
    # redshifts in both fields, and redward regions any other absorber
    if blinding == CURRENT_DESI_BLINDING:
        _check_lya_region_absorber(filename, lambda_abs)
        if in_dir2 is not None:
            lambda_abs_field2 = (lambda_abs if lambda_abs2 is None
                                 else lambda_abs2)
            _check_lya_region_absorber(_find_first_delta_file(in_dir2),
                                       lambda_abs_field2)

    absorbers = tuple(absorber for absorber in (lambda_abs, lambda_abs2)
                      if absorber is not None)
    if not absorbers:
        return blinding

    civ_absorbers = ("CIV(eff)", "CIV(1548)", "CIV(1551)")
    standard_absorbers = ("LYA", "LYB") + civ_absorbers
    if not any(absorber in standard_absorbers for absorber in absorbers):
        return "none"
    if blinding == CURRENT_DESI_BLINDING and all(
            absorber in civ_absorbers for absorber in absorbers):
        return CURRENT_DESI_BLINDING + "_civ"

    return blinding


def read_delta_file(filename, z_min_qso=0, z_max_qso=10, rebin_factor=None, order=None):
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
        order: 0, 1 or None - default: None
            Order of the log10(lambda) polynomial for the continuum fit
            None will result in the code crashing if the deltas are projected
            or if they are used to compute the distortion matrix
    Returns:
        deltas:
            A dictionary with the data. Keys are the healpix numbers of each
                spectrum. Values are lists of delta instances.
    """

    hdul = fitsio.FITS(filename)
    # If there is an extension called lambda format is image
    if 'LAMBDA' in hdul:
        deltas = Delta.from_image(hdul, z_min_qso=z_min_qso, z_max_qso=z_max_qso, order=order)
    else:
        deltas = [Delta.from_fitsio(hdu, order=order) 
                  for hdu in hdul[1:] if z_min_qso<hdu.read_header()['Z']<z_max_qso]

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
                z_max_qso=10,
                delta_attributes=None):
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
            Specifies thet maximum redshift for QSOs
        delta_attributes: str or None - default: None
            Filename for the delta attributes file. This will be used to read the
            order of the polynomial used for the continuum fitting, which is needed
            for the projection of the delta field. If None, the code will look for it 
            at the standard position. 

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
        files += sorted(glob.glob(in_dir + '/delta-*.fits')
                        + glob.glob(in_dir + '/delta-*.fits.gz'))
    files = sorted(files)

    if rebin_factor is not None:
        userprint(f"Rebinning deltas by a factor of {rebin_factor}\n")

    order = find_order(in_dir, delta_attributes)

    arguments = [(f, z_min_qso, z_max_qso, rebin_factor, order) for f in files]
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
