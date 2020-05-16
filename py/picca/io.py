"""This module defines a set of functions to manage reading of data.

This module provides a class (Metadata) and several functions:
    - read_dlas
    - read_absorbers
    - read_drq
    - read_dust_map
    - read_data
    - read_from_spec
    - read_from_mock_1d
    - read_from_pix
    - read_from_spcframe
    - read_from_spplate
    - read_from_desi
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

from picca.utils import userprint
from picca.data import Forest, Delta, QSO
from picca.prep_pk1d import exp_diff, spectral_resolution
from picca.prep_pk1d import spectral_resolution_desi

## use a metadata class to simplify things
class Metadata(object):
    """Class defined to organize the storage of metadata.

    Attributes:
        thingid: integer or None
            Thingid of the observation.
        ra: float or None
            Right-ascension of the quasar (in radians).
        dec: float or None
            Declination of the quasar (in radians).
        z_qso: float or None
            Redshift of the quasar.
        plate: integer or None
            Plate number of the observation.
        mjd: integer or None
            Modified Julian Date of the observation.
        fiberid: integer or None
            Fiberid of the observation.
        order: 0 or 1 or None
            Order of the log10(lambda) polynomial for the continuum fit

    Methods:
        __init__
    """
    def __init__(self):
        """Initialize instance."""
        self.thingid = None
        self.ra = None
        self.dec = None
        self.z_qso = None
        self.plate = None
        self.mjd = None
        self.fiberid = None
        self.order = None

def read_dlas(filename):
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
    columns_list = ['THING_ID', 'Z', 'NHI']
    hdul = fitsio.FITS(filename)
    cat = {col: hdul['DLACAT'][col][:] for col in columns_list}
    hdul.close()

    # sort the items in the dictionary according to THING_ID and redshift
    w = np.argsort(cat['Z'])
    for key in cat.keys():
        cat[key] = cat[key][w]
    w = np.argsort(cat['THING_ID'])
    for key in cat.keys():
        cat[key] = cat[key][w]

    # group DLAs on the same line of sight together
    dlas = {}
    for thingid in np.unique(cat['THING_ID']):
        w = (thingid == cat['THING_ID'])
        dlas[thingid] = list(zip(cat['Z'][w], cat['NHI'][w]))
    num_dlas = np.sum([len(dla) for dla in dlas.values()])

    userprint('\n')
    userprint(' In catalog: {} DLAs'.format(num_dlas))
    userprint(' In catalog: {} forests have a DLA'.format(len(dlas)))
    userprint('\n')

    return dlas

def read_absorbers(filename):
    """Reads the absorbers catalog from an ascii file.

    Args:
        filename: str
            File containing the absorbers

    Returns:
        A dictionary with the absorbers's information. Keys are the THING_ID
        associated with the DLA. Values are a tuple with its redshift and
        column density.
    """
    file = open(filename)
    absorbers = {}
    num_absorbers = 0
    col_names = None
    for line in file.readlines():
        cols = line.split()
        if len(cols) == 0:
            continue
        if cols[0][0] == "#":
            continue
        if cols[0] == "ThingID":
            col_names = cols
            continue
        if cols[0][0] == "-":
            continue
        thingid = int(cols[col_names.index("ThingID")])
        if thingid not in absorbers:
            absorbers[thingid] = []
        lambda_abs = float(cols[col_names.index("lambda")])
        absorbers[thingid].append(lambda_abs)
        num_absorbers += 1
    file.close()

    userprint("")
    userprint(" In catalog: {} absorbers".format(num_absorbers))
    userprint(" In catalog: {} forests have absorbers".format(len(absorbers)))
    userprint("")

    return absorbers

def read_drq(drq_filename, z_min, z_max, keep_bal, bi_max=None):
    """Reads the quasars in the DRQ quasar catalog.

    Args:
        drq_filename: str
            Filename of the DRQ catalogue
        z_min: float
            Minimum redshift. Quasars with redshifts lower than z_min will be
            discarded
        z_max: float
            Maximum redshift. Quasars with redshifts higher than or equal to
            z_max will be discarded
        keep_bal: bool
            If False, remove the quasars flagged as having a Broad Absorption
            Line. Ignored if bi_max is not None
        bi_max: float or None - default: None
            Maximum value allowed for the Balnicity Index to keep the quasar

    Returns:
        The arrays containing
            ra: the right ascension of the quasars (in radians)
            dec: the declination of the quasars (in radians)
            z_qso: the redshift of the quasars
            thingid: the thingid of the observations
            plate: the plates of the observations
            mjd: the Modified Julian Date of the observation
            fiberid: the fiberid of the observations
    """
    hdul = fitsio.FITS(drq_filename)

    ## Redshift
    try:
        z_qso = hdul[1]['Z'][:]
    except ValueError:
        userprint("Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)")
        z_qso = hdul[1]['Z_VI'][:]

    ## Info of the primary observation
    thingid = hdul[1]['THING_ID'][:]
    ra = hdul[1]['RA'][:].astype('float64')
    dec = hdul[1]['DEC'][:].astype('float64')
    plate = hdul[1]['PLATE'][:]
    mjd = hdul[1]['MJD'][:]
    fiberid = hdul[1]['FIBERID'][:]

    ## Sanity
    userprint('')
    w = np.ones(ra.size, dtype=bool)
    userprint(" start                 : nb object in cat = {}".format(w.sum()))
    w &= thingid > 0
    userprint(" and thingid > 0       : nb object in cat = {}".format(w.sum()))
    w &= ra != dec
    userprint(" and ra != dec         : nb object in cat = {}".format(w.sum()))
    w &= ra != 0.
    userprint(" and ra != 0.          : nb object in cat = {}".format(w.sum()))
    w &= dec != 0.
    userprint(" and dec != 0.         : nb object in cat = {}".format(w.sum()))
    w &= z_qso > 0.
    userprint(" and z > 0.            : nb object in cat = {}".format(w.sum()))

    ## Redshift range
    if not z_min is None:
        w &= z_qso >= z_min
        userprint((" and z >= z_min        : nb object in cat"
                   "= {}".format(w.sum())))
    if not z_max is None:
        w &= z_qso < z_max
        userprint((" and z < z_max         : nb object in cat"
                   "= {}".format(w.sum())))

    ## BAL visual
    if not keep_bal and bi_max is None:
        try:
            bal_flag = hdul[1]['BAL_FLAG_VI'][:]
            w &= bal_flag == 0
            userprint((" and BAL_FLAG_VI == 0  : nb object in cat"
                       "= {}".format(ra[w].size)))
        except ValueError:
            userprint("BAL_FLAG_VI not found\n")
    ## BAL CIV
    if bi_max is not None:
        try:
            bi = hdul[1]['BI_CIV'][:]
            w &= bi <= bi_max
            userprint((" and BI_CIV <= bi_max  : nb object in cat"
                       "= {}".format(ra[w].size)))
        except ValueError:
            userprint("--bi-max set but no BI_CIV field in HDU")
            sys.exit(1)
    userprint("")

    ra = ra[w]*np.pi/180.
    dec = dec[w]*np.pi/180.
    z_qso = z_qso[w]
    thingid = thingid[w]
    plate = plate[w]
    mjd = mjd[w]
    fiberid = fiberid[w]
    hdul.close()

    return ra, dec, z_qso, thingid, plate, mjd, fiberid

def read_dust_map(drq_filename, extinction_conversion_r=3.793):
    """Reads the dust map.

    Args:
        drq_filename: str
            Filename of the DRQ catalogue
        extinction_conversion_r: float
            Conversion from E(B-V) to total extinction for band r.
            Note that the EXTINCTION values given in DRQ are in fact E(B-V)

    Returns:
        A dictionary with the extinction map. Keys are the THING_ID
        associated with the observation. Values are the extinction for that
        line of sight.
    """
    hdul = fitsio.FITS(drq_filename)
    thingid = hdul[1]['THING_ID'][:]
    ext = hdul[1]['EXTINCTION'][:][:, 1]/extinction_conversion_r
    hdul.close()

    return dict(zip(thingid, ext))

def read_data(in_dir, drq_filename, mode, z_min=2.1, z_max=3.5,
              max_num_spec=None, log_file=None, keep_bal=False, bi_max=None,
              order=1, best_obs=False, single_exp=False, pk1d=None):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        in_dir: str
            Directory to spectra files. If mode is "spec-mock-1D", then it is
            the filename of the fits file contianing the mock spectra
        drq_filename: str
            Filename of the DRQ catalogue
        mode: str
            One of 'pix', 'spec', 'spcframe', 'spplate', 'corrected-spec',
            'spec-mock-1d' or 'desi'. Open mode of the spectra files
        z_min: float - default: 2.1
            Minimum redshift. Quasars with redshifts lower than z_min will be
            discarded
        z_max: float - default: 3.5
            Maximum redshift. Quasars with redshifts higher than or equal to
            z_max will be discarded
        max_num_spec: int or None - default: None
            Maximum number of spectra to read
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log
        keep_bal: bool - default: False
            If False, remove the quasars flagged as having a Broad Absorption
            Line. Ignored if bi_max is not None
        bi_max: float or None - default: None
            Maximum value allowed for the Balnicity Index to keep the quasar
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        best_obs: bool - default: False
            If set, reads only the best observation for objects with repeated
            observations
        single_exp: bool - default: False
            If set, reads only one observation for objects with repeated
            observations (chosen randomly)
        pk1d: str or None - default: None
            Format for Pk 1D: Pk1D

    Returns:
        The following variables:
            data: A dictionary with the data. Keys are the healpix numbers of
                each spectrum. Values are lists of Forest instances.
            num_data: Number of spectra in data.
            nside: The healpix nside parameter.
            "RING": The healpix pixel ordering used.
    """
    userprint("mode: " + mode)
    # read quasar characteristics from DRQ catalogue
    ra, dec, z_qso, thingid, plate, mjd, fiberid = read_drq(drq_filename,
                                                            z_min, z_max,
                                                            keep_bal,
                                                            bi_max=bi_max)

    # if there is a maximum number of spectra, make sure they are selected
    # in a contiguous regions
    if max_num_spec is not None:
        ## choose them in a small number of pixels
        healpixs = healpy.ang2pix(16, np.pi/2 - dec, ra)
        sorted_healpixs = np.argsort(healpixs)
        ra = ra[sorted_healpixs][:max_num_spec]
        dec = dec[sorted_healpixs][:max_num_spec]
        z_qso = z_qso[sorted_healpixs][:max_num_spec]
        thingid = thingid[sorted_healpixs][:max_num_spec]
        plate = plate[sorted_healpixs][:max_num_spec]
        mjd = mjd[sorted_healpixs][:max_num_spec]
        fiberid = fiberid[sorted_healpixs][:max_num_spec]

    data = {}
    num_data = 0

    # read data taking the mode into account
    if mode == "desi":
        nside = 8
        userprint("Found {} qsos".format(len(z_qso)))
        data, num_data = read_from_desi(nside, in_dir, thingid, ra, dec, z_qso,
                                        plate, mjd, fiberid, order, pk1d=pk1d)

    elif mode in ["spcframe", "spplate", "spec", "corrected-spec"]:
        nside, healpixs = find_nside(ra, dec, log_file)

        if mode == "spcframe":
            pix_data = read_from_spcframe(in_dir, thingid, ra, dec, z_qso,
                                          plate, mjd, fiberid, order,
                                          log_file=log_file,
                                          single_exp=single_exp)
        elif mode == "spplate":
            pix_data = read_from_spplate(in_dir, thingid, ra, dec, z_qso,
                                         plate, mjd, fiberid, order,
                                         log_file=log_file,
                                         best_obs=best_obs)
        else:
            pix_data = read_from_spec(in_dir, thingid, ra, dec, z_qso, plate,
                                      mjd, fiberid, order, mode=mode,
                                      log_file=log_file,
                                      pk1d=pk1d, best_obs=best_obs)
        ra = np.array([d.ra for d in pix_data])
        dec = np.array([d.dec for d in pix_data])
        healpixs = healpy.ang2pix(nside, np.pi / 2 - dec, ra)
        for index, healpix in enumerate(healpixs):
            if healpix not in data:
                data[healpix] = []
            data[healpix].append(pix_data[index])
            num_data += 1

    elif mode in ["pix", "spec-mock-1D"]:
        data = {}
        num_data = 0

        if mode == "pix":
            try:
                filename = in_dir + "/master.fits.gz"
                hdul = fitsio.FITS(filename)
            except IOError:
                try:
                    filename = in_dir + "/master.fits"
                    hdul = fitsio.FITS(filename)
                except IOError:
                    try:
                        filename = in_dir + "/../master.fits"
                        hdul = fitsio.FITS(filename)
                    except IOError:
                        userprint("error reading master")
                        sys.exit(1)
            nside = hdul[1].read_header()['NSIDE']
            hdul.close()
            healpixs = healpy.ang2pix(nside, np.pi/2 - dec, ra)
        else:
            nside, healpixs = find_nside(ra, dec, log_file)

        unique_healpix = np.unique(healpixs)

        for index, healpix in enumerate(unique_healpix):
            w = healpixs == healpix
            ## read all hiz qsos
            if mode == "pix":
                t0 = time.time()
                pix_data = read_from_pix(in_dir, healpix, thingid[w], ra[w],
                                         dec[w], z_qso[w], plate[w], mjd[w],
                                         fiberid[w], order, log_file=log_file)
                read_time = time.time() - t0
            elif mode == "spec-mock-1D":
                t0 = time.time()
                pix_data = read_from_mock_1d(in_dir, thingid[w], ra[w], dec[w],
                                             z_qso[w], plate[w], mjd[w],
                                             fiberid[w], order,
                                             log_file=log_file)
                read_time = time.time() - t0

            if not pix_data is None:
                userprint(("{} read from pix {}, {} {} in {} secs per"
                           "spectrum").format(len(pix_data), healpix, index,
                                              len(unique_healpix),
                                              read_time/(len(pix_data) + 1e-3)))
            if not pix_data is None and len(pix_data) > 0:
                data[healpix] = pix_data
                num_data += len(pix_data)

    else:
        userprint("I don't know mode: {}".format(mode))
        sys.exit(1)

    return data, num_data, nside, "RING"

def find_nside(ra, dec, log_file):
    """Determines nside such that there are 1000 objs per pixel on average.

    Args:
        ra: array of floats
            The right ascension of the quasars (in radians)
        dec: array of floats
            The declination of the quasars (in radians)
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log

    Returns:
        The value of nside and the healpixs for the objects
    """
    ## determine nside such that there are 1000 objs per pixel on average
    userprint("determining nside")
    nside = 256
    healpixs = healpy.ang2pix(nside, np.pi/2 - dec, ra)
    mean_num_obj = len(healpixs)/len(np.unique(healpixs))
    target_mean_num_obj = 500
    nside_min = 8
    while mean_num_obj < target_mean_num_obj and nside >= nside_min:
        nside //= 2
        healpixs = healpy.ang2pix(nside, np.pi/2 - dec, ra)
        mean_num_obj = len(healpixs)/len(np.unique(healpixs))
    userprint("nside = {} -- mean #obj per pixel = {}".format(nside,
                                                              mean_num_obj))
    if log_file is not None:
        log_file.write(("nside = {} -- mean #obj per pixel"
                        " = {}\n").format(nside, mean_num_obj))

    return nside, healpixs

def read_from_spec(in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid, order,
                   mode, log_file=None, pk1d=None, best_obs=None):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        in_dir: str
            Directory to spectra files
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        mode: str
            One of 'spec' or 'corrected-spec'. Open mode of the spectra files
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log
        pk1d: str or None - default: None
            Format for Pk 1D: Pk1D
        best_obs: bool - default: False
            If set, reads only the best observation for objects with repeated
            observations

    Returns:
        List of read spectra for all the healpixs
    """
    # since thingid might change, keep a dictionary with the coordinates info
    drq_dict = {t: (r, d, z) for t, r, d, z in zip(thingid, ra, dec, z_qso)}

    ## if using multiple observations,
    ## then replace thingid, plate, mjd, fiberid
    ## by what's available in spAll
    if not best_obs:
        thingid, plate, mjd, fiberid = read_spall(in_dir, thingid)

    ## to simplify, use a list of all metadata
    all_metadata = []
    ## Used to preserve original order and pass unit tests.
    thingid_list = []
    thingid_set = set()
    for t, p, m, f in zip(thingid, plate, mjd, fiberid):
        if t not in thingid_set:
            thingid_list.append(t)
            thingid_set.add(t)
        r, d, z = drq_dict[t]
        metadata = Metadata()
        metadata.thingid = t
        metadata.ra = r
        metadata.dec = d
        metadata.z_qso = z
        metadata.plate = p
        metadata.mjd = m
        metadata.fiberid = f
        metadata.order = order
        all_metadata.append(metadata)

    pix_data = []
    thingids = {}

    for metadata in all_metadata:
        t = metadata.thingid
        if not t in thingids:
            thingids[t] = []
        thingids[t].append(metadata)

    userprint("reading {} thingids".format(len(thingids)))

    for t in thingid_list:
        deltas = None
        for metadata in thingids[t]:
            filename = in_dir + ("/{}/{}-{}-{}-{:04d}"
                                 ".fits").format(metadata.plate, mode,
                                                 metadata.plate, metadata.mjd,
                                                 metadata.fiberid)
            try:
                hdul = fitsio.FITS(filename)
            except IOError:
                log_file.write("error reading {}\n".format(filename))
                continue
            log_file.write("{} read\n".format(filename))
            log_lambda = hdul[1]["loglam"][:]
            flux = hdul[1]["flux"][:]
            ivar = hdul[1]["ivar"][:]*(hdul[1]["and_mask"][:] == 0)

            if pk1d is not None:
                # compute difference between exposure
                exposures_diff = exp_diff(hdul, log_lambda)
                # compute spectral resolution
                wdisp = hdul[1]["wdisp"][:]
                reso = spectral_resolution(wdisp, True, metadata.fiberid,
                                           log_lambda)
            else:
                exposures_diff = None
                reso = None
            if deltas is None:
                deltas = Forest(log_lambda, flux, ivar, metadata.thingid,
                                metadata.ra, metadata.dec, metadata.z_qso,
                                metadata.plate, metadata.mjd, metadata.fiberid,
                                order, exposures_diff=exposures_diff, reso=reso)
            else:
                deltas += Forest(log_lambda, flux, ivar, metadata.thingid,
                                 metadata.ra, metadata.dec, metadata.z_qso,
                                 metadata.plate, metadata.mjd, metadata.fiberid,
                                 order, exposures_diff=exposures_diff,
                                 reso=reso)
            hdul.close()
        if deltas is not None:
            pix_data.append(deltas)

    return pix_data

def read_from_mock_1d(filename, thingid, ra, dec, z_qso, plate, mjd, fiberid,
                      order, log_file=None):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        filename: str
            Filename of the fits file contianing the mock spectra
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log

    Returns:
        List of read spectra for all the healpixs
    """
    pix_data = []

    try:
        hdul = fitsio.FITS(filename)
    except IOError:
        log_file.write("error reading {}\n".format(filename))

    for t, r, d, z, p, m, f in zip(thingid, ra, dec, z_qso, plate, mjd,
                                   fiberid):
        hdu = hdul["{}".format(t)]
        log_file.write("file: {} hdus {} read  \n".format(filename, hdu))
        lambda_ = hdu["wavelength"][:]
        log_lambda = np.log10(lambda_)
        flux = hdu["flux"][:]
        error = hdu["error"][:]
        ivar = 1.0/error**2

        # compute difference between exposure
        exposures_diff = np.zeros(len(lambda_))
        # compute spectral resolution
        wdisp = hdu["psf"][:]
        reso = spectral_resolution(wdisp)

        # compute the mean expected flux
        mean_flux_transmission = hdu.read_header()["MEANFLUX"]
        cont = hdu["continuum"][:]
        mef = mean_flux_transmission*remove_keys
        pix_data.append(Forest(log_lambda, flux, ivar, t, r, d, z, p, m, f,
                               order, exposures_diff, reso, mef))

    hdul.close()

    return pix_data


def read_from_pix(in_dir, healpix, thingid, ra, dec, z_qso, plate, mjd, fiberid,
                  order, log_file=None):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        in_dir: str
            Directory to spectra files
        healpix: int
            The pixel number of a particular healpix
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log
        pk1d: str or None - default: None
            Format for Pk 1D: Pk1D
        best_obs: bool - default: False
            If set, reads only the best observation for objects with repeated
            observations

    Returns:
        List of read spectra for all the healpixs
    """
    try:
        filename = in_dir + "/pix_{}.fits.gz".format(healpix)
        hdul = fitsio.FITS(filename)
    except IOError:
        try:
            filename = in_dir + "/pix_{}.fits".format(healpix)
            hdul = fitsio.FITS(filename)
        except IOError:
            userprint("error reading {}".format(healpix))
            return None

    ## fill log
    if log_file is not None:
        for t in thingid:
            if t not in hdul[0][:]:
                log_file.write("{} missing from pixel {}\n".format(t, healpix))
                userprint("{} missing from pixel {}".format(t, healpix))

    pix_data = []
    thingid_list = list(hdul[0][:])
    thingid2index = {t: thingid_list.index(t)
                     for t in thingid if t in thingid_list}
    log_lambda = hdul[1][:]
    flux = hdul[2].read()
    ivar = hdul[3].read()
    mask = hdul[4].read()
    for (t, r, d, z, p, m, f) in zip(thingid, ra, dec, z_qso, plate, mjd,
                                     fiberid):
        try:
            index = thingid2index[t]
        except KeyError:
            ## fill log
            if log_file is not None:
                log_file.write("{} missing from pixel {}\n".format(t, healpix))
            userprint("{} missing from pixel {}".format(t, healpix))
            continue
        pix_data.append(Forest(log_lambda, flux[:, index],
                               ivar[:, index]*(mask[:, index] == 0),
                               t, r, d, z, p, m, f, order))

        if log_file is not None:
            log_file.write("{} read\n".format(t))

    hdul.close()

    return pix_data

def read_from_spcframe(in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid,
                       order, log_file=None, single_exp=False):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        in_dir: str
            Directory to spectra files
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log
        single_exp: bool - default: False
            If set, reads only one observation for objects with repeated
            observations (chosen randomly)

    Returns:
        List of read spectra for all the healpixs
    """
    if not single_exp:
        userprint(("ERROR: multiple observations not (yet) compatible with "
                   "spframe option"))
        userprint("ERROR: rerun with the --single-exp option")
        sys.exit(1)

    # store all the metadata in a single variable
    all_metadata = []
    for t, r, d, z, p, m, f in zip(thingid, ra, dec, z_qso, plate, mjd,
                                   fiberid):
        metadata = Metadata()
        metadata.thingid = t
        metadata.ra = r
        metadata.dec = d
        metadata.z_qso = z
        metadata.plate = p
        metadata.mjd = m
        metadata.fiberid = f
        metadata.order = order
        all_metadata.append(metadata)

    # group the metadata with respect to their plate and mjd
    platemjd = {}
    for index in range(len(thingid)):
        if (plate[index], mjd[index]) not in platemjd:
            platemjd[(plate[index], mjd[index])] = []
        platemjd[(plate[index], mjd[index])].append(all_metadata[index])

    pix_data = {}
    userprint("reading {} plates".format(len(platemjd)))

    for key in platemjd:
        p, m = key
        # list all the exposures
        exps = []
        spplate = in_dir + "/{0}/spPlate-{0}-{1}.fits".format(p, m)
        userprint("INFO: reading file {}".format(spplate))
        hdul = fitsio.FITS(spplate)
        header = hdul[0].read_header()
        hdul.close()

        for card_suffix in ["B1", "B2", "R1", "R2"]:
            card = "NEXP_{}".format(card_suffix)
            if card in header:
                num_exp = header["NEXP_{}".format(card_suffix)]
            else:
                continue
            for index_exp in range(1, num_exp + 1):
                card = "EXPID{:02d}".format(index_exp)
                if not card in header:
                    continue
                exps.append(header[card][:11])

        userprint("INFO: found {} exposures in plate {}-{}".format(len(exps),
                                                                   p, m))

        if len(exps) == 0:
            continue

        # select a single exposure randomly
        selected_exps = [exp[3:] for exp in exps]
        selected_exps = np.unique(selected_exps)
        np.random.shuffle(selected_exps)
        selected_exps = selected_exps[0]

        for exp in exps:
            if single_exp:
                # if the exposure is not selected, ignore it
                if not selected_exps in exp:
                    continue
            t0 = time.time()
            # find the spectrograph number
            spectro = int(exp[1])
            assert spectro in [1, 2]

            spcframe = fitsio.FITS(in_dir +
                                   "/{}/spCFrame-{}.fits".format(p, exp))

            flux = spcframe[0].read()
            ivar = spcframe[1].read()*(spcframe[2].read() == 0)
            log_lambda = spcframe[3].read()

            ## now convert all those fluxes into forest objects
            for metadata in platemjd[key]:
                if spectro == 1 and metadata.fiberid > 500:
                    continue
                if spectro == 2 and metadata.fiberid <= 500:
                    continue
                index = (metadata.fiberid - 1) % 500
                t = metadata.thingid
                r = metadata.ra
                d = metadata.dec
                z = metadata.z_qso
                f = metadata.fiberid
                order = metadata.order
                if t in pix_data:
                    pix_data[t] += Forest(log_lambda[index], flux[index],
                                          ivar[index], t, r, d, z, p, m, f,
                                          order)
                else:
                    pix_data[t] = Forest(log_lambda[index], flux[index],
                                         ivar[index], t, r, d, z, p, m, f,
                                         order)
                if log_file is not None:
                    log_file.write(("{} read from exp {} and"
                                    " mjd {}\n").format(t, exp, m))
            num_read = len(platemjd[key])

            userprint(("INFO: read {} from {} in {} per spec. Progress: "
                       "{} of {} \n").format(num_read, exp,
                                             (time.time()-t0)/(num_read+1e-3),
                                             len(pix_data), len(thingid)))
            spcframe.close()

    data = list(pix_data.values())

    return data

def read_from_spplate(in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid,
                      order, log_file=None, best_obs=False):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        in_dir: str
            Directory to spectra files
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        log_file: _io.TextIOWrapper or None - default: None
            Opened file to print log
        best_obs: bool - default: False
            If set, reads only the best observation for objects with repeated
            observations

    Returns:
        List of read spectra for all the healpixs
    """
    drq_dict = {t: (r, d, z) for t, r, d, z in zip(thingid, ra, dec, z_qso)}

    ## if using multiple observations,
    ## then replace thingid, plate, mjd, fiberid
    ## by what's available in spAll
    if not best_obs:
        thingid, plate, mjd, fiberid = read_spall(in_dir, thingid)

    ## to simplify, use a list of all metadata
    all_metadata = []
    for t, p, m, f in zip(thingid, plate, mjd, fiberid):
        r, d, z = drq_dict[t]
        metadata = Metadata()
        metadata.thingid = t
        metadata.ra = r
        metadata.dec = d
        metadata.z_qso = z
        metadata.plate = p
        metadata.mjd = m
        metadata.fiberid = f
        metadata.order = order
        all_metadata.append(metadata)

    pix_data = {}
    platemjd = {}
    for p, m, metadata in zip(plate, mjd, all_metadata):
        if (p, m) not in platemjd:
            platemjd[(p, m)] = []
        platemjd[(p, m)].append(metadata)

    userprint("reading {} plates".format(len(platemjd)))

    for key in platemjd:
        p, m = key
        spplate = in_dir + "/{0}/spPlate-{0}-{1}.fits".format(str(p).zfill(4),
                                                              m)

        try:
            hdul = fitsio.FITS(spplate)
            header = hdul[0].read_header()
        except IOError:
            log_file.write("error reading {}\n".format(spplate))
            continue

        coeff0 = header["COEFF0"]
        coeff1 = header["COEFF1"]

        flux = hdul[0].read()
        ivar = hdul[1].read()*(hdul[2].read() == 0)
        log_lambda = coeff0 + coeff1*np.arange(flux.shape[1])

        ## now convert all those fluxes into forest objects
        for metadata in platemjd[(p, m)]:
            t = metadata.thingid
            r = metadata.ra
            d = metadata.dec
            z = metadata.z_qso
            f = metadata.fiberid
            order = metadata.order

            i = metadata.fiberid-1
            if t in pix_data:
                pix_data[t] += Forest(log_lambda, flux[i], ivar[i], t, r, d, z,
                                      p, m, f, order)
            else:
                pix_data[t] = Forest(log_lambda, flux[i], ivar[i], t, r, d, z,
                                     p, m, f, order)
            if log_file is not None:
                log_file.write("{} read from file {} and mjd {}\n".format(t, spplate, m))
        num_read = len(platemjd[(p, m)])
        userprint(("INFO: read {} from {} in {} per spec. Progress: {} "
                   "of {} \n").format(num_read,
                                      os.path.basename(spplate),
                                      (time.time() - 0)/(num_read + 1e-3),
                                      len(pix_data),
                                      len(thingid)))
        hdul.close()

    data = list(pix_data.values())
    return data

def read_from_desi(nside, in_dir, thingid, ra, dec, z_qso, plate, mjd, fiberid,
                   order, pk1d=None):
    """Reads the spectra and formats its data as Forest instances.

    Args:
        nside: int
            The healpix nside parameter
        in_dir: str
            Directory to spectra files
        thingid: array of int
            Thingid of the observations
        ra: array of float
            Right-ascension of the quasars (in radians)
        dec: array of float
            Declination of the quasars (in radians)
        z_qso: array of float
            Redshift of the quasars
        plate: array of integer
            Plate number of the observations
        mjd: array of integer
            Modified Julian Date of the observations
        fiberid: array of integer
            Fiberid of the observations
        order: 0 or 1 - default: 1
            Order of the log10(lambda) polynomial for the continuum fit
        pk1d: str or None - default: None
            Format for Pk 1D: Pk1D

    Returns:
        List of read spectra for all the healpixs
    """
    in_nside = int(in_dir.split('spectra-')[-1].replace('/', ''))
    nest = True
    data = {}
    num_data = 0

    z_table = dict(zip(thingid, z_qso))
    in_healpixs = healpy.ang2pix(in_nside, np.pi/2. - dec, ra, nest=nest)
    unique_in_healpixs = np.unique(in_healpixs)

    for index, healpix in enumerate(unique_in_healpixs):
        filename = (in_dir + "/" + str(int(healpix/100)) + "/" + str(healpix) +
                    "/spectra-" + str(in_nside) + "-" + str(healpix) + ".fits")

        userprint(("\rread {} of {}. "
                   "num_data: {}").format(index, len(unique_in_healpixs),
                                          num_data))
        try:
            hdul = fitsio.FITS(filename)
        except IOError:
            userprint("Error reading pix {}\n".format(healpix))
            continue

        ## get the quasars
        thingid_qsos = thingid[(in_healpixs == healpix)]
        plate_qsos = plate[(in_healpixs == healpix)]
        mjd_qsos = mjd[(in_healpixs == healpix)]
        fiberid_qsos = fiberid[(in_healpixs == healpix)]
        if 'TARGET_RA' in hdul["FIBERMAP"].get_colnames():
            ra = hdul["FIBERMAP"]["TARGET_RA"][:]*np.pi/180.
            dec = hdul["FIBERMAP"]["TARGET_DEC"][:]*np.pi/180.
        elif 'RA_TARGET' in hdul["FIBERMAP"].get_colnames():
            ## TODO: These lines are for backward compatibility
            ## Should be removed at some point
            ra = hdul["FIBERMAP"]["RA_TARGET"][:]*np.pi/180.
            dec = hdul["FIBERMAP"]["DEC_TARGET"][:]*np.pi/180.
        healpixs = healpy.ang2pix(nside, np.pi / 2 - dec, ra)
        #exp = h["FIBERMAP"]["EXPID"][:]
        #night = h["FIBERMAP"]["NIGHT"][:]
        #fib = h["FIBERMAP"]["FIBER"][:]
        in_thingids = hdul["FIBERMAP"]["TARGETID"][:]

        spec_data = {}
        for spectrogrpah in ["B", "R", "Z"]:
            spec = {}
            try:
                lambda_ = hdul["{}_WAVELENGTH".format(spectrogrpah)].read()
                spec["log_lambda"] = np.log10(lambda_)
                spec["FL"] = hdul["{}_FLUX".format(spectrogrpah)].read()
                spec["IV"] = (hdul["{}_IVAR".format(spectrogrpah)].read()*
                              (hdul["{}_MASK".format(spectrogrpah)].read() == 0))
                w = np.isnan(spec["FL"]) | np.isnan(spec["IV"])
                for key in ["FL", "IV"]:
                    spec[key][w] = 0.
                spec["RESO"] = hdul["{}_RESOLUTION".format(spectrogrpah)].read()
                spec_data[spectrogrpah] = spec
            except OSError:
                pass
        hdul.close()

        for t, p, m, f in zip(thingid_qsos, plate_qsos, mjd_qsos, fiberid_qsos):
            w_t = in_thingids == t
            if w_t.sum() == 0:
                userprint("\nError reading thingid {}\n".format(t))
                continue

            forest = None
            for spec in spec_data.values():
                ivar = spec['IV'][w_t]
                flux = (ivar*spec['FL'][w_t]).sum(axis=0)
                ivar = ivar.sum(axis=0)
                w = ivar > 0.
                flux[w] /= ivar[w]
                if not pk1d is None:
                    reso_sum = spec['RESO'][w_t].sum(axis=0)
                    reso_in_km_per_s = spectral_resolution_desi(reso_sum,
                                                                spec['log_lambda'])
                    exposures_diff = np.zeros(spec['log_lambda'].shape)
                else:
                    reso_in_km_per_s = None
                    exposures_diff = None

                if forest is None:
                    forest = copy.deepcopy(Forest(spec['log_lambda'], flux,
                                                  ivar, t, ra[w_t][0],
                                                  dec[w_t][0], z_table[t], p, m,
                                                  f, order, exposures_diff,
                                                  reso_in_km_per_s))
                else:
                    forest += Forest(spec['log_lambda'], flux, ivar, t,
                                     ra[w_t][0], dec[w_t][0], z_table[t], p, m,
                                     f, order, exposures_diff, reso_in_km_per_s)

            pix = healpixs[w_t][0]
            if pix not in data:
                data[pix] = []
            data[pix].append(forest)
            num_data += 1

    userprint("found {} quasars in input files\n".format(num_data))

    return data, num_data


def read_deltas(in_dir, nside, lambda_abs, alpha, z_ref, cosmo,
                max_num_spec=None, no_project=False, from_image=None):
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
            If True, project the deltas (see equation 5 of du Mas des Bourboux
            et al. 2020)
        from_image: list or None - default: None
            If not None, read the deltas from image files. The list of
            filenname for the image files should be paassed in from_image

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
    if from_image is None or len(from_image) == 0:
        if len(in_dir) > 8 and in_dir[-8:] == '.fits.gz':
            files += glob.glob(in_dir)
        elif len(in_dir) > 5 and in_dir[-5:] == '.fits':
            files += glob.glob(in_dir)
        else:
            files += glob.glob(in_dir+'/*.fits') + glob.glob(in_dir+'/*.fits.gz')
    else:
        for arg in from_image:
            if len(arg) > 8 and arg[-8:] == '.fits.gz':
                files += glob.glob(arg)
            elif len(arg) > 5 and arg[-5:] == '.fits':
                files += glob.glob(arg)
            else:
                files += glob.glob(arg+'/*.fits') + glob.glob(arg+'/*.fits.gz')
    files = sorted(files)

    deltas = []
    num_data = 0
    for index, filename in enumerate(files):
        userprint("\rread {} of {} {}".format(index, len(files), num_data))
        if from_image is None:
            hdul = fitsio.FITS(filename)
            deltas += [Delta.from_fitsio(hdu) for hdu in hdul[1:]]
            hdul.close()
        else:
            deltas += Delta.from_image(filename)

        num_data = len(deltas)
        if max_num_spec is not None:
            if num_data > max_num_spec:
                break

    # truncate the deltas if we load too many lines of sight
    if max_num_spec is not None:
        deltas = deltas[:max_num_spec]
        num_data = len(deltas)

    userprint("\n")

    # compute healpix numbers
    phi = [delta.ra for delta in deltas]
    theta = [np.pi/2. - delta.dec for delta in deltas]
    healpixs = healpy.ang2pix(nside, theta, phi)
    if healpixs.size == 0:
        raise AssertionError('ERROR: No data in {}'.format(in_dir))

    data = {}
    z_min = 10**deltas[0].log_lambda[0]/lambda_abs - 1.
    z_max = 0.
    for delta, healpix in zip(deltas, healpixs):
        z = 10**delta.log_lambda/lambda_abs - 1.
        z_min = min(z_min, z.min())
        z_max = max(z_max, z.max())
        delta.z = z
        if not cosmo is None:
            delta.r_comov = cosmo.get_r_comov(z)
            delta.dist_m = cosmo.get_dist_m(z)
        delta.weights *= ((1 + z)/(1 + z_ref))**(alpha - 1)

        if not no_project:
            delta.project()

        if not healpix in data:
            data[healpix] = []
        data[healpix].append(delta)

    return data, num_data, z_min, z_max


def read_objects(filename, nside, z_min, z_max, alpha, z_ref, cosmo,
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
    ra, dec, z_qso, thingid, plate, mjd, fiberid = read_drq(filename, z_min,
                                                            z_max,
                                                            keep_bal=keep_bal)
    phi = ra
    theta = np.pi/2. - dec
    healpixs = healpy.ang2pix(nside, theta, phi)
    if healpixs.size == 0:
        raise AssertionError()
    userprint("reading qsos")

    unique_healpix = np.unique(healpixs)
    for index, healpix in enumerate(unique_healpix):
        userprint("\r{} of {}".format(index, len(unique_healpix)))
        w = healpixs == healpix
        objs[healpix] = [QSO(t, r, d, z, p, m, f)
                         for t, r, d, z, p, m, f in zip(thingid[w], ra[w],
                                                        dec[w], z_qso[w],
                                                        plate[w], mjd[w],
                                                        fiberid[w])]
        for obj in objs[healpix]:
            obj.weights = ((1. + obj.z_qso)/(1. + z_ref))**(alpha - 1.)
            if not cosmo is None:
                obj.r_comov = cosmo.get_r_comov(obj.z_qso)
                obj.dist_m = cosmo.get_dist_m(obj.z_qso)

    userprint("\n")

    return objs, z_qso.min()

def read_spall(in_dir, thingid):
    """Loads thingid, plate, mjd, and fiberid from spAll file

    Args:
        in_dir: str
            Directory to spectra files
        thingid: array of int
            Thingid of the observations
    Returns:
        Arrays with thingid, plate, mjd, and fiberid
    """
    folder = in_dir.replace("spectra/", "")
    folder = folder.replace("lite", "").replace("full", "")
    filenames = glob.glob(folder + "/spAll-*.fits")

    if len(filenames) > 1:
        userprint("ERROR: found multiple spAll files")
        userprint(("ERROR: try running with --bestobs option (but you will "
                   "lose reobservations)"))
        for filename in filenames:
            userprint("found: ", filename)
        sys.exit(1)
    if len(filenames) == 0:
        userprint(("ERROR: can't find required spAll file in "
                   "{}").format(in_dir))
        userprint(("ERROR: try runnint with --best-obs option (but you "
                   "will lose reobservations)"))
        sys.exit(1)

    spall = fitsio.FITS(filenames[0])
    userprint("INFO: reading spAll from {}".format(filenames[0]))
    thingid_spall = spall[1]["THING_ID"][:]
    plate_spall = spall[1]["PLATE"][:]
    mjd_spall = spall[1]["MJD"][:]
    fiberid_spall = spall[1]["FIBERID"][:]
    quality_spall = spall[1]["PLATEQUALITY"][:].astype(str)
    z_warn_spall = spall[1]["ZWARNING"][:]

    w = np.in1d(thingid_spall, thingid)
    userprint("INFO: Found {} spectra with required THING_ID".format(w.sum()))
    w &= quality_spall == "good"
    userprint("INFO: Found {} spectra with 'good' plate".format(w.sum()))
    ## Removing spectra with the following ZWARNING bits set:
    ## SKY, LITTLE_COVERAGE, UNPLUGGED, BAD_TARGET, NODATA
    ## https://www.sdss.org/dr14/algorithms/bitmasks/#ZWARNING
    bad_z_warn_bit = {0: 'SKY',
                      1: 'LITTLE_COVERAGE',
                      7: 'UNPLUGGED',
                      8: 'BAD_TARGET',
                      9: 'NODATA'}
    for z_warn_bit, z_warn_bit_name in bad_z_warn_bit.items():
        w &= z_warn_spall & 2**z_warn_bit
        userprint(("INFO: Found {} spectra without {} bit set: "
                   "{}").format(w.sum(), z_warn_bit, z_warn_bit_name))
    userprint("INFO: # unique objs: ", len(thingid))
    userprint("INFO: # spectra: ", w.sum())
    spall.close()

    return thingid_spall[w], plate_spall[w], mjd_spall[w], fiberid_spall[w]
