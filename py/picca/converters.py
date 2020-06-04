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
import numpy as np
import scipy as sp
import sys
import fitsio
import glob
import healpy
import scipy.interpolate as interpolate
import iminuit

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
            dla_cat = {key:[] for key in from_key_to_index}
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
        'Dec': 'DEC'}
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
        'DEC': np.float64}

    # format catalogue
    cat = {value: np.array(dla_cat[key], dtype=from_picca_key_to_type[value])
           for key, value in from_noterdaeme_key_to_picca_key.items()}

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
    cat['RA'] = np.array([ra[from_thingid_to_index[t]]
                          for t in cat['THING_ID']])
    cat['DEC'] = np.array([dec[from_thingid_to_index[t]]
                           for t in cat['THING_ID']])
    cat['ZQSO'] = np.array([z_qso[from_thingid_to_index[t]]
                            for t in cat['THING_ID']])

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


def desi_from_truth_to_drq(truth_filename, targets_filename, out_path,
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
    userprint(" and TRUESPECTYPE=={}  : nb object in cat = {}".format(spec_type,
                                                                      w.sum()))
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


def desi_from_ztarget_to_drq(in_path, out_path, spec_type='QSO',
                             downsampling_z_cut=None, downsampling_num=None):
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
    userprint(' and spectype=={}    : nb object in cat = {}'.format(spec_type,
                                                                    w.sum()))
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
            select_fraction = (downsampling_num/
                               (cat['Z'] > downsampling_z_cut).sum())
            np.random.seed(0)
            w = np.random.choice(np.arange(cat['RA'].size),
                                 size=int(cat['RA'].size*select_fraction),
                                 replace=False)
            for key in cat:
                cat[key] = cat[key][w]
            userprint((" and donsampling     : nb object in cat = {}, nb z > "
                       "{} = {}").format(cat['RA'].size,
                                         downsampling_z_cut,
                                         (cat["Z"] > downsampling_z_cut).sum()))

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


def desi_convert_transmission_to_delta_files(zcat,outdir,indir=None,infiles=None,lObs_min=3600.,lObs_max=5500.,lRF_min=1040.,lRF_max=1200.,delta_log_lambda=3.e-4,nspec=None):
    """Convert desi transmission files to picca delta files

    Args:
        zcat (str): path to the catalog of object to extract the transmission from
        indir (str): path to transmission files directory
        outdir (str): path to write delta files directory
        lObs_min (float) = 3600.: min observed wavelength in Angstrom
        lObs_max (float) = 5500.: max observed wavelength in Angstrom
        lRF_min (float) = 1040.: min Rest Frame wavelength in Angstrom
        lRF_max (float) = 1200.: max Rest Frame wavelength in Angstrom
        delta_log_lambda (float) = 3.e-4: size of the bins in log lambda
        nspec (int) = None: number of spectra, if 'None' use all

    Returns:
        None
    """

    ### Catalog of objects
    h = fitsio.FITS(zcat)
    key_val = sp.char.strip(sp.array([ h[1].read_header()[k] for k in h[1].read_header().keys()]).astype(str))
    if 'TARGETID' in key_val:
        zcat_thingid = h[1]['TARGETID'][:]
    elif 'THING_ID' in key_val:
        zcat_thingid = h[1]['THING_ID'][:]
    w = h[1]['Z'][:]>max(0.,lObs_min/lRF_max -1.)
    w &= h[1]['Z'][:]<max(0.,lObs_max/lRF_min -1.)
    zcat_ra = h[1]['RA'][:][w].astype('float64')*sp.pi/180.
    zcat_dec = h[1]['DEC'][:][w].astype('float64')*sp.pi/180.
    zcat_thingid = zcat_thingid[w]
    h.close()
    userprint('INFO: Found {} quasars'.format(zcat_ra.size))

    ### List of transmission files
    if (indir is None and infiles is None) or (indir is not None and infiles is not None):
        userprint("ERROR: No transmisson input files or both 'indir' and 'infiles' given")
        sys.exit()
    elif indir is not None:
        fi = glob.glob(indir+'/*/*/transmission*.fits*')
        fi = sp.sort(sp.array(fi))
        h = fitsio.FITS(fi[0])
        in_nside = h['METADATA'].read_header()['HPXNSIDE']
        nest = h['METADATA'].read_header()['HPXNEST']
        h.close()
        in_pixs = healpy.ang2pix(in_nside, sp.pi/2.-zcat_dec, zcat_ra, nest=nest)
        if fi[0].endswith('.gz'):
            endoffile = '.gz'
        else:
            endoffile = ''
        fi = sp.sort(sp.array(['{}/{}/{}/transmission-{}-{}.fits{}'.format(indir,int(f//100),f,in_nside,f,endoffile) for f in np.unique(in_pixs)]))
    else:
        fi = sp.sort(sp.array(infiles))
    userprint('INFO: Found {} files'.format(fi.size))

    ### Stack the transmission
    log_lambda_min = sp.log10(lObs_min)
    log_lambda_max = sp.log10(lObs_max)
    nstack = int((log_lambda_max-log_lambda_min)/delta_log_lambda)+1
    T_stack = np.zeros(nstack)
    n_stack = np.zeros(nstack)

    deltas = {}

    ### Read
    for nf, f in enumerate(fi):
        userprint("\rread {} of {} {}".format(nf,fi.size,sp.sum([ len(deltas[p]) for p in deltas.keys()])), end="")
        h = fitsio.FITS(f)
        thingid = h['METADATA']['MOCKID'][:]
        if sp.in1d(thingid,zcat_thingid).sum()==0:
            h.close()
            continue
        ra = h['METADATA']['RA'][:].astype(sp.float64)*sp.pi/180.
        dec = h['METADATA']['DEC'][:].astype(sp.float64)*sp.pi/180.
        z = h['METADATA']['Z'][:]
        log_lambda = sp.log10(h['WAVELENGTH'].read())
        if 'F_LYA' in h :
            trans = h['F_LYA'].read()
        else:
            trans = h['TRANSMISSION'].read()

        nObj = z.size
        pixnum = f.split('-')[-1].split('.')[0]

        if trans.shape[0]!=nObj:
            trans = trans.transpose()

        bins = sp.floor((log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
        tll = log_lambda_min + bins*delta_log_lambda
        lObs = (10**tll)*sp.ones(nObj)[:,None]
        lRF = (10**tll)/(1.+z[:,None])
        w = np.zeros_like(trans).astype(int)
        w[ (lObs>=lObs_min) & (lObs<lObs_max) & (lRF>lRF_min) & (lRF<lRF_max) ] = 1
        nbPixel = sp.sum(w,axis=1)
        cut = nbPixel>=50
        cut &= sp.in1d(thingid,zcat_thingid)
        if cut.sum()==0:
            h.close()
            continue

        ra = ra[cut]
        dec = dec[cut]
        z = z[cut]
        thingid = thingid[cut]
        trans = trans[cut,:]
        w = w[cut,:]
        nObj = z.size
        h.close()

        deltas[pixnum] = []
        for i in range(nObj):
            tll = log_lambda[w[i,:]>0]
            ttrans = trans[i,:][w[i,:]>0]

            bins = sp.floor((tll-log_lambda_min)/delta_log_lambda+0.5).astype(int)
            cll = log_lambda_min + np.arange(nstack)*delta_log_lambda
            cfl = sp.bincount(bins,weights=ttrans,minlength=nstack)
            civ = sp.bincount(bins,minlength=nstack).astype(float)

            ww = civ>0.
            if ww.sum()<50: continue
            T_stack += cfl
            n_stack += civ
            cll = cll[ww]
            cfl = cfl[ww]/civ[ww]
            civ = civ[ww]
            deltas[pixnum].append(Delta(thingid[i],ra[i],dec[i],z[i],thingid[i],thingid[i],thingid[i],cll,civ,None,cfl,1,None,None,None,None,None,None))
        if not nspec is None and sp.sum([ len(deltas[p]) for p in deltas.keys()])>=nspec: break

    userprint('\n')

    ### Get stacked transmission
    w = n_stack>0.
    T_stack[w] /= n_stack[w]

    ### Transform transmission to delta and store it
    for nf, p in enumerate(sorted(deltas.keys())):
        if len(deltas[p])==0:
            userprint('No data in {}'.format(p))
            continue
        out = fitsio.FITS(outdir+'/delta-{}'.format(p)+'.fits.gz','rw',clobber=True)
        for d in deltas[p]:
            bins = sp.floor((d.log_lambda-log_lambda_min)/delta_log_lambda+0.5).astype(int)
            d.delta = d.delta/T_stack[bins] - 1.
            d.weights *= T_stack[bins]**2

            hd = {}
            hd['RA'] = d.ra
            hd['DEC'] = d.dec
            hd['Z'] = d.z_qso
            hd['PMF'] = '{}-{}-{}'.format(d.plate,d.mjd,d.fiberid)
            hd['THING_ID'] = d.thingid
            hd['PLATE'] = d.plate
            hd['MJD'] = d.mjd
            hd['FIBERID'] = d.fiberid
            hd['ORDER'] = d.order

            cols = [d.log_lambda,d.delta,d.weights,sp.ones(d.log_lambda.size)]
            names = ['LOGLAM','DELTA','WEIGHT','CONT']
            out.write(cols,names=names,header=hd,extname=str(d.thingid))
        out.close()
        userprint("\rwrite {} of {}: {} quasars".format(nf,len(list(deltas.keys())), len(deltas[p])), end="")

    userprint("")

    return
