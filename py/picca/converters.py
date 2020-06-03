"""This module provides functions to format several catalogues

These are:
    - eBOSS_convert_DLA
    - desi_convert_DLA
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


def eBOSS_convert_DLA(inPath,drq,outPath,drqzkey='Z'):
    """Converts Pasquier Noterdaeme ASCII DLA catalog to a fits file

    Args:

    """

    f = open(os.path.expandvars(inPath),'r')
    for l in f:
        l = l.split()
        if (len(l)==0) or (l[0][0]=='#') or (l[0][0]=='-'):
            continue
        elif l[0]=='ThingID':
            fromkeytoindex = { el:i for i,el in enumerate(l) }
            dcat = { el:[] for el in fromkeytoindex.keys() }
            for kk in 'MJD-plate-fiber'.split('-'):
                dcat[kk] = []
            continue
        else:
            for k in fromkeytoindex.keys():
                v = l[fromkeytoindex[k]]
                if k=='MJD-plate-fiber':
                    v = v.split('-')
                    for i,kk in enumerate('MJD-plate-fiber'.split('-')):
                        dcat[kk] += [v[i]]
                dcat[k] += [v]
    f.close()
    userprint('INFO: Found {} DLA from {} quasars'.format(len(dcat['ThingID']), np.unique(dcat['ThingID']).size))

    fromNoterdaemeKey2Picca = {'ThingID':'THING_ID', 'z_abs':'Z', 'zqso':'ZQSO','NHI':'NHI',
        'plate':'PLATE','MJD':'MJD','fiber':'FIBERID',
        'RA':'RA', 'Dec':'DEC'}
    fromPiccaKey2Type = {'THING_ID':sp.int64, 'Z':sp.float64, 'ZQSO':sp.float64, 'NHI':sp.float64,
        'PLATE':sp.int64,'MJD':sp.int64,'FIBERID':sp.int64,
        'RA':sp.float64, 'DEC':sp.float64}
    cat = { v:sp.array(dcat[k],dtype=fromPiccaKey2Type[v]) for k,v in fromNoterdaemeKey2Picca.items() }

    w = cat['THING_ID']>0
    userprint('INFO: Removed {} DLA, because THING_ID<=0'.format((cat['THING_ID']<=0).sum()))
    w &= cat['Z']>0.
    userprint('INFO: Removed {} DLA, because Z<=0.'.format((cat['Z']<=0.).sum()))
    for k in cat.keys():
        cat[k] = cat[k][w]

    h = fitsio.FITS(drq)
    thingid = h[1]['THING_ID'][:]
    ra = h[1]['RA'][:]
    dec = h[1]['DEC'][:]
    z_qso = h[1][drqzkey][:]
    h.close()
    fromThingid2idx = { el:i for i,el in enumerate(thingid) }
    cat['RA'] = sp.array([ ra[fromThingid2idx[el]] for el in cat['THING_ID'] ])
    cat['DEC'] = sp.array([ dec[fromThingid2idx[el]] for el in cat['THING_ID'] ])
    cat['ZQSO'] = sp.array([ z_qso[fromThingid2idx[el]] for el in cat['THING_ID'] ])

    w = cat['RA']!=cat['DEC']
    userprint('INFO: Removed {} DLA, because RA==DEC'.format((cat['RA']==cat['DEC']).sum()))
    w &= cat['RA']!=0.
    userprint('INFO: Removed {} DLA, because RA==0'.format((cat['RA']==0.).sum()))
    w &= cat['DEC']!=0.
    userprint('INFO: Removed {} DLA, because DEC==0'.format((cat['DEC']==0.).sum()))
    w &= cat['ZQSO']>0.
    userprint('INFO: Removed {} DLA, because ZQSO<=0.'.format((cat['ZQSO']<=0.).sum()))
    for k in cat.keys():
        cat[k] = cat[k][w]

    w = sp.argsort(cat['Z'])
    for k in cat.keys():
        cat[k] = cat[k][w]
    w = sp.argsort(cat['THING_ID'])
    for k in cat.keys():
        cat[k] = cat[k][w]
    cat['DLAID'] = np.arange(1,cat['Z'].size+1,dtype=sp.int64)

    for k in ['RA','DEC']:
        cat[k] = cat[k].astype('float64')

    ### Save
    out = fitsio.FITS(outPath,'rw',clobber=True)
    cols = [ v for v in cat.values() ]
    names = [ k for k in cat.keys() ]
    out.write(cols,names=names,extname='DLACAT')
    out.close()

    return


def desi_convert_DLA(inPath,outPath):
    """
    Convert a catalog of DLA from a DESI format to
    the format used by picca
    """

    fromDESIkey2piccaKey = {'RA':'RA', 'DEC':'DEC',
        'Z':'Z_DLA_RSD', 'ZQSO':'Z_QSO_RSD',
        'NHI':'N_HI_DLA', 'THING_ID':'MOCKID', 'DLAID':'DLAID',
        'PLATE':'MOCKID', 'MJD':'MOCKID', 'FIBERID':'MOCKID' }

    cat = {}
    h = fitsio.FITS(inPath)
    for k,v in fromDESIkey2piccaKey.items():
        cat[k] = h['DLACAT'][v][:]
    h.close()
    userprint('INFO: Found {} DLA from {} quasars'.format(cat['Z'].size, np.unique(cat['THING_ID']).size))

    w = sp.argsort(cat['THING_ID'])
    for k in cat.keys():
        cat[k] = cat[k][w]

    for k in ['RA','DEC']:
        cat[k] = cat[k].astype('float64')

    ### Save
    out = fitsio.FITS(outPath,'rw',clobber=True)
    cols = [ v for v in cat.values() ]
    names = [ k for k in cat.keys() ]
    out.write(cols,names=names,extname='DLACAT')
    out.close()

    return
def desi_from_truth_to_drq(truth,targets,drq,spectype="QSO"):
    '''
    Transform a desi truth.fits file and a
    desi targets.fits into a drq like file

    '''

    ## Truth table
    vac = fitsio.FITS(truth)

    w = sp.ones(vac[1]['TARGETID'][:].size).astype(bool)
    userprint(" start                 : nb object in cat = {}".format(w.sum()) )
    w &= sp.char.strip(vac[1]['TRUESPECTYPE'][:].astype(str))==spectype
    userprint(" and TRUESPECTYPE=={}  : nb object in cat = {}".format(spectype,w.sum()) )

    thingid = vac[1]['TARGETID'][:][w]
    z_qso = vac[1]['TRUEZ'][:][w]
    vac.close()
    ra = np.zeros(thingid.size)
    dec = np.zeros(thingid.size)
    plate = thingid
    mjd = thingid
    fiberid = thingid

    ### Get RA and DEC from targets
    vac = fitsio.FITS(targets)
    thingidTargets = vac[1]['TARGETID'][:]
    raTargets = vac[1]['RA'][:].astype('float64')
    decTargets = vac[1]['DEC'][:].astype('float64')
    vac.close()

    from_TARGETID_to_idx = {}
    for i,t in enumerate(thingidTargets):
        from_TARGETID_to_idx[t] = i
    keys_from_TARGETID_to_idx = from_TARGETID_to_idx.keys()

    for i,t in enumerate(thingid):
        if t not in keys_from_TARGETID_to_idx: continue
        idx = from_TARGETID_to_idx[t]
        ra[i] = raTargets[idx]
        dec[i] = decTargets[idx]
    if (ra==0.).sum()!=0 or (dec==0.).sum()!=0:
        w = ra!=0.
        w &= dec!=0.
        userprint(" and RA and DEC        : nb object in cat = {}".format(w.sum()))

        ra = ra[w]
        dec = dec[w]
        z_qso = z_qso[w]
        thingid = thingid[w]
        plate = plate[w]
        mjd = mjd[w]
        fiberid = fiberid[w]

    ### Save
    out = fitsio.FITS(drq,'rw',clobber=True)
    cols=[ra,dec,thingid,plate,mjd,fiberid,z_qso]
    names=['RA','DEC','THING_ID','PLATE','MJD','FIBERID','Z']
    out.write(cols,names=names,extname='CAT')
    out.close()

    return

def desi_from_ztarget_to_drq(ztarget,drq,spectype='QSO',downsampling_z_cut=None, downsampling_nb=None):
    """Transforms a catalog of object in desi format to a catalog in DRQ format

    Args:
        zcat (str): path to the catalog of object
        drq (str): path to write the DRQ catalog
        spectype (str): Spectype of the object, can be any spectype
            in desi catalog. Ex: 'STAR', 'GALAXY', 'QSO'
        downsampling_z_cut (float) : Minimum redshift to downsample
            the data, if 'None' no downsampling
        downsampling_nb (int) : Target number of object above redshift
            downsampling-z-cut, if 'None' no downsampling

    Returns:
        None

    """

    ## Info of the primary observation
    h = fitsio.FITS(ztarget)
    sptype = sp.char.strip(h[1]['SPECTYPE'][:].astype(str))

    ## Sanity
    userprint(' start               : nb object in cat = {}'.format(sptype.size) )
    w = h[1]['ZWARN'][:]==0.
    userprint(' and zwarn==0        : nb object in cat = {}'.format(w.sum()) )
    w &= sptype==spectype
    userprint(' and spectype=={}    : nb object in cat = {}'.format(spectype,w.sum()) )

    cat = {}
    lst = {'RA':'RA', 'DEC':'DEC', 'Z':'Z',
        'THING_ID':'TARGETID', 'PLATE':'TARGETID', 'MJD':'TARGETID', 'FIBERID':'TARGETID'}
    for k,v in lst.items():
        cat[k] = h[1][v][:][w]
    h.close()

    for k in ['RA','DEC']:
        cat[k] = cat[k].astype('float64')

    ###
    if not downsampling_z_cut is None and not downsampling_nb is None:
        if cat['RA'].size<downsampling_nb:
            userprint('WARNING:: Trying to downsample, when nb cat = {} and nb downsampling = {}'.format(cat['RA'].size,downsampling_nb) )
        else:
            select_fraction = downsampling_nb/(cat['Z']>downsampling_z_cut).sum()
            sp.random.seed(0)
            w = sp.random.choice(np.arange(cat['RA'].size),size=int(cat['RA'].size*select_fraction),replace=False)
            for k in cat.keys():
                cat[k] = cat[k][w]
            userprint(' and donsampling     : nb object in cat = {}, nb z > {} = {}'.format(cat['RA'].size, downsampling_z_cut, (z_qso>downsampling_z_cut).sum()) )

    w = sp.argsort(cat['THING_ID'])
    for k in cat.keys():
        cat[k] = cat[k][w]

    ### Save
    out = fitsio.FITS(drq,'rw',clobber=True)
    cols = [ v for v in cat.values() ]
    names = [ k for k in cat.keys() ]
    out.write(cols, names=names,extname='CAT')
    out.close()

    return

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
