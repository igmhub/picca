from __future__ import print_function

import scipy as sp
import sys
import fitsio
import glob
import healpy

try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

def print(*args, **kwds):
    __builtin__.print(*args,**kwds)
    sys.stdout.flush()

def cov(da,we):

    mda = (da*we).sum(axis=0)
    swe = we.sum(axis=0)
    w = swe>0.
    mda[w] /= swe[w]

    wda = we*(da-mda)

    print("Computing cov...")

    co = wda.T.dot(wda)
    sswe = swe*swe[:,None]
    w = sswe>0.
    co[w] /= sswe[w]

    return co
def smooth_cov(da,we,rp,rt,drt=4,drp=4):

    co = cov(da,we)

    nda = da.shape[1]
    var = sp.diagonal(co)
    if sp.any(var==0.):
        print('WARNING: data has some empty bins, impossible to smooth')
        print('WARNING: returning the unsmoothed covariance')
        return co

    cor = co/sp.sqrt(var*var[:,None])

    cor_smooth = sp.zeros([nda,nda])

    dcor={}
    dncor={}

    for i in range(nda):
        print("\rsmoothing {}".format(i),end="")
        for j in range(i+1,nda):
            idrp = round(abs(rp[j]-rp[i])/drp)
            idrt = round(abs(rt[i]-rt[j])/drt)
            if not (idrp,idrt) in dcor:
                dcor[(idrp,idrt)]=0.
                dncor[(idrp,idrt)]=0

            dcor[(idrp,idrt)] +=cor[i,j]
            dncor[(idrp,idrt)] +=1

    for i in range(nda):
        cor_smooth[i,i]=1.
        for j in range(i+1,nda):
            idrp = round(abs(rp[j]-rp[i])/drp)
            idrt = round(abs(rt[i]-rt[j])/drt)
            cor_smooth[i,j]=dcor[(idrp,idrt)]/dncor[(idrp,idrt)]
            cor_smooth[j,i]=cor_smooth[i,j]


    print("\n")
    co_smooth = cor_smooth * sp.sqrt(var*var[:,None])
    return co_smooth

def desi_from_truth_to_drq(truth,targets,drq,spectype="QSO"):
    '''
    Transform a desi truth.fits file and a
    desi targets.fits into a drq like file

    '''

    ## Truth table
    vac = fitsio.FITS(truth)

    w = sp.ones(vac[1]["TARGETID"][:].size).astype(bool)
    print(" start                 : nb object in cat = {}".format(w.sum()) )
    w &= sp.char.strip(vac[1]["TRUESPECTYPE"][:].astype(str))==spectype
    print(" and TRUESPECTYPE=={}  : nb object in cat = {}".format(spectype,w.sum()) )

    thid = vac[1]["TARGETID"][:][w]
    zqso = vac[1]["TRUEZ"][:][w]
    vac.close()
    ra = sp.zeros(thid.size)
    dec = sp.zeros(thid.size)
    plate = thid
    mjd = thid
    fid = thid

    ### Get RA and DEC from targets
    vac = fitsio.FITS(targets)
    thidTargets = vac[1]["TARGETID"][:]
    raTargets = vac[1]["RA"][:].astype('float64')
    decTargets = vac[1]["DEC"][:].astype('float64')
    vac.close()

    from_TARGETID_to_idx = {}
    for i,t in enumerate(thidTargets):
        from_TARGETID_to_idx[t] = i
    keys_from_TARGETID_to_idx = from_TARGETID_to_idx.keys()

    for i,t in enumerate(thid):
        if t not in keys_from_TARGETID_to_idx: continue
        idx = from_TARGETID_to_idx[t]
        ra[i] = raTargets[idx]
        dec[i] = decTargets[idx]
    if (ra==0.).sum()!=0 or (dec==0.).sum()!=0:
        w = ra!=0.
        w &= dec!=0.
        print(" and RA and DEC        : nb object in cat = {}".format(w.sum()))

        ra = ra[w]
        dec = dec[w]
        zqso = zqso[w]
        thid = thid[w]
        plate = plate[w]
        mjd = mjd[w]
        fid = fid[w]

    ### Save
    out = fitsio.FITS(drq,'rw',clobber=True)
    cols=[ra,dec,thid,plate,mjd,fid,zqso]
    names=['RA','DEC','THING_ID','PLATE','MJD','FIBERID','Z']
    out.write(cols,names=names)
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
    vac = fitsio.FITS(ztarget)
    sptype = sp.char.strip(vac[1]['SPECTYPE'][:].astype(str))

    ## Sanity
    print(' start               : nb object in cat = {}'.format(sptype.size) )
    w = vac[1]['ZWARN'][:]==0.
    print(' and zwarn==0        : nb object in cat = {}'.format(w.sum()) )
    w &= sptype==spectype
    print(' and spectype=={}    : nb object in cat = {}'.format(spectype,w.sum()) )

    ra = vac[1]['RA'][:][w].astype('float64')
    dec = vac[1]['DEC'][:][w].astype('float64')
    zqso = vac[1]['Z'][:][w]
    thid = vac[1]['TARGETID'][:][w]

    vac.close()

    ###
    if not downsampling_z_cut is None and not downsampling_nb is None:
        if ra.size<downsampling_nb:
            print('WARNING:: Trying to downsample, when nb cat = {} and nb downsampling = {}'.format(ra.size,downsampling_nb) )
        else:
            select_fraction = downsampling_nb/(zqso>downsampling_z_cut).sum()
            sp.random.seed(0)
            select = sp.random.choice(sp.arange(ra.size),size=int(ra.size*select_fraction),replace=False)
            ra = ra[select]
            dec = dec[select]
            zqso = zqso[select]
            thid = thid[select]
            print(' and donsampling     : nb object in cat = {}, nb z > {} = {}'.format(ra.size, downsampling_z_cut, (zqso>downsampling_z_cut).sum()) )

    ### Save
    out = fitsio.FITS(drq,'rw',clobber=True)
    cols = [ra,dec,thid,thid,thid,thid,zqso]
    names = ['RA','DEC','THING_ID','PLATE','MJD','FIBERID','Z']
    out.write(cols, names=names)
    out.close()

    return
def desi_convert_transmission_to_delta_files(zcat,outdir,indir=None,infiles=None,lObs_min=3600.,lObs_max=5500.,lRF_min=1040.,lRF_max=1200.,dll=3.e-4,nspec=None):
    from picca.data import delta
    """Convert desi transmission files to picca delta files

    Args:
        zcat (str): path to the catalog of object to extract the transmission from
        indir (str): path to transmission files directory
        outdir (str): path to write delta files directory
        lObs_min (float) = 3600.: min observed wavelength in Angstrom
        lObs_max (float) = 5500.: max observed wavelength in Angstrom
        lRF_min (float) = 1040.: min Rest Frame wavelength in Angstrom
        lRF_max (float) = 1200.: max Rest Frame wavelength in Angstrom
        dll (float) = 3.e-4: size of the bins in log lambda
        nspec (int) = None: number of spectra, if 'None' use all

    Returns:
        None
    """

    ### Catalog of objects
    h = fitsio.FITS(zcat)
    key_val = sp.char.strip(sp.array([ h[1].read_header()[k] for k in h[1].read_header().keys()]).astype(str))
    if 'TARGETID' in key_val:
        zcat_thid = h[1]['TARGETID'][:]
    elif 'THING_ID' in key_val:
        zcat_thid = h[1]['THING_ID'][:]
    w = h[1]['Z'][:]>max(0.,lObs_min/lRF_max -1.)
    w &= h[1]['Z'][:]<max(0.,lObs_max/lRF_min -1.)
    zcat_ra = h[1]['RA'][:][w].astype('float64')*sp.pi/180.
    zcat_dec = h[1]['DEC'][:][w].astype('float64')*sp.pi/180.
    zcat_thid = zcat_thid[w]
    h.close()

    ### List of transmission files
    if (indir is None and infiles is None) or (indir is not None and infiles is not None):
        print("ERROR: No transmisson input files or both 'indir' and 'infiles' given")
        sys.exit()
    elif indir is not None:
        fi = glob.glob(indir+'/*/*/transmission*.fits') + glob.glob(indir+'/*/*/transmission*.fits.gz')
        h = fitsio.FITS(sp.sort(sp.array(fi))[0])
        in_nside = h[1].read_header()['NSIDE']
        nest = True
        h.close()
        in_pixs = healpy.ang2pix(in_nside, sp.pi/2.-zcat_dec, zcat_ra, nest=nest)
        fi = sp.sort(sp.array([ indir+'/'+str(int(f/100))+'/'+str(f)+'/transmission-'+str(in_nside)+'-'+str(f)+'.fits' for f in sp.unique(in_pixs)]))
    else:
        fi = sp.sort(sp.array(infiles))

    ### Stack the transmission
    lmin = sp.log10(lObs_min)
    lmax = sp.log10(lObs_max)
    nstack = int((lmax-lmin)/dll)+1
    T_stack = sp.zeros(nstack)
    n_stack = sp.zeros(nstack)

    deltas = {}

    ### Read
    for nf, f in enumerate(fi):
        print("\rread {} of {} {}".format(nf,fi.size,sp.sum([ len(deltas[p]) for p in list(deltas.keys())])), end="")
        h = fitsio.FITS(f)
        thid = h[1]['MOCKID'][:]
        if sp.in1d(thid,zcat_thid).sum()==0:
            h.close()
            continue
        ra = h[1]['RA'][:].astype('float64')*sp.pi/180.
        dec = h[1]['DEC'][:].astype('float64')*sp.pi/180.
        z = h[1]['Z'][:]
        ll = sp.log10(h[2].read())
        trans = h[3].read()
        nObj = z.size
        pixnum = f.split('-')[-1].split('.')[0]

        if trans.shape[0]!=nObj:
            trans = trans.transpose()

        bins = sp.floor((ll-lmin)/dll+0.5).astype(int)
        tll = lmin + bins*dll
        lObs = (10**tll)*sp.ones(nObj)[:,None]
        lRF = (10**tll)/(1.+z[:,None])
        w = sp.zeros_like(trans).astype(int)
        w[ (lObs>=lObs_min) & (lObs<lObs_max) & (lRF>lRF_min) & (lRF<lRF_max) ] = 1
        nbPixel = sp.sum(w,axis=1)
        cut = nbPixel>=50
        cut &= sp.in1d(thid,zcat_thid)
        if cut.sum()==0:
            h.close()
            continue

        ra = ra[cut]
        dec = dec[cut]
        z = z[cut]
        thid = thid[cut]
        trans = trans[cut,:]
        w = w[cut,:]
        nObj = z.size
        h.close()

        deltas[pixnum] = []
        for i in range(nObj):
            tll = ll[w[i,:]>0]
            ttrans = trans[i,:][w[i,:]>0]

            bins = sp.floor((tll-lmin)/dll+0.5).astype(int)
            cll = lmin + sp.arange(nstack)*dll
            cfl = sp.bincount(bins,weights=ttrans,minlength=nstack)
            civ = sp.bincount(bins,minlength=nstack).astype(float)

            ww = civ>0.
            if ww.sum()<50: continue
            T_stack += cfl
            n_stack += civ
            cll = cll[ww]
            cfl = cfl[ww]/civ[ww]
            civ = civ[ww]
            deltas[pixnum].append(delta(thid[i],ra[i],dec[i],z[i],thid[i],thid[i],thid[i],cll,civ,None,cfl,1,None,None,None,None,None,None))
        if not nspec is None and sp.sum([ len(deltas[p]) for p in list(deltas.keys())])>=nspec: break

    print('\n')

    ### Get stacked transmission
    w = n_stack>0.
    T_stack[w] /= n_stack[w]

    ### Transform transmission to delta and store it
    for nf, p in enumerate(sorted(list(deltas.keys()))):
        print("\rwrite {} of {} ".format(nf,len(list(deltas.keys()))), end="")
        out = fitsio.FITS(outdir+'/delta-{}'.format(p)+'.fits.gz','rw',clobber=True)
        for d in deltas[p]:
            bins = sp.floor((d.ll-lmin)/dll+0.5).astype(int)
            d.de = d.de/T_stack[bins] - 1.
            d.we *= T_stack[bins]**2

            hd = {}
            hd['RA'] = d.ra
            hd['DEC'] = d.dec
            hd['Z'] = d.zqso
            hd['PMF'] = '{}-{}-{}'.format(d.plate,d.mjd,d.fid)
            hd['THING_ID'] = d.thid
            hd['PLATE'] = d.plate
            hd['MJD'] = d.mjd
            hd['FIBERID'] = d.fid
            hd['ORDER'] = d.order

            cols = [d.ll,d.de,d.we,sp.ones(d.ll.size)]
            names = ['LOGLAM','DELTA','WEIGHT','CONT']
            out.write(cols,names=names,header=hd,extname=str(d.thid))
        out.close()

    print("")

    return
def compute_ang_max(cosmo,rt_max,zmin,zmin2=None):
    '''
    Compute the maximum angle given by the maximum transverse
    separation the correlation should be calculated to
    '''

    if zmin2 is None:
        zmin2 = zmin

    rmin1 = cosmo.r_comoving(zmin)
    rmin2 = cosmo.r_comoving(zmin2)

    if rmin1+rmin2<rt_max:
        angmax = sp.pi
    else:
        angmax = 2.*sp.arcsin(rt_max/(rmin1+rmin2))

    return angmax
def shuffle_distrib_obj(obj,seed):
    '''Shuffle the distribution of objects by giving to an object the redshift
        of another random one.

    Args:
        obj (dic): Catalog of objects
        seed (int): seed for the given realization of the shuffle

    Returns:
        obj (dic): Catalog of objects
    '''
    dic = {}
    lst_p = ['we','zqso','r_comov']
    for p in lst_p:
        dic[p] = [getattr(o, p) for oss in obj.values() for o in oss]

    sp.random.seed(seed)
    idx = sp.arange(len(dic['zqso']))
    sp.random.shuffle(idx)

    i = 0
    for oss in obj.values():
        for o in oss:
            for p in lst_p:
                setattr(o,p,dic[p][idx[i]])
            i += 1
    return obj
