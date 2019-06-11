from __future__ import print_function

import os
import scipy as sp
import sys
import fitsio
import glob
import healpy
import scipy.interpolate as interpolate
import iminuit

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
def smooth_cov(da,we,rp,rt,drt=4,drp=4,co=None):

    if co is None:
        co = cov(da,we)

    nda = co.shape[1]
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

def smooth_cov_wick(infile,Wick_infile,outfile):
    """
    Model the missing correlation in the Wick computation
    with an exponential

    Args:
        infile (str): path to the correlation function
            (produced by picca_cf, picca_xcf)
        Wick_infile (str): path to the Wick correlation function
            (produced by picca_wick, picca_xwick)
        outfile (str): poutput path

    Returns:
        None
    """

    h = fitsio.FITS(infile)
    da = sp.array(h[2]['DA'][:])
    we = sp.array(h[2]['WE'][:])
    head = h[1].read_header()
    nt = head['NT']
    np = head['NP']
    h.close()

    co = cov(da,we)

    nbin = da.shape[1]
    var = sp.diagonal(co)
    if sp.any(var==0.):
        print('WARNING: data has some empty bins, impossible to smooth')
        print('WARNING: returning the unsmoothed covariance')
        return co

    cor = co/sp.sqrt(var*var[:,None])
    cor1d = cor.reshape(nbin*nbin)

    h = fitsio.FITS(Wick_infile)
    cow = sp.array(h[1]['CO'][:])
    h.close()

    varw = sp.diagonal(cow)
    if sp.any(varw==0.):
        print('WARNING: Wick covariance has bins with var = 0')
        print('WARNING: returning the unsmoothed covariance')
        return co

    corw = cow/sp.sqrt(varw*varw[:,None])
    corw1d = corw.reshape(nbin*nbin)

    Dcor1d = cor1d - corw1d

    #### indices
    ind = sp.arange(nbin)
    rtindex = ind%nt
    rpindex = ind//nt
    idrt2d = abs(rtindex-rtindex[:,None])
    idrp2d = abs(rpindex-rpindex[:,None])
    idrt1d = idrt2d.reshape(nbin*nbin)
    idrp1d = idrp2d.reshape(nbin*nbin)

    #### reduced covariance  (50*50)
    Dcor_red1d = sp.zeros(nbin)
    for idr in range(0,nbin):
        print("\rsmoothing {}".format(idr),end="")
        Dcor_red1d[idr] = sp.mean(Dcor1d[(idrp1d==rpindex[idr])&(idrt1d==rtindex[idr])])
    Dcor_red = Dcor_red1d.reshape(np,nt)
    print("")

    #### fit for L and A at each drp
    def corrfun(idrp,idrt,L,A):
        r = sp.sqrt(float(idrt)**2+float(idrp)**2) - float(idrp)
        return A*sp.exp(-r/L)
    def chisq(L,A,idrp):
        chi2 = 0.
        idrp = int(idrp)
        for idrt in range(1,nt):
            chi = Dcor_red[idrp,idrt]-corrfun(idrp,idrt,L,A)
            chi2 += chi**2
        chi2 = chi2*np*nbin
        return chi2

    Lfit = sp.zeros(np)
    Afit = sp.zeros(np)
    for idrp in range(np):
        m = iminuit.Minuit(chisq,L=5.,error_L=0.2,limit_L=(1.,400.),
            A=1.,error_A=0.2,
            idrp=idrp,fix_idrp=True,
            print_level=1,errordef=1.)
        m.migrad()
        Lfit[idrp] = m.values['L']
        Afit[idrp] = m.values['A']

    #### hybrid covariance from wick + fit
    co_smooth = sp.sqrt(var*var[:,None])

    cor0 = Dcor_red1d[rtindex==0]
    for i in range(nbin):
        print("\rupdating {}".format(i),end="")
        for j in range(i+1,nbin):
            idrp = idrp2d[i,j]
            idrt = idrt2d[i,j]
            newcov = corw[i,j]
            if (idrt == 0):
                newcov += cor0[idrp]
            else:
                newcov += corrfun(idrp,idrt,Lfit[idrp],Afit[idrp])
            co_smooth[i,j] *= newcov
            co_smooth[j,i] *= newcov

    print("\n")

    h = fitsio.FITS(outfile,'rw',clobber=True)
    h.write([co_smooth],names=['CO'],extname='COR')
    h.close()
    print(outfile,' written')

    return

def eBOSS_convert_DLA(inPath,drq,outPath,drqzkey='Z'):
    """
    Convert Pasquier Noterdaeme ASCII DLA catalog
    to a fits file
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
    print('INFO: Found {} DLA from {} quasars'.format(len(dcat['ThingID']), sp.unique(dcat['ThingID']).size))

    fromNoterdaemeKey2Picca = {'ThingID':'THING_ID', 'z_abs':'Z', 'zqso':'ZQSO','NHI':'NHI',
        'plate':'PLATE','MJD':'MJD','fiber':'FIBERID',
        'RA':'RA', 'Dec':'DEC'}
    fromPiccaKey2Type = {'THING_ID':sp.int64, 'Z':sp.float64, 'ZQSO':sp.float64, 'NHI':sp.float64,
        'PLATE':sp.int64,'MJD':sp.int64,'FIBERID':sp.int64,
        'RA':sp.float64, 'DEC':sp.float64}
    cat = { v:sp.array(dcat[k],dtype=fromPiccaKey2Type[v]) for k,v in fromNoterdaemeKey2Picca.items() }

    w = cat['THING_ID']>0
    print('INFO: Removed {} DLA, because THING_ID<=0'.format((cat['THING_ID']<=0).sum()))
    w &= cat['Z']>0.
    print('INFO: Removed {} DLA, because Z<=0.'.format((cat['Z']<=0.).sum()))
    for k in cat.keys():
        cat[k] = cat[k][w]

    h = fitsio.FITS(drq)
    thid = h[1]['THING_ID'][:]
    ra = h[1]['RA'][:]
    dec = h[1]['DEC'][:]
    zqso = h[1][drqzkey][:]
    h.close()
    fromThingid2idx = { el:i for i,el in enumerate(thid) }
    cat['RA'] = sp.array([ ra[fromThingid2idx[el]] for el in cat['THING_ID'] ])
    cat['DEC'] = sp.array([ dec[fromThingid2idx[el]] for el in cat['THING_ID'] ])
    cat['ZQSO'] = sp.array([ zqso[fromThingid2idx[el]] for el in cat['THING_ID'] ])

    w = cat['RA']!=cat['DEC']
    print('INFO: Removed {} DLA, because RA==DEC'.format((cat['RA']==cat['DEC']).sum()))
    w &= cat['RA']!=0.
    print('INFO: Removed {} DLA, because RA==0'.format((cat['RA']==0.).sum()))
    w &= cat['DEC']!=0.
    print('INFO: Removed {} DLA, because DEC==0'.format((cat['DEC']==0.).sum()))
    w &= cat['ZQSO']>0.
    print('INFO: Removed {} DLA, because ZQSO<=0.'.format((cat['ZQSO']<=0.).sum()))
    for k in cat.keys():
        cat[k] = cat[k][w]

    w = sp.argsort(cat['Z'])
    for k in cat.keys():
        cat[k] = cat[k][w]
    w = sp.argsort(cat['THING_ID'])
    for k in cat.keys():
        cat[k] = cat[k][w]
    cat['DLAID'] = sp.arange(1,cat['Z'].size+1,dtype=sp.int64)

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
    print('INFO: Found {} DLA from {} quasars'.format(cat['Z'].size, sp.unique(cat['THING_ID']).size))

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
    print(" start                 : nb object in cat = {}".format(w.sum()) )
    w &= sp.char.strip(vac[1]['TRUESPECTYPE'][:].astype(str))==spectype
    print(" and TRUESPECTYPE=={}  : nb object in cat = {}".format(spectype,w.sum()) )

    thid = vac[1]['TARGETID'][:][w]
    zqso = vac[1]['TRUEZ'][:][w]
    vac.close()
    ra = sp.zeros(thid.size)
    dec = sp.zeros(thid.size)
    plate = thid
    mjd = thid
    fid = thid

    ### Get RA and DEC from targets
    vac = fitsio.FITS(targets)
    thidTargets = vac[1]['TARGETID'][:]
    raTargets = vac[1]['RA'][:].astype('float64')
    decTargets = vac[1]['DEC'][:].astype('float64')
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
    print(' start               : nb object in cat = {}'.format(sptype.size) )
    w = h[1]['ZWARN'][:]==0.
    print(' and zwarn==0        : nb object in cat = {}'.format(w.sum()) )
    w &= sptype==spectype
    print(' and spectype=={}    : nb object in cat = {}'.format(spectype,w.sum()) )

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
            print('WARNING:: Trying to downsample, when nb cat = {} and nb downsampling = {}'.format(cat['RA'].size,downsampling_nb) )
        else:
            select_fraction = downsampling_nb/(cat['Z']>downsampling_z_cut).sum()
            sp.random.seed(0)
            w = sp.random.choice(sp.arange(cat['RA'].size),size=int(cat['RA'].size*select_fraction),replace=False)
            for k in cat.keys():
                cat[k] = cat[k][w]
            print(' and donsampling     : nb object in cat = {}, nb z > {} = {}'.format(cat['RA'].size, downsampling_z_cut, (zqso>downsampling_z_cut).sum()) )

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
    print('INFO: Found {} quasars'.format(zcat_ra.size))

    ### List of transmission files
    if (indir is None and infiles is None) or (indir is not None and infiles is not None):
        print("ERROR: No transmisson input files or both 'indir' and 'infiles' given")
        sys.exit()
    elif indir is not None:
        fi = glob.glob(indir+'/*/*/transmission*.fits*')
        fi = sp.sort(sp.array(fi))
        h = fitsio.FITS(fi[0])
        in_nside = h['METADATA'].read_header()['HPXNSIDE']
        nest = h['METADATA'].read_header()['HPXNEST']
        h.close()
        in_pixs = healpy.ang2pix(in_nside, sp.pi/2.-zcat_dec, zcat_ra, nest=nest)
        fi = sp.sort(sp.array(['{}/{}/{}/transmission-{}-{}.fits'.format(indir,int(f//100),f,in_nside,f) for f in sp.unique(in_pixs)]))
    else:
        fi = sp.sort(sp.array(infiles))
    print('INFO: Found {} files'.format(fi.size))

    ### Stack the transmission
    lmin = sp.log10(lObs_min)
    lmax = sp.log10(lObs_max)
    nstack = int((lmax-lmin)/dll)+1
    T_stack = sp.zeros(nstack)
    n_stack = sp.zeros(nstack)

    deltas = {}

    ### Read
    for nf, f in enumerate(fi):
        print("\rread {} of {} {}".format(nf,fi.size,sp.sum([ len(deltas[p]) for p in deltas.keys()])), end="")
        h = fitsio.FITS(f)
        thid = h['METADATA']['MOCKID'][:]
        if sp.in1d(thid,zcat_thid).sum()==0:
            h.close()
            continue
        ra = h['METADATA']['RA'][:].astype(sp.float64)*sp.pi/180.
        dec = h['METADATA']['DEC'][:].astype(sp.float64)*sp.pi/180.
        z = h['METADATA']['Z'][:]
        ll = sp.log10(h['WAVELENGTH'].read())
        trans = h['TRANSMISSION'].read()
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
        if not nspec is None and sp.sum([ len(deltas[p]) for p in deltas.keys()])>=nspec: break

    print('\n')

    ### Get stacked transmission
    w = n_stack>0.
    T_stack[w] /= n_stack[w]

    ### Transform transmission to delta and store it
    for nf, p in enumerate(sorted(deltas.keys())):
        if len(deltas[p])==0:
            print('No data in {}'.format(p))
            continue
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
        print("\rwrite {} of {}: {} quasars".format(nf,len(list(deltas.keys())), len(deltas[p])), end="")

    print("")

    return
def compute_ang_max(cosmo,rt_max,zmin,zmin2=None):
    '''
    Compute the maximum angle given by the maximum transverse
    separation the correlation should be calculated to
    '''

    if zmin2 is None:
        zmin2 = zmin

    rmin1 = cosmo.dm(zmin)
    rmin2 = cosmo.dm(zmin2)

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
def shuffle_distrib_forests(obj,seed):
    '''Shuffle the distribution of forests by assiging the angular
        positions from another forest

    Args:
        obj (dic): Catalog of forests
        seed (int): seed for the given realization of the shuffle

    Returns:
        obj (dic): Catalog of forest
    '''

    print('INFO: Shuffling the forests angular position with seed {}'.format(seed))

    dic = {}
    lst_p = ['ra','dec','xcart','ycart','zcart','cosdec','thid']
    for p in lst_p:
        dic[p] = [getattr(o, p) for oss in obj.values() for o in oss]
    sp.random.seed(seed)
    idx = sp.arange(len(dic['ra']))
    sp.random.shuffle(idx)

    i = 0
    for oss in obj.values():
        for o in oss:
            for p in lst_p:
                setattr(o,p,dic[p][idx[i]])
            i += 1
    return obj

def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    """
    https://github.com/sczesla/PyAstronomy
    in /src/pyasl/asl/unred
    """

    x = 10000./wave # Convert to inverse microns
    curve = x*0.

    # Set some standard values:
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c2 = -0.824 + 4.717/R_V
    c1 = 2.030 - 3.007*c2

    if LMC2:
        x0    =  4.626
        gamma =  1.05
        c4   =  0.42
        c3    =  1.92
        c2    = 1.31
        c1    =  -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4   =  0.64
        c3    =  2.73
        c2    = 1.11
        c1    =  -1.28

    # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and
    # R-dependent coefficients
    xcutuv = sp.array([10000.0/2700.0])
    xspluv = 10000.0/sp.array([2700.0,2600.0])

    iuv = sp.where(x >= xcutuv)[0]
    N_UV = iuv.size
    iopir = sp.where(x < xcutuv)[0]
    Nopir = iopir.size
    if N_UV>0:
        xuv = sp.concatenate((xspluv,x[iuv]))
    else:
        xuv = xspluv

    yuv = c1 + c2*xuv
    yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
    yuv = yuv + c4*(0.5392*(sp.maximum(xuv,5.9)-5.9)**2+0.05644*(sp.maximum(xuv,5.9)-5.9)**3)
    yuv = yuv + R_V
    yspluv = yuv[0:2]  # save spline points

    if N_UV>0:
        curve[iuv] = yuv[2::] # remove spline points

    # Compute optical portion of A(lambda)/E(B-V) curve
    # using cubic spline anchored in UV, optical, and IR
    xsplopir = sp.concatenate(([0],10000.0/sp.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
    ysplir = sp.array([0.0,0.26469,0.82925])*R_V/3.1
    ysplop = sp.array((sp.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ),
            sp.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ),
            sp.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ),
            sp.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
    ysplopir = sp.concatenate((ysplir,ysplop))

    if Nopir>0:
        tck = interpolate.splrep(sp.concatenate((xsplopir,xspluv)),sp.concatenate((ysplopir,yspluv)),s=0)
        curve[iopir] = interpolate.splev(x[iopir], tck)

    #Now apply extinction correction to input flux vector
    curve *= ebv
    corr = 1./(10.**(0.4*curve))

    return corr
