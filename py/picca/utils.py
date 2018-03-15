import scipy as sp
import sys
import fitsio

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
        print('ERROR: data has some empty bins, impossible to smooth')
        sys.exit()

    cor = co/sp.sqrt(var*var[:,None])

    cor_smooth = sp.zeros([nda,nda])

    dcor={}
    dncor={}

    for i in range(nda):
        sys.stderr.write("\rsmoothing {}".format(i))
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


    sys.stderr.write("\n")
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
    plate = 1+sp.arange(thid.size)
    mjd = 1+sp.arange(thid.size)
    fid = 1+sp.arange(thid.size)

    ### Get RA and DEC from targets
    vac = fitsio.FITS(targets)
    thidTargets = vac[1]["TARGETID"][:]
    raTargets = vac[1]["RA"][:]
    decTargets = vac[1]["DEC"][:]
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

def desi_from_ztarget_to_drq(ztarget,drq,spectype="QSO"):
    '''
    Transform a desi truth.fits file and a
    desi targets.fits into a drq like file

    '''

    vac = fitsio.FITS(ztarget)

    ## Info of the primary observation
    thid  = vac[1]["TARGETID"][:]
    ra    = vac[1]["RA"][:]
    dec   = vac[1]["DEC"][:]
    zqso  = vac[1]["Z"][:]
    plate = 1+sp.arange(thid.size)
    mjd   = 1+sp.arange(thid.size)
    fid   = 1+sp.arange(thid.size)
    sptype = sp.chararray.strip(vac[1]["SPECTYPE"][:].astype(str))

    ## Sanity
    print(" start               : nb object in cat = {}".format(ra.size) )
    w = (vac[1]["ZWARN"][:]==0.)
    print(" and zwarn==0        : nb object in cat = {}".format(ra[w].size) )
    w = w & (sptype==spectype)
    print(" and spectype=={}    : nb object in cat = {}".format(spectype,ra[w].size) )

    ra    = ra[w]
    dec   = dec[w]
    zqso  = zqso[w]
    thid  = thid[w]
    plate = plate[w]
    mjd   = mjd[w]
    fid   = fid[w]

    vac.close()

    ### Save
    out = fitsio.FITS(drq,'rw',clobber=True)
    cols=[ra,dec,thid,plate,mjd,fid,zqso]
    names=['RA','DEC','THING_ID','PLATE','MJD','FIBERID','Z']
    out.write(cols,names=names)
    out.close()

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
