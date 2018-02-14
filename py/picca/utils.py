import scipy as sp
import sys
import fitsio

def cov(da,we):

    npix = da.shape[0]
    nda = da.shape[1]
    co = sp.zeros([nda,nda])

    mda = (da*we).sum(axis=0)/we.sum(axis=0)

    wda = we*(da-mda)

    print("Computing cov...")
    '''
    for ipix in range(npix):
        sys.stderr.write("\r {} {}".format(ipix,npix))
        co += sp.outer(wda[ipix,:],wda[ipix,:])
    '''
    co = wda.T.dot(wda)
    swe = we.sum(axis=0)

    co/=swe*swe[:,None]

    return co
def smooth_cov(da,we,rp,rt,drt=4,drp=4):
    
    co = cov(da,we)

    nda = da.shape[1]
    var = sp.diagonal(co)

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

    vac = fitsio.FITS(truth)
    vacTargets = fitsio.FITS(targets)

    ## Info of the primary observation
    thid  = vac[1]["TARGETID"][:]
    ra    = vacTargets[1]["RA"][:]
    dec   = vacTargets[1]["DEC"][:]
    zqso  = vac[1]["TRUEZ"][:]
    plate = 1+sp.arange(thid.size)
    mjd   = 1+sp.arange(thid.size)
    fid   = 1+sp.arange(thid.size)
    sptype = sp.chararray.strip(vac[1]["TRUESPECTYPE"][:].astype(str))

    ## Sanity
    print(" start               : nb object in cat = {}".format(ra.size) )
    w = (sptype==spectype)
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
