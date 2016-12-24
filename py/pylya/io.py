import fitsio
import scipy as sp
import healpy
import glob
import sys

from pylya.data import forest

def read_dlas(fdla):
    f=open(fdla)
    dlas={}
    for l in f:
        l = l.split()
        if len(l)==0:continue
        if l[0][0]=="#":continue
        if l[0]=="ThingID":continue
        if l[0][0]=="-":continue
        thid = int(l[0])
        if not dlas.has_key(thid):
            dlas[int(l[0])]=[]
        zabs = float(l[9])
        nhi = float(l[10])
        dlas[thid].append((zabs,nhi))

    return dlas

def read_drq(drq,zmin,zmax,keep_bal):
    vac = fitsio.FITS(drq)
    zqso = vac[1]["Z_VI"][:] 
    thid = vac[1]["THING_ID"][:]
    ra = vac[1]["RA"][:]
    dec = vac[1]["DEC"][:]
    bal_flag = vac[1]["BAL_FLAG_VI"][:]

    ## info of the primary observation
    plate = vac[1]["PLATE"][:]
    mjd = vac[1]["MJD"][:]
    fid = vac[1]["FIBERID"][:]
    ## sanity
    w = thid>0
    w = w &  (zqso > zmin) & (zqso < zmax)
    if not keep_bal:
        w = w & (bal_flag == 0)

    ra = ra[w] * sp.pi / 180
    dec = dec[w] * sp.pi / 180
    zqso = zqso[w]
    thid = thid[w]
    plate = plate[w]
    mjd = mjd[w]
    fid = fid[w]
    vac.close()
    return ra,dec,zqso,thid,plate,mjd,fid

target_mobj = 500
nside_min = 8
def read_data(in_dir,drq,mode,zmin = 2.1,zmax = 3.5,nspec=None,log=None,keep_bal=False):
    ra,dec,zqso,thid,plate,mjd,fid = read_drq(drq,zmin,zmax,keep_bal)

    if mode == "pix":
        ## hardcoded for now, need to coordinate with Jose how to get this info
        nside = 64
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
    elif mode == "spec" or mode =="corrected-spec":
        nside = 256
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
        mobj = sp.bincount(pixs).sum()/len(sp.unique(pixs))

        ## determine nside such that there are 1000 objs per pixel on average
        print("determining nside")
        while mobj<target_mobj and nside >= nside_min:
            nside /= 2
            pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
            mobj = sp.bincount(pixs).sum()/len(sp.unique(pixs))
        print("nside = {} -- mean #obj per pixel = {}".format(nside,mobj))

    data ={}
    ndata = 0

    ## minimum number of unmasked forest pixels after rebinning

    upix = sp.unique(pixs)
    for i, pix in enumerate(upix):
        w = pixs == pix
        ## read all hiz qsos
        if mode == "pix":
            pix_data = read_from_pix(in_dir,pix,thid[w], ra[w], dec[w], zqso[w], plate[w], mjd[w], fid[w],log=log)
        elif mode == "spec" or mode =="corrected-spec":
            pix_data = read_from_spec(in_dir,thid[w], ra[w], dec[w], zqso[w], plate[w], mjd[w], fid[w],mode=mode,log=log)
        if not pix_data is None:
            sys.stderr.write("\r{} read from pix {}, {} {}".format(len(pix_data),pix,i,len(upix)))
        if not pix_data is None and len(pix_data)>0:
            data[pix] = pix_data
            ndata += len(pix_data)

        if not nspec is None:
            if ndata > nspec:break
	
 #       sys.stderr.write("\rread {} pixels of {}".format(i,len(sp.unique(pixs[s]))))
    print ""

    return data,ndata

def read_from_spec(in_dir,thid,ra,dec,zqso,plate,mjd,fid,mode,log=None):
    pix_data = []
    for t,r,d,z,p,m,f in zip(thid,ra,dec,zqso,plate,mjd,fid):
        try:
            fid = str(f)
            if f<10:
                fid='0'+fid
            if f<100:
                fid = '0'+fid
            if f<1000:
                fid = '0'+fid
            fin = in_dir + "/{}/{}-{}-{}-{}.fits".format(p,mode,p,m,fid)
            h = fitsio.FITS(fin)
        except IOError:
            log.write("error reading {}\n".format(fin))
            continue

        log.write("{} read\n".format(fin))
        d = forest(h[1], t, r, d, z, p, m, f,mode="spec")
        pix_data.append(d)
        h.close()
    return pix_data

def read_from_pix(in_dir,pix,thid,ra,dec,zqso,plate,mjd,fid,log=None):
        try:
            fin = in_dir + "/pix_{}.fits.gz".format(pix)
	    h = fitsio.FITS(fin)
        except IOError:
            print "error reading {}".format(p)
            return None

        pix_data=[]
        for (t, r, d, z, p, m, f) in zip(thid, ra, dec, zqso, plate, mjd, fid):
            if not str(t) in h:
                log.write("{} not found in file {}\n".format(t,fin))
                continue
        
            d = forest(h[str(t)], t, r, d, z, p, m, f)

            log.write("{} read\n".format(t))
            pix_data.append(d)
        h.close()
        return pix_data
