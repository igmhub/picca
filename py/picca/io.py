import fitsio
import scipy as sp
import healpy
import glob
import sys
import time

from picca.data import forest
from picca.data import delta
from picca.data import qso

def read_dlas(fdla):
    f=open(fdla)
    dlas={}
    nb_dla = 0
    col_names=None
    for l in f:
        l = l.split()
        if len(l)==0:continue
        if l[0][0]=="#":continue
        if l[0]=="ThingID":
            col_names = l
            continue
        if l[0][0]=="-":continue
        thid = int(l[col_names.index("ThingID")])
        if not dlas.has_key(thid):
            dlas[thid]=[]
        zabs = float(l[col_names.index("z_abs")])
        nhi = float(l[col_names.index("NHI")])
        dlas[thid].append((zabs,nhi))
        nb_dla += 1

    print("")
    print(" In catalog: {} DLAs".format(nb_dla) )
    print(" In catalog: {} forests have a DLA".format(len(dlas)) )
    print("")

    return dlas

def read_drq(drq,zmin,zmax,keep_bal,bi_max=None):
    vac = fitsio.FITS(drq)

    ## Redshift
    try:
        zqso = vac[1]["Z"][:] 
    except:
        sys.stderr.write("Z not found (new DRQ >= DRQ14 style), using Z_VI (DRQ <= DRQ12)\n")
        zqso = vac[1]["Z_VI"][:] 

    ## Info of the primary observation
    thid  = vac[1]["THING_ID"][:]
    ra    = vac[1]["RA"][:]
    dec   = vac[1]["DEC"][:]
    plate = vac[1]["PLATE"][:]
    mjd   = vac[1]["MJD"][:]
    fid   = vac[1]["FIBERID"][:]

    print
    ## Sanity
    print(" start               : nb object in cat = {}".format(ra.size) )
    w = (thid>0)
    print(" and thid>0          : nb object in cat = {}".format(ra[w].size) )
    w = w & (ra!=dec)
    print(" and ra!=dec         : nb object in cat = {}".format(ra[w].size) )
    w = w & (ra!=0.)
    print(" and ra!=0.          : nb object in cat = {}".format(ra[w].size) )
    w = w & (dec!=0.)
    print(" and dec!=0.         : nb object in cat = {}".format(ra[w].size) )
    w = w & (zqso>0.)
    print(" and z>0.            : nb object in cat = {}".format(ra[w].size) )

    ## Redshift range
    w = w & (zqso>zmin)
    print(" and z>zmin          : nb object in cat = {}".format(ra[w].size) )
    w = w & (zqso<zmax)
    print(" and z<zmax          : nb object in cat = {}".format(ra[w].size) )

    ## BAL visual
    if not keep_bal and bi_max==None:
        try:
            bal_flag = vac[1]["BAL_FLAG_VI"][:]
            w = w & (bal_flag==0)
            print(" and BAL_FLAG_VI == 0  : nb object in cat = {}".format(ra[w].size) )
        except:
            sys.stderr.write("BAL_FLAG_VI not found\n")
    ## BAL CIV
    if bi_max is not None:
        try:
            bi = vac[1]["BI_CIV"][:]
            w = w & (bi<=bi_max)
            print(" and BI_CIV<=bi_max  : nb object in cat = {}".format(ra[w].size) )
        except:
            sys.stderr.write("--bi-max set but no BI_CIV field in vac")
            sys.exit(1)
    print

    ra    = ra[w]*sp.pi/180.
    dec   = dec[w]*sp.pi/180.
    zqso  = zqso[w]
    thid  = thid[w]
    plate = plate[w]
    mjd   = mjd[w]
    fid   = fid[w]
    vac.close()

    return ra,dec,zqso,thid,plate,mjd,fid

target_mobj = 500
nside_min = 8
def read_data(in_dir,drq,mode,zmin = 2.1,zmax = 3.5,nspec=None,log=None,keep_bal=False,bi_max=None,order=1):

    if mode != "desi":
        sys.stderr.write("mode: "+mode)
        ra,dec,zqso,thid,plate,mjd,fid = read_drq(drq,zmin,zmax,keep_bal,bi_max=bi_max)

    if mode == "pix":
        try:
            fin = in_dir + "/master.fits.gz"
            h = fitsio.FITS(fin)
        except IOError:
            try:
                fin = in_dir + "/master.fits"
                h = fitsio.FITS(fin)
            except IOError:
                try:
                    fin = in_dir + "/../master.fits"
                    h = fitsio.FITS(fin)
                except:
                    print "error reading master"
                    sys.exit(1)
        nside = h[1].read_header()['NSIDE']
        h.close()
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
    elif mode in ["spec","corrected-spec","spcframe"]:
        nside = 256
        pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
        mobj = sp.bincount(pixs).sum()/len(sp.unique(pixs))

        ## determine nside such that there are 1000 objs per pixel on average
        sys.stderr.write("determining nside\n")
        while mobj<target_mobj and nside >= nside_min:
            nside /= 2
            pixs = healpy.ang2pix(nside, sp.pi / 2 - dec, ra)
            mobj = sp.bincount(pixs).sum()/len(sp.unique(pixs))
        sys.stderr.write("nside = {} -- mean #obj per pixel = {}\n".format(nside,mobj))
        if log is not None:
            log.write("nside = {} -- mean #obj per pixel = {}\n".format(nside,mobj))

    elif mode=="desi":
        sys.stderr.write("reading truth table desi style\n")
        nside = 8
        h = fitsio.FITS(drq)
        zs = h[1]["TRUEZ"][:]
        tid = h[1]["TARGETID"][:]
        typ = h[1]["TRUESPECTYPE"][:]
        w = ["QSO" in t for t in typ]
        w = sp.array(w)
        ztable = {t:z for t,z in zip(tid[w],zs[w]) if z > zmin and z<zmax}
        sys.stderr.write("Found {} qsos\n".format(len(ztable)))
        return read_from_desi(nside,ztable,in_dir)

    else:
        sys.stderr.write("I don't know mode: {}".format(mode))
        sys.exit(1)

    data ={}
    ndata = 0

    upix = sp.unique(pixs)
    for i, pix in enumerate(upix):
        w = pixs == pix
        ## read all hiz qsos
        if mode == "pix":
            t0 = time.time()
            pix_data = read_from_pix(in_dir,pix,thid[w], ra[w], dec[w], zqso[w], plate[w], mjd[w], fid[w], order, log=log)
            read_time=time.time()-t0
        elif mode == "spec" or mode =="corrected-spec":
            t0 = time.time()
            pix_data = read_from_spec(in_dir,thid[w], ra[w], dec[w], zqso[w], plate[w], mjd[w], fid[w], order, mode=mode,log=log)
            read_time=time.time()-t0
        elif mode =="spcframe":
            t0 = time.time()
            pix_data = read_from_spcframe(in_dir,thid[w], ra[w], dec[w], zqso[w], plate[w], mjd[w], fid[w], order, mode=mode, log=log)
            read_time=time.time()-t0
        if not pix_data is None:
            sys.stderr.write("{} read from pix {}, {} {} in {} secs per spectrum\n".format(len(pix_data),pix,i,len(upix),read_time/(len(pix_data)+1e-3)))
        if not pix_data is None and len(pix_data)>0:
            data[pix] = pix_data
            ndata += len(pix_data)

        if not nspec is None:
            if ndata > nspec:break

    return data,ndata

def read_from_spec(in_dir,thid,ra,dec,zqso,plate,mjd,fid,order,mode,log=None):
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
        ll = h[1]["loglam"][:]
        fl = h[1]["flux"][:]
        iv = h[1]["ivar"][:]*(h[1]["and_mask"][:]==0)
        d = forest(ll,fl,iv, t, r, d, z, p, m, f,order)
        pix_data.append(d)
        h.close()
    return pix_data

def read_from_pix(in_dir,pix,thid,ra,dec,zqso,plate,mjd,fid,order,log=None):
        try:
            fin = in_dir + "/pix_{}.fits.gz".format(pix)
            h = fitsio.FITS(fin)
        except IOError:
            try:
                fin = in_dir + "/pix_{}.fits".format(pix)
                h = fitsio.FITS(fin)
            except IOError:
                print "error reading {}".format(pix)
                return None

        ## fill log
        if log is not None:
            for t in thid:
                if t not in h[0][:]:
                    log.write("{} missing from pixel {}\n".format(t,pix))
                    sys.stderr.write("{} missing from pixel {}\n".format(t,pix))

        pix_data=[]
        thid_list=list(h[0][:])
        thid2idx = {t:thid_list.index(t) for t in thid if t in thid_list}
        loglam  = h[1][:]
        flux = h[2].read()
        ivar = h[3].read()
        andmask = h[4].read()
        for (t, r, d, z, p, m, f) in zip(thid, ra, dec, zqso, plate, mjd, fid):
            try:
                idx = thid2idx[t]
            except:
                ## fill log
                if log is not None:
                    log.write("{} missing from pixel {}\n".format(t,pix))
                sys.stderr.write("{} missing from pixel {}\n".format(t,pix))
                continue
            d = forest(loglam,flux[:,idx],ivar[:,idx]*(andmask[:,idx]==0), t, r, d, z, p, m, f,order)

            log.write("{} read\n".format(t))
            pix_data.append(d)
        h.close()
        return pix_data

def read_from_spcframe(in_dir,thid,ra,dec,zqso,plate,mjd,fid,order,mode=None,log=None):
    pix_data={}
    plates = sp.unique(plate)

    prefix='spCFrame'
    sufix=''

    for p in plates:

        name = in_dir+"/{}/{}-*-*.fits{}".format(p,prefix,sufix)
        fi = glob.glob(name)
        w = (plate==p)

        if len(fi)==0:
            sys.stderr.write("No files found in {}".format(name))
            continue

        for the_file in fi:

            sys.stderr.write("reading {}\n".format(the_file))
            h = fitsio.FITS(the_file)
            fib_list=list(h[5]["FIBERID"][:])
            flux = h[0].read()
            ivar = h[1].read()*(h[2].read()==0)
            llam = h[3].read()

            if "-r1-" in the_file or "-b1-" in the_file:
                ww = w & (fid<=500)
            elif "-r2-" in the_file or "-b2-" in the_file:
                ww = w & (fid>=501)

            for (t, r, d, z, p, m, f) in zip(thid[ww], ra[ww], dec[ww], zqso[ww], plate[ww], mjd[ww], fid[ww]):

                index = fib_list.index(f)
                d = forest(llam[index],flux[index],ivar[index], t, r, d, z, p, m, f, order)
                if t in pix_data:
                    pix_data[t] += d
                else:
                    pix_data[t] = d
                log.write("{} read\n".format(t))

            h.close()

    return pix_data.values()

def read_from_desi(nside,ztable,in_dir,order):
    fi = glob.glob(in_dir+"/spectra-*.fits")
    data = {}
    ndata=0
    for i,f in enumerate(fi):
        sys.stderr.write("\rread {} of {}. ndata: {}".format(i,len(fi),ndata))
        try:
            h = fitsio.FITS(f)
        except IOError:
            sys.stderr.write("Error reading {}\n".format(f))
            continue
        ## get the quasars
        w = sp.in1d(h[1]["TARGETID"][:],ztable.keys())
        tid_qsos = h[1]["TARGETID"][:][w]
        b_ll = sp.log10(h["B_WAVELENGTH"].read())
        b_iv  = h["B_IVAR"].read()*(h["B_MASK"].read()==0)
        b_fl  = h["B_FLUX"].read()
        ra = h["FIBERMAP"]["RA_TARGET"][:]*sp.pi/180.
        de = h["FIBERMAP"]["DEC_TARGET"][:]*sp.pi/180.
        pixs = healpy.ang2pix(nside, sp.pi / 2 - de, ra)

        exp = h["FIBERMAP"]["EXPID"][:]
        night = h["FIBERMAP"]["NIGHT"][:]
        fib = h["FIBERMAP"]["FIBER"][:]
        for t in tid_qsos:
            wt = h[1]["TARGETID"][:] == t
            iv = b_iv[wt]
            fl = (iv*b_fl[wt]).sum(axis=0)
            iv = iv.sum(axis=0)
            w = iv>0
            fl[w]/=iv[w]
            d = forest(b_ll,fl,iv,t,ra[wt][0],de[wt][0],ztable[t],exp[wt][0],night[wt][0],fib[wt][0],order)
            pix = pixs[wt][0]
            if pix not in data:
                data[pix]=[]
            data[pix].append(d)
            ndata+=1

    sys.stderr.write("found {} quasars in input files\n".format(ndata))
    return data,ndata


def read_deltas(indir,nside,lambda_abs,alpha,zref,cosmo,nspec=None):
    '''
    reads deltas from indir
    fills the fields delta.z and multiplies the weights by (1+z)^(alpha-1)/(1+zref)^(alpha-1)
    returns data,zmin_pix
    '''
    dels = []
    fi = glob.glob(indir+"/*.fits.gz")
    ndata=0
    for i,f in enumerate(fi):
        sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
        hdus = fitsio.FITS(f)
        dels += [delta.from_fitsio(h) for h in hdus[1:]]
        ndata+=len(hdus[1:])
        hdus.close()
        if not nspec is None:
            if ndata>nspec:break

    sys.stderr.write("\n")

    phi = [d.ra for d in dels]
    th = [sp.pi/2.-d.dec for d in dels]
    pix = healpy.ang2pix(nside,th,phi)

    data = {}
    zmin = 10**dels[0].ll[0]/lambda_abs-1.
    zmax = 0.
    for d,p in zip(dels,pix):
        if not p in data:
            data[p]=[]
        data[p].append(d)

        z = 10**d.ll/lambda_abs-1.
        min_z = z.min()
        max_z = z.max()
        if zmin>min_z:
            zmin = min_z
        if zmax < max_z:
            zmax = max_z
        d.z = z
        d.r_comov = cosmo.r_comoving(z)
        d.we *= ((1+z)/(1+zref))**(alpha-1)

    return data,ndata,zmin,zmax


def read_objects(drq,nside,zmin,zmax,alpha,zref,cosmo,keep_bal=True):
    objs = {}
    ra,dec,zqso,thid,plate,mjd,fid = read_drq(drq,zmin,zmax,keep_bal=True)
    phi = ra
    th = sp.pi/2.-dec
    pix = healpy.ang2pix(nside,th,phi)
    print("reading qsos")

    upix = sp.unique(pix)
    for i,ipix in enumerate(upix):
        sys.stderr.write("\r{} of {}".format(i,len(upix)))
        w=pix==ipix
        objs[ipix] = [qso(t,r,d,z,p,m,f) for t,r,d,z,p,m,f in zip(thid[w],ra[w],dec[w],zqso[w],plate[w],mjd[w],fid[w])]
        for q in objs[ipix]:
            q.we = ((1.+q.zqso)/(1.+zref))**(alpha-1.)
            q.r_comov = cosmo.r_comoving(q.zqso)

    sys.stderr.write("\n")

    return objs,zqso.min()
