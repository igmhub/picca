#!/usr/bin/env python

from __future__ import print_function
import os
import fitsio
import argparse
import scipy as sp
import scipy.linalg

from picca.utils import smooth_cov, print

parser = argparse.ArgumentParser()

parser.add_argument("--data",type=str,nargs="*",required=True,
        help="the (x)cf_z_....fits files to be coadded")

parser.add_argument("--out",type=str,required=True,
        help="output file")

parser.add_argument("--no-dmat",action='store_true',default=False,required=False,
        help='Use an identity matrix as the distortion matrix.')

parser.add_argument('--remove-shuffled-correlation',type=str,default=None,nargs="*",required=False,
        help="the shuffled (x)cf_z_....fits files to be coadded and then removed")

args=parser.parse_args()

for f in args.data:
    if not os.path.isfile(f):
        args.data.remove(f)

# Function to coadd a set of cf of xcf measurements.
def coadd_correlations(fi):
    # Define variables of the correct shape to store correlation information.
    h = fitsio.FITS(fi[0])
    head = h[1].read_header()
    rp = h[1]['RP'][:]*0
    rt = h[1]['RT'][:]*0
    nb = h[1]['NB'][:]*0
    z = h[1]['Z'][:]*0
    wet = rp*0
    h.close()
    da = {}
    we = {}

    # For each data file:
    for f in fi:
        if not (os.path.isfile(f.replace('cf','dmat')) or args.no_dmat):
            continue

        # Add information about the weights and correlation bins.
        h = fitsio.FITS(f)
        we_aux = h[2]["WE"][:]
        wet_aux = we_aux.sum(axis=0)
        rp += h[1]['RP'][:]*wet_aux
        rt += h[1]['RT'][:]*wet_aux
        z  += h[1]['Z'][:]*wet_aux
        nb += h[1]['NB'][:]
        wet += wet_aux

        # Add to the data and weights dictionaries.
        hid = h[2]['HEALPID'][:]
        for i,p in enumerate(hid):
            print("\rcoadding healpix {} in file {}".format(p,f),end="")
            if p in da:
                da[p] += h[2]["DA"][:][i]*we_aux[i]
                we[p] += we_aux[i,:]
            else:
                da[p] = h[2]["DA"][:][i]*we_aux[i]
                we[p] = we_aux[i]
        h.close()


    # Normalise all variables by the total weights.
    for p in da:
        w = we[p]>0
        da[p][w] /= we[p][w]
    rp /= wet
    rt /= wet
    z /= wet

    return rp,rt,nb,z,wet,da,we,dm,head

# Function to coadd a set of cf of xcf distortion matrices.
def coadd_dmats(fi):

    #If there are distortion matrices, set up variables of the correct shapes.
    h = fitsio.FITS(fi[0].replace('cf','dmat'))
    dm = h[1]['DM'][:]*0
    try:
        dmrp = sp.zeros(h[2]['RP'][:].size)
        dmrt = sp.zeros(h[2]['RT'][:].size)
        dmz = sp.zeros(h[2]['Z'][:].size)
        nbdm = 0.
    except IOError:
        pass
    h.close()

    for f in fi:
        if not (os.path.isfile(f.replace('cf','dmat')) or args.no_dmat):
            continue
        # If there are distortion matrices, add in contributions.
        h = fitsio.FITS(f.replace('cf','dmat'))
        dm += h[1]['DM'][:]*wet_aux[:,None]
        if 'dmrp' in locals():
            ## TODO: get the weights
            dmrp += h[2]['RP'][:]
            dmrt += h[2]['RT'][:]
            dmz += h[2]['Z'][:]
            nbdm += 1.
        h.close()

    # Normalise all variables by the total weights.
    dm /= wet[:,None]
    if 'dmrp' in locals():
        dmrp /= nbdm
        dmrt /= nbdm
        dmz /= nbdm

    return dmrp,dmrt,dmz,nbdm,dm

# Coadd the data files.
rp,rt,nb,z,wet,da,we,head = coadd_correlations(args.data)

# If required, coadd the distortion matrices.
if not args.no_dmat:
    dmrp,dmrt,dmz,nbdm,dm = coadd_dmats(args.data)
else:
    dm = sp.eye(nb.shape[0])

# If required, remove the shuffled correlations.
if not args.remove_shuffled_correlation is None:
    rp_s,rt_s,nb_s,z_s,wet_s,da_s,we_s,dm_s,head_s = coadd_correlations(args.remove_shuffled_correlation)
    da_s = (da_s*we_s).sum(axis=1)
    we_s = we_s.sum(axis=1)
    w = we_s>0.
    da_s[w] /= we_s[w]
    da -= da_s[:,None]

# Average data and weights over all HEALPix pixels.
da = sp.vstack(list(da.values()))
we = sp.vstack(list(we.values()))
da = (da*we).sum(axis=0)
da /= wet

# Calculate the smoothed covariance from subsampling.
co = smooth_cov(da,we,rp,rt)

# Not sure what this does.
if ('dmrp' not in locals()) or (dmrp.size==rp.size):
    dmrp = rp.copy()
    dmrt = rt.copy()
    dmz = z.copy()

# Check to see if the covariance is positive definite.
try:
    scipy.linalg.cholesky(co)
except scipy.linalg.LinAlgError:
    print('WARNING: Matrix is not positive definite')

# Write the exported correlation file.
h = fitsio.FITS(args.out,"rw",clobber=True)
comment = ['R-parallel','R-transverse','Redshift','Correlation','Covariance matrix','Distortion matrix','Number of pairs']
h.write([rp,rt,z,da,co,dm,nb],names=['RP','RT','Z','DA','CO','DM','NB'],comment=comment,header=head,extname='COR')
comment = ['R-parallel model','R-transverse model','Redshift model']
h.write([dmrp,dmrt,dmz],names=['DMRP','DMRT','DMZ'],comment=comment,extname='DMATTRI')
h.close()
