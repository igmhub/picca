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

parser.add_argument("--coadd-out",type=str,default=None,required=False,
        help="coadded (not exported) output file")

parser.add_argument("--no-dmat",action='store_true',default=False,required=False,
        help='Use an identity matrix as the distortion matrix.')

parser.add_argument('--remove-shuffled-correlation',type=str,default=None,nargs="*",required=False,
        help="the shuffled (x)cf_z_....fits files to be coadded and then removed")

parser.add_argument("--coadd-out-shuffled",type=str,default=None,required=False,
        help="coadded (not exported) shuffled output file")

args=parser.parse_args()

for f in args.data:
    if not os.path.isfile(f):
        args.data.remove(f)

# Function to coadd a set of cf of xcf measurements.
def coadd_correlations(fi,fout=None):
    # Define variables of the correct shape to store correlation information.
    h = fitsio.FITS(fi[0])
    head = h[1].read_header()
    rp = h[1]['RP'][:]*0
    rt = h[1]['RT'][:]*0
    nb = h[1]['NB'][:]*0
    z = h[1]['Z'][:]*0
    hid = h[2]['HEALPID'][:]
    wet = rp*0
    da = np.zeros(h[2]['DA'].shape)
    we = np.zeros(h[2]['WE'].shape)
    h.close()
    """
    ## OLD METHOD USING A DICTIONARY FOR DA AND WE
    da = {}
    we = {}
    """

    # For each data file:
    for f in fi:
        print("coadding file {}".format(f),end="\r")

        # Add information about the weights and correlation bins.
        h = fitsio.FITS(f)
        we_aux = h[2]['WE'][:]
        wet_aux = we_aux.sum(axis=0)
        rp += h[1]['RP'][:]*wet_aux
        rt += h[1]['RT'][:]*wet_aux
        z  += h[1]['Z'][:]*wet_aux
        nb += h[1]['NB'][:]
        wet += wet_aux

        """
        ## OLD METHOD USING A DICTIONARY FOR DA AND WE
        # Add to the data and weights dictionaries.
        f_hid = h[2]['HEALPID'][:]
        #hid = sp.array(list(range(10)))
        for i,p in enumerate(hid):
            print("coadding healpix {} in file {}".format(p,f),end="\r")
            if p in da:
                da[p] += h[2]["DA"][:][i]*we_aux[i]
                we[p] += we_aux[i,:]
            else:
                da[p] = h[2]["DA"][:][i]*we_aux[i]
                we[p] = we_aux[i]
        """

        #Check that the HEALPix pixels are the same.
        if f_hid == hid:
            da += h[2]["DA"][:] * we_aux
            we += h[2]['WE'][:]
        elif set(f_hid) == set(hid):
            # TODO: Add in check to see if they're the same but just ordered differently.
            raise IOError('Correlations\' pixels are not ordered in the same way!')
        else:
            raise IOError('Correlations do not have the same footprint!')

        h.close()
        print('')

    # Normalise all variables by the total weights.
    """
    ## OLD METHOD USING A DICTIONARY FOR DA AND WE
    for p in da:
        w = we[p]>0
        da[p][w] /= we[p][w]
    """
    w = we>0
    da[w] /= we[w]
    rp /= wet
    rt /= wet
    z /= wet

    if fout is not None:

        """
        ## OLD METHOD USING A DICTIONARY FOR DA AND WE
        hid = sp.array(list(da.keys()))
        da_arr = sp.vstack(list(da.values()))
        we_arr = sp.vstack(list(we.values()))
        """
        da_arr = da
        we_arr = we

        out = fitsio.FITS(fout,'rw',clobber=True)
        out.write([rp,rt,z,nb],names=['RP','RT','Z','NB'],
            comment=['R-parallel','R-transverse','Redshift','Number of pairs'],
            units=['h^-1 Mpc','h^-1 Mpc','',''],
            header=head,extname='ATTRI')

        head2 = [{'name':'HLPXSCHM','value':'RING','comment':'Healpix scheme'}]
        out.write([hid,we_arr,da_arr],names=['HEALPID','WE','DA'],
            comment=['Healpix index', 'Sum of weight', 'Correlation'],
            header=head2,extname='COR')

        out.close()

    return rp,rt,nb,z,wet,da,we,head

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
rp,rt,nb,z,wet,da,we,head = coadd_correlations(args.data,args.coadd_out)

# If required, coadd the distortion matrices.
if not args.no_dmat:
    dmrp,dmrt,dmz,nbdm,dm = coadd_dmats(args.data)
else:
    dm = sp.eye(nb.shape[0])

"""
# Stack data and weights arrays from all zbins.
da = sp.vstack(list(da.values()))
we = sp.vstack(list(we.values()))
"""

# If required, remove the shuffled correlations.
if not args.remove_shuffled_correlation is None:
    rp_s,rt_s,nb_s,z_s,wet_s,da_s,we_s,head_s = coadd_correlations(args.remove_shuffled_correlation,args.coadd_out_shuffled)
    print('')
    """
    ## OLD METHOD USING A DICTIONARY FOR DA AND WE
    pix = list(da.keys())
    for p in pix:
        print('Removing shuffled correlation from HEALPix pixel {}'.format(p),end='\r')
        da_s_p = da_s[p]
        we_s_p = we_s[p]
        da_s_p = (da_s_p*we_s_p).sum()
        we_s_p = (we_s_p).sum()
        if we_s_p>0.:
            da_s_p /= we_s_p
            da[p] -= da_s_p
    """
    print('Removing shuffled correlation...')
    da_s = (da_s*we_s).sum(axis=1)
    we_s = we_s.sum(axis=1)
    w = we_s>0
    da_s[w] /= we_s[w]
    da -= da_s[:,None]
    print('')

"""
## OLD METHOD USING A DICTIONARY FOR DA AND WE
# Stack data and weights arrays from all zbins.
da = sp.vstack(list(da.values()))
we = sp.vstack(list(we.values()))
"""

# Calculate the smoothed covariance from subsampling.
co = smooth_cov(da,we,rp,rt)

# Average over all z bins and HEAlPix pixels.
da = (da*we).sum(axis=0)
da /= wet

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
