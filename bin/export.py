import fitsio
import scipy as sp

import argparse
from pylya.utils import smooth_cov


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type = str, default = None, required=True,
                        help = 'data file')

    parser.add_argument('--out', type = str, default = None, required=True,
                        help = 'output file')

    parser.add_argument('--dmat', type = str, default = None, required=False,
                        help = 'distorsion matrix file')

    parser.add_argument('--cov', type = str, default = None, required=False,
                        help = 'covariance matrix file (if not provided it will be calculated by subsampling)')

    
    args = parser.parse_args()

    h = fitsio.FITS(args.data)

    rp = sp.array(h[1]['RP'][:])
    rt = sp.array(h[1]['RT'][:])
    z  = sp.array(h[1]['Z'][:])
    da = sp.array(h[2]['DA'][:])
    we = sp.array(h[2]['WE'][:])
    co = smooth_cov(da,we,rp,rt)
    da = (da*we).sum(axis=0)/we.sum(axis=0)

    h.close()

    if args.dmat is not None:
        h = fitsio.FITS(args.dmat)
        dm = h[1]['DM'][:]
    else:
        dm = sp.eye(len(da))

    h.close()

    h = fitsio.FITS(args.out,'rw',clobber=True)

    h.write([rp,rt,z,da,co,dm],names=['RP','RT','Z','DA','CO','DM'])
    h.close()
    



