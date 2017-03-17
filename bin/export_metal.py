import fitsio
import scipy as sp

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, default=None, required=True,
                        help = 'data file')

    parser.add_argument('--out', type=str, default=None, required=True,
                        help = 'output folder')

    parser.add_argument('--line1-name', type=str, default=None, required=True,
                        help = 'line1_name or QSO')

    parser.add_argument('--line2-name', type=str, default=None, required=True,
                        help = 'line2_name')

    args = parser.parse_args()

    ### Grid for cross-correlation
    if (args.line1_name=='QSO'):

        h  = fitsio.FITS(args.data)
        rp = sp.array(h[1]['RP'][:])
        rt = sp.array(h[1]['RT'][:])
        z  = sp.array(h[1]['Z'][:])
        h.close()

        array_to_save = sp.asarray(zip(numpy.arange(rp.size),rp,rt,zz))
        sp.savetxt(args.out+'/metTemp_'+args.line1_name+'_'+args.line2_name+'.grid',array_to_save)

    ### Needs to be written for auto-correlation
    



