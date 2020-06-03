#!/usr/bin/python3
"""Compute the wick covariance for the auto-correlation of forests

The wick covariance is computed as explained in Delubac et al. 2015
"""
import sys
import fitsio
import argparse
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from multiprocessing import Pool,Lock,cpu_count,Value

from picca import constants, cf, utils, io
from picca.utils import userprint

def calc_wickT(p):
    cf.fill_neighs(p)
    np.random.seed(p[0])
    tmp = cf.compute_wick_terms(p)
    return tmp

def main():
    """Computes the wick covariance for the auto-correlation of forests"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute the wick covariance for the auto-correlation of '
                     'forests'))

    parser.add_argument(
        '--out',
        type=str,
        default=None,
        required=True,
        help='Output file name')

    parser.add_argument(
        '--in-dir',
        type=str,
        default=None,
        required=True,
        help='Directory to delta files')

    parser.add_argument(
        '--in-dir2',
        type=str,
        default=None,
        required=False,
        help='Directory to 2nd delta files')

    parser.add_argument(
        '--rp-min',
        type=float,
        default=0.,
        required=False,
        help='Min r-parallel [h^-1 Mpc]')

    parser.add_argument(
        '--rp-max',
        type=float,
        default=200.,
        required=False,
        help='Max r-parallel [h^-1 Mpc]')

    parser.add_argument(
        '--rt-max',
        type=float,
        default=200.,
        required=False,
        help='Max r-transverse [h^-1 Mpc]')

    parser.add_argument(
        '--np',
        type=int,
        default=50,
        required=False,
        help='Number of r-parallel bins')

    parser.add_argument(
        '--nt',
        type=int,
        default=50,
        required=False,
        help='Number of r-transverse bins')

    parser.add_argument(
        '--z-cut-min',
        type=float,
        default=0.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift larger than '
              'z-cut-min'))

    parser.add_argument(
        '--z-cut-max',
        type=float,
        default=10.,
        required=False,
        help=('Use only pairs of forest x object with the mean of the last '
              'absorber redshift and the object redshift smaller than '
              'z-cut-max'))

    parser.add_argument(
        '--lambda-abs',
        type=str,
        default='LYA',
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the delta'))

    parser.add_argument(
        '--lambda-abs2',
        type=str,
        default=None,
        required=False,
        help=('Name of the absorption in picca.constants defining the redshift '
              'of the 2nd delta'))

    parser.add_argument(
        '--z-ref',
        type=float,
        default=2.25,
        required=False,
        help='Reference redshift')

    parser.add_argument(
        '--z-evol',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the delta field')

    parser.add_argument(
        '--z-evol2',
        type=float,
        default=2.9,
        required=False,
        help='Exponent of the redshift evolution of the 2nd delta field')

    parser.add_argument(
        '--fid-Om',
        type=float,
        default=0.315,
        required=False,
        help='Omega_matter(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-Or',
        type=float,
        default=0.,
        required=False,
        help='Omega_radiation(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-Ok',
        type=float,
        default=0.,
        required=False,
        help='Omega_k(z=0) of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--fid-wl',
        type=float,
        default=-1.,
        required=False,
        help='Equation of state of dark energy of fiducial LambdaCDM cosmology')

    parser.add_argument(
        '--no-project',
        action='store_true',
        required=False,
        help='Do not project out continuum fitting modes')

    parser.add_argument(
        '--max-diagram',
        type=int,
        default=3,
        required=False,
        help='Maximum diagram to compute')

    parser.add_argument(
        '--cf1d',
        type=str,
        required=True,
        help=('1D auto-correlation of pixels from the same forest file: '
              'picca_cf1d.py'))

    parser.add_argument(
        '--cf1d2',
        type=str,
        default=None,
        required=False,
        help=('1D auto-correlation of pixels from the same forest file of the '
              '2nd delta field: picca_cf1d.py'))

    parser.add_argument(
        '--cf',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests: '
              'picca_cf.py'))

    parser.add_argument(
        '--cf2',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests for 2nd '
              'catalog: picca_cf.py'))

    parser.add_argument(
        '--cf12',
        type=str,
        default=None,
        required=False,
        help=('3D auto-correlation of pixels from different forests for cross '
              '1st and 2nd catalog: picca_cf.py'))

    parser.add_argument(
        '--unfold-cf',
        action='store_true',
        required=False,
        help=('rp can be positive or negative depending on the relative '
              'position between absorber1 and absorber2'))

    parser.add_argument(
        '--rej',
        type=float,
        default=1.,
        required=False,
        help='Fraction of rejected pairs: -1=no rejection, 1=all rejection')

    parser.add_argument(
        '--nside',
        type=int,
        default=16,
        required=False,
        help='Healpix nside')

    parser.add_argument(
        '--nproc',
        type=int,
        default=None,
        required=False,
        help='Number of processors')

    parser.add_argument(
        '--nspec',
        type=int,
        default=None,
        required=False,
        help='Maximum number of spectra to read')

    args = parser.parse_args()

    if args.nproc is None:
        args.nproc = cpu_count()//2

    userprint("nproc", args.nproc)

    # setup variables in module cf
    cf.r_par_max = args.rp_max
    cf.r_trans_max = args.rt_max
    cf.r_par_min = args.rp_min
    cf.z_cut_max = args.z_cut_max
    cf.z_cut_min = args.z_cut_min
    cf.num_bins_r_par = args.np
    cf.num_bins_r_trans = args.nt
    cf.nside = args.nside
    cf.z_ref = args.z_ref
    cf.alpha = args.z_evol
    cf.alpha2 = args.z_evol
    cf.lambda_abs = constants.ABSORBER_IGM[args.lambda_abs]
    cf.reject = args.rej
    cf.max_diagram = args.max_diagram

    ### Cosmo
    if (args.fid_Or != 0.) or (args.fid_Ok != 0.) or (args.fid_wl != -1.):
        userprint("ERROR: Cosmology with other than Omega_m set are not yet implemented")
        sys.exit()
    cosmo = constants.Cosmo(Om=args.fid_Om,Or=args.fid_Or,Ok=args.fid_Ok,wl=args.fid_wl)

    ### Read data
    data, num_data, zmin_pix, zmax_pix = io.read_deltas(args.in_dir, cf.nside, cf.lambda_abs, cf.alpha, cf.z_ref, cosmo, max_num_spec=args.nspec)
    for p,datap in data.items():
        for d in datap:
            d.fname = 'D1'
            for k in ['cont','delta','order','ivar','exposures_diff','mean_snr','mean_reso','mean_z','delta_log_lambda']:
                setattr(d,k,None)
    cf.npix = len(data)
    cf.data = data
    cf.num_data = num_data
    cf.ang_max = utils.compute_ang_max(cosmo,cf.r_trans_max,zmin_pix)
    sys.stderr.write("\n")
    userprint("done, npix = {}".format(cf.npix))

    ### Load cf1d
    dic_cf1d = { 'D1':args.cf1d, 'D2':args.cf1d2 }
    for n,p in dic_cf1d.items():
        if p is None:
            continue
        h = fitsio.FITS(p)
        head = h[1].read_header()
        log_lambda_min = head['LLMIN']
        log_lambda_max = head['LLMAX']
        delta_log_lambda = head['DLL']
        num_pairs_variance_1d = h[1]['nv1d'][:]
        variance_1d = h[1]['v1d'][:]
        log_lambda = log_lambda_min + delta_log_lambda*np.arange(len(variance_1d))
        cf.get_variance_1d[n] = interp1d(log_lambda[num_pairs_variance_1d>0],variance_1d[num_pairs_variance_1d>0],kind='nearest',fill_value='extrapolate')

        num_pairs1d = h[1]['nb1d'][:]
        xi_1d = h[1]['c1d'][:]
        cf.xi_1d[n] = interp1d((log_lambda-log_lambda_min)[num_pairs1d>0],xi_1d[num_pairs1d>0],kind='nearest',fill_value='extrapolate')
        h.close()

    ### Load cf
    dic_cf = { 'D1_D1':args.cf, 'D2_D2':args.cf2, 'D1_D2':args.cf12, 'D2_D1':args.cf12 }
    for n,p in dic_cf.items():
        if p is None:
            continue
        h = fitsio.FITS(p)
        head = h[1].read_header()
        assert cf.num_bins_r_par == head['NP']
        assert cf.num_bins_r_trans == head['NT']
        assert cf.r_par_min == head['RPMIN']
        assert cf.r_par_max == head['RPMAX']
        assert cf.r_trans_max == head['RTMAX']
        da = h[2]['DA'][:]
        weights = h[2]['WE'][:]
        da = (da*weights).sum(axis=0)
        weights = weights.sum(axis=0)
        w = weights>0.
        da[w] /= weights[w]
        cf.xi_wick[n] = da.copy()
        h.close()

    ### Read data 2
    if args.in_dir2 or args.lambda_abs2:

        if args.lambda_abs2 or args.unfold_cf:
            cf.x_correlation = True
        cf.alpha2 = args.z_evol2
        if args.in_dir2 is None:
            args.in_dir2 = args.in_dir
        if args.lambda_abs2:
            cf.lambda_abs2 = constants.ABSORBER_IGM[args.lambda_abs2]
        else:
            cf.lambda_abs2 = cf.lambda_abs

        data2, num_data2, zmin_pix2, zmax_pix2 = io.read_deltas(args.in_dir2, cf.nside, cf.lambda_abs2, cf.alpha2, cf.z_ref, cosmo, max_num_spec=args.nspec)
        for p,datap in data2.items():
            for d in datap:
                d.fname = 'D2'
                for k in ['cont','delta','order','ivar','exposures_diff','mean_snr','mean_reso','mean_z','delta_log_lambda']:
                    setattr(d,k,None)
        cf.data2 = data2
        cf.num_data2 = num_data2
        cf.ang_max = utils.compute_ang_max(cosmo,cf.r_trans_max,zmin_pix,zmin_pix2)
        userprint("")
        userprint("done, npix = {}".format(len(data2)))

    cf.counter = Value('i',0)
    cf.lock = Lock()

    cpu_data = {}
    for i,p in enumerate(sorted(data.keys())):
        ip = i%args.nproc
        if not ip in cpu_data:
            cpu_data[ip] = []
        cpu_data[ip].append(p)

    pool = Pool(processes=min(args.nproc,len(cpu_data.values())))
    userprint(" \nStarting\n")
    wickT = pool.map(calc_wickT,sorted(cpu_data.values()))
    userprint(" \nFinished\n")
    pool.close()

    wickT = sp.array(wickT)
    weights_wick = wickT[:,0].sum(axis=0)
    num_pairs_wick = wickT[:,1].sum(axis=0)
    npairs = wickT[:,2].sum(axis=0)
    npairs_used = wickT[:,3].sum(axis=0)
    t1 = wickT[:,4].sum(axis=0)
    t2 = wickT[:,5].sum(axis=0)
    t3 = wickT[:,6].sum(axis=0)
    t4 = wickT[:,7].sum(axis=0)
    t5 = wickT[:,8].sum(axis=0)
    t6 = wickT[:,9].sum(axis=0)
    weights = weights_wick*weights_wick[:,None]
    w = weights>0.
    t1[w] /= weights[w]
    t2[w] /= weights[w]
    t3[w] /= weights[w]
    t4[w] /= weights[w]
    t5[w] /= weights[w]
    t6[w] /= weights[w]
    t1 *= 1.*npairs_used/npairs
    t2 *= 1.*npairs_used/npairs
    t3 *= 1.*npairs_used/npairs
    t4 *= 1.*npairs_used/npairs
    t5 *= 1.*npairs_used/npairs
    t6 *= 1.*npairs_used/npairs
    Ttot = t1+t2+t3+t4+t5+t6

    out = fitsio.FITS(args.out,'rw',clobber=True)
    head = [
        {'name':'RPMIN','value':cf.r_par_min,'comment':'Minimum r-parallel [h^-1 Mpc]'},
        {'name':'RPMAX','value':cf.r_par_max,'comment':'Maximum r-parallel [h^-1 Mpc]'},
        {'name':'RTMAX','value':cf.r_trans_max,'comment':'Maximum r-transverse [h^-1 Mpc]'},
        {'name':'NP','value':cf.num_bins_r_par,'comment':'Number of bins in r-parallel'},
        {'name':'NT','value':cf.num_bins_r_trans,'comment':'Number of bins in r-transverse'},
        {'name':'ZCUTMIN','value':cf.z_cut_min,'comment':'Minimum redshift of pairs'},
        {'name':'ZCUTMAX','value':cf.z_cut_max,'comment':'Maximum redshift of pairs'},
        {'name':'REJ','value':cf.reject,'comment':'Rejection factor'},
        {'name':'NPALL','value':npairs,'comment':'Number of pairs'},
        {'name':'NPUSED','value':npairs_used,'comment':'Number of used pairs'},
    ]
    comment = ['Sum of weight','Covariance','Nomber of pairs','T1','T2','T3','T4','T5','T6']
    out.write([Ttot,weights_wick,num_pairs_wick,t1,t2,t3,t4,t5,t6],names=['CO','WALL','NB','T1','T2','T3','T4','T5','T6'],comment=comment,header=head,extname='COV')
    out.close()


if __name__ == '__main__':
    main()
