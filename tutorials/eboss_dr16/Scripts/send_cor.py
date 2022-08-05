import scipy as sp
import scipy.linalg
import argparse
import subprocess
import fitsio
import os
import h5py
import glob
import time
import matplotlib.pyplot as plt

from picca.constants import ABSORBER_IGM


path_here = os.environ['DR16_BASE']
path_drq = os.environ['QSO_CAT']
path_deltas = os.environ['DR16_BASE']

metList = {}
metList['LYA'] = ['CIV(eff)','SiII(1260)','SiIII(1207)','SiII(1193)','SiII(1190)']
metList['LYB'] = ['CIV(eff)','SiII(1260)','SiIII(1207)','SiII(1193)','SiII(1190)']


def send_xcf(zmin,zmax,do_corr,do_dist,do_met,f='LYA',l='LYA'):

    if (zmin==0.) and (zmax==10.):
        zmin = int(zmin)
        zmax = int(zmax)
        in_dir = path_deltas+'/Delta_{}/Delta/'.format(f)
    else:
        if False:
            in_dir = path_deltas+'/Delta_{}_z_{}_{}/Delta/'.format(f,zmin,zmax)
        else:
            print('\nNot use /Delta_{}_z_{}_{}/Delta/ \n'.format(f,zmin,zmax))
            in_dir = path_deltas+'/Delta_{}/Delta/'.format(f,zmin,zmax)
    strl = l.replace('(','').replace(')','')

    cmd = 'picca_xcf.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_drq
    cmd += ' --out {}/Correlations/xcf_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    cmd += ' --z-evol-obj 1.44 '
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    if l!='LYA':
        cmd += ' --lambda-abs '+l.replace('(','\(').replace(')','\)')

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('xcf_','xcf_{}_in_{}_'.format(strl,f))
    print('')
    
    print(cmd)
    if do_corr: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_xcf = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_xdmat.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_drq
    cmd += ' --out {}/Correlations/xdmat_z_{}_{}.fits.gz'.format(path_here, zmin, zmax)
    cmd += ' --z-evol-obj 1.44 '
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.99'
    if l!='LYA':
        cmd += ' --lambda-abs '+l.replace('(','\(').replace(')','\)')

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('xdmat_','xdmat_{}_in_{}_'.format(strl,f))
    
    print('')
    print(cmd)
    if do_dist: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_xdmat = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_export.py'
    cmd += ' --data {}/Correlations/xcf_z_{}_{}.fits.gz'.format(path_here, zmin, zmax)
    cmd += ' --dmat {}/Correlations/xdmat_z_{}_{}.fits.gz'.format(path_here, zmin, zmax)
    cmd += ' --out {}/Correlations/xcf_z_{}_{}-exp.fits.gz'.format(path_here, zmin, zmax)

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('xcf_','xcf_{}_in_{}_'.format(strl,f))
        cmd = cmd.replace('xdmat_','xdmat_{}_in_{}_'.format(strl,f))
    
    print('')
    print(cmd)
    if do_dist: subprocess.call(cmd, shell=True)

    cmd = 'picca_metal_xdmat.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_drq
    cmd += ' --out {}/Correlations/metal_xdmat_z_{}_{}.fits.gz'.format(path_here, zmin, zmax)
    cmd += ' --z-evol-obj 1.44 '
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.999'
    cmd += ' --abs-igm '
    for m in metList[f]:
        cmd += m+' '
    if l!='LYA':
        cmd += ' --lambda-abs '+l#.replace('(','\(').replace(')','\)')

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('metal_xdmat_','metal_xdmat_{}_in_{}_'.format(strl,f))
    cmd = cmd.replace('(','\(').replace(')','\)')
    print('')
    print(cmd)
    if do_met: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_metal_xdmat = {} minutes\n\n'.format((done-start)/60))

    return


def send_cf(zmin,zmax,do_corr,do_dist,do_met,f='LYA',l='LYA'):

    if (zmin==0.) and (zmax==10.):
        zmin = int(zmin)
        zmax = int(zmax)
    strl = l.replace('(','').replace(')','')

    ###
    cmd = 'picca_cf.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f)
    cmd += ' --out {}/Correlations/cf_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    #cmd += ' --remove-same-half-plate-close-pairs'
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    if l!='LYA':
        cmd += ' --lambda-abs '+l.replace('(','\(').replace(')','\)')

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('cf_','cf_{}_in_{}_'.format(strl,f))
    print('')
    print(cmd)
    if do_corr: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_cf = {} minutes\n\n'.format((done-start)/60))

    ###
    cmd = 'picca_dmat.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f)
    cmd += ' --out {}/Correlations/dmat_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    #cmd += ' --remove-same-half-plate-close-pairs'
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.99'
    if l!='LYA':
        cmd += ' --lambda-abs '+l.replace('(','\(').replace(')','\)')

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('dmat_','dmat_{}_in_{}_'.format(strl,f))
    print('')
    print(cmd)
    if do_dist: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_dmat = {} minutes\n\n'.format((done-start)/60))

    ###
    cmd = 'picca_export.py'
    cmd += ' --data {}/Correlations/cf_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    cmd += ' --dmat {}/Correlations/dmat_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    cmd += ' --out {}/Correlations/cf_z_{}_{}-exp.fits.gz'.format(path_here,zmin,zmax)

    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('cf_','cf_{}_in_{}_'.format(strl,f))
        cmd = cmd.replace('dmat_','dmat_{}_in_{}_'.format(strl,f))
    print('')
    print(cmd)
    if do_dist: subprocess.call(cmd, shell=True)

    ###
    cmd = 'picca_metal_dmat.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f)
    cmd += ' --out {}/Correlations/metal_dmat_z_{}_{}.fits.gz'.format(path_here,zmin,zmax)
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    #cmd += ' --remove-same-half-plate-close-pairs'
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.999'
    cmd += ' --abs-igm '
    for m in metList[f]:
        cmd += m+' '
    if l!='LYA':
        cmd += ' --lambda-abs '+l.replace('dmat_','dmat_{}_in_{}_'.format(strl,f))
    if (f!='LYA') or (l!='LYA'):
        cmd = cmd.replace('metal_dmat_','metal_dmat_{}_in_{}_'.format(strl,f))
    cmd = cmd.replace('(','\(').replace(')','\)')
    print('')
    print(cmd)
    if do_met: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_metal_dmat = {} minutes\n\n'.format((done-start)/60))


    return


def send_cf_cross(zmin,zmax,do_corr,do_dist,do_met,f1='LYA',l1='LYA',f2='LYB',l2='LYA'):

    strl1 = l1.replace('(','').replace(')','')
    strl2 = l2.replace('(','').replace(')','')

    if (zmin==0.) and (zmax==10.):
        zmin = int(zmin)
        zmax = int(zmax)

    cmd = 'picca_cf.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f1)
    cmd += ' --in-dir2 {}/Delta_{}/Delta/'.format(path_deltas,f2)
    cmd += ' --out {}/Correlations/cf_{}_in_{}_{}_in_{}_z_{}_{}.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    print('')
    print(cmd)
    if do_corr: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_cf (cross) = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_dmat.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f1)
    cmd += ' --in-dir2 {}/Delta_{}/Delta/'.format(path_deltas,f2)
    cmd += ' --out {}/Correlations/dmat_{}_in_{}_{}_in_{}_z_{}_{}.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.99'
    print('')
    print(cmd)
    if do_dist: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_dmat (cross) = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_export.py'
    cmd += ' --data {}/Correlations/cf_{}_in_{}_{}_in_{}_z_{}_{}.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    cmd += ' --dmat {}/Correlations/dmat_{}_in_{}_{}_in_{}_z_{}_{}.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    cmd += ' --out {}/Correlations/cf_{}_in_{}_{}_in_{}_z_{}_{}-exp.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    print('')
    print(cmd)
    if do_dist: subprocess.call(cmd, shell=True)

    ###
    cmd = 'picca_metal_dmat.py'
    cmd += ' --in-dir {}/Delta_{}/Delta/'.format(path_deltas,f1)
    cmd += ' --in-dir2 {}/Delta_{}/Delta/'.format(path_deltas,f2)
    cmd += ' --out {}/Correlations/metal_dmat_{}_in_{}_{}_in_{}_z_{}_{}.fits.gz'.format(path_here,strl1,f1,strl2,f2,zmin,zmax)
    cmd += ' --z-cut-min {} --z-cut-max {}'.format(zmin, zmax)
    cmd += ' --fid-Om 0.314569514863487 --fid-Or 7.97505418919554e-5'
    cmd += ' --nside 16'
    cmd += ' --rej 0.999'
    cmd += ' --abs-igm '
    for m in metList[f1]:
        cmd += m+' '
    cmd += ' --abs-igm2 '
    for m in metList[f2]:
        cmd += m+' '
    cmd = cmd.replace('(','\(').replace(')','\)')
    print('')
    print(cmd)
    if do_met: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_metal_dmat (cross) = {} minutes\n\n'.format((done-start)/60))


def parse():

    parser=argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Measure a particular correlation")

    parser.add_argument('--corr_type', type=str, required=True, 
                        help="Correlation type (LyaLya, LyaQSO, LyaLyb or LybQSO)")
    parser.add_argument('--zmin', type=float, default=0.0, help="minimum redshift")
    parser.add_argument('--zmax', type=float, default=10.0, help="maximum redshift")
    parser.add_argument('--do_corr', action = "store_true", 
                        help="compute correlation (auto or cross)")
    parser.add_argument('--do_dist', action = "store_true", 
                        help="compute distortion matrix (assumes correlation is done)")
    parser.add_argument('--do_met', action = "store_true", 
                        help="compute metal distortion matrix")
    
    return parser.parse_args()


print('start job')

args = parse()

corr_type=args.corr_type
zmin=args.zmin
zmax=args.zmax
do_corr=args.do_corr
do_dist=args.do_dist
do_met=args.do_met

if corr_type == 'LyaQSO':
    print('compute LyaQSO')
    send_xcf(zmin,zmax,do_corr=do_corr,do_dist=do_dist,do_met=do_met)
    print('\n\n\n\n')
elif corr_type == 'LyaLya':
    print('compute LyaLya')
    send_cf(zmin,zmax,do_corr=do_corr,do_dist=do_dist,do_met=do_met)
    print('\n\n\n\n')
elif corr_type == 'LybQSO':
    print('compute LybQSO')
    send_xcf(zmin,zmax,do_corr=do_corr,do_dist=do_dist,do_met=do_met,f='LYB')
    print('\n\n\n\n')
elif corr_type == 'LyaLyb':
    print('compute LyaLyb')
    send_cf_cross(zmin,zmax,do_corr=do_corr,do_dist=do_dist,do_met=do_met)
    print('\n\n\n\n')

