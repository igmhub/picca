import scipy as sp
import scipy.linalg
import subprocess
import fitsio
import os
import h5py
import glob
import time
import matplotlib.pyplot as plt

from picca.constants import absorber_IGM

# full path to DR12Q file
path_dr12q = os.environ['PICCA_DR12']+'/Catalogs/DR12Q_v2_10_addDR7.fits'

# full path to this folder
path_here = os.environ['REDO_dMdB17']

metList = ['CIV(eff)','SiII(1260)','SiIII(1207)','SiII(1193)','SiII(1190)']


def send_xcf(send=False):

    in_dir = path_here+'/Delta_LYA/Delta/'

    cmd = 'picca_xcf.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_dr12q
    cmd += ' --out {}/Correlations/xcf.fits.gz'.format(path_here)
    cmd += ' --z-evol-obj 1.44'
    cmd += ' --nside 16'
    
    print('')
    print(cmd)
    if send: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_xcf = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_xdmat.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_dr12q
    cmd += ' --out {}/Correlations/xdmat.fits.gz'.format(path_here)
    cmd += ' --z-evol-obj 1.44 '
    cmd += ' --nside 16'
    cmd += ' --rej 0.99'
    
    print('')
    print(cmd)
    if send: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_xdmat = {} minutes\n\n'.format((done-start)/60))

    cmd = 'picca_export.py'
    cmd += ' --data {}/Correlations/xcf.fits.gz'.format(path_here)
    cmd += ' --dmat {}/Correlations/xdmat.fits.gz'.format(path_here)
    cmd += ' --out {}/Correlations/xcf-exp.fits.gz'.format(path_here)

    print('')
    print(cmd)
    if send: subprocess.call(cmd, shell=True)

    cmd = 'picca_metal_xdmat.py'
    cmd += ' --in-dir '+in_dir
    cmd += ' --drq '+path_dr12q
    cmd += ' --out {}/Correlations/metal_xdmat.fits.gz'.format(path_here)
    cmd += ' --z-evol-obj 1.44 '
    cmd += ' --nside 16'
    cmd += ' --rej 0.999'
    cmd += ' --abs-igm '
    for m in metList:
        cmd += m+' '
    cmd = cmd.replace('(','\(').replace(')','\)')

    print('')
    print(cmd)
    if send: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_metal_xdmat = {} minutes\n\n'.format((done-start)/60))

    return


print('start job')

send_xcf(send=True)

print('finished job')

