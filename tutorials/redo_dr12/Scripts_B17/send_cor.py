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


# full path to files
path_here = os.environ['REDO_B17']

metList = ['CIV(eff)','SiII(1260)','SiIII(1207)','SiII(1193)','SiII(1190)']


def send_cf(send=False):

    ###
    cmd = 'picca_cf.py'
    cmd += ' --in-dir {}/Delta_LYA/Delta/'.format(path_here)
    cmd += ' --out {}/Correlations/cf.fits.gz'.format(path_here)
    cmd += ' --z-cut-min 0.0 --z-cut-max 10.0'
    cmd += ' --remove-same-half-plate-close-pairs'
    cmd += ' --nside 16'

    print('')
    print(cmd)

    if send: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_cf = {} minutes\n\n'.format((done-start)/60))

    ###
    cmd = 'picca_dmat.py'
    cmd += ' --in-dir {}/Delta_LYA/Delta/'.format(path_here)
    cmd += ' --out {}/Correlations/dmat.fits.gz'.format(path_here)
    cmd += ' --z-cut-min 0.0 --z-cut-max 10.0'
    cmd += ' --remove-same-half-plate-close-pairs'
    cmd += ' --nside 16'
    cmd += ' --rej 0.99'

    print('')
    print(cmd)

    if send: 
        start = time.time()
        subprocess.call(cmd, shell=True)
        done = time.time()
        print('\n\nTime spent in picca_dmat = {} minutes\n\n'.format((done-start)/60))

    ###
    cmd = 'picca_export.py'
    cmd += ' --data {}/Correlations/cf.fits.gz'.format(path_here)
    cmd += ' --dmat {}/Correlations/dmat.fits.gz'.format(path_here)
    cmd += ' --out {}/Correlations/cf-exp.fits.gz'.format(path_here)

    print('')
    print(cmd)

    if send: subprocess.call(cmd, shell=True)

    ###
    cmd = 'picca_metal_dmat.py'
    cmd += ' --in-dir {}/Delta_LYA/Delta/'.format(path_here)
    cmd += ' --out {}/Correlations/metal_dmat.fits.gz'.format(path_here)
    cmd += ' --z-cut-min 0.0 --z-cut-max 10.0'
    cmd += ' --remove-same-half-plate-close-pairs'
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
        print('\n\nTime spent in picca_metal_dmat = {} minutes\n\n'.format((done-start)/60))

    return


print('start job')

send_cf(send=True)

print('finished job')
