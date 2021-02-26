#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import fitsio
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from multiprocessing import Pool
from math import isnan
import configparser
import argparse

def bias_vs_z_std(z, zref, alpha):
    r = ((1.+z)/(1+zref))**alpha
    return r

def growthRateStructure(z, omega_M_0=0.31457):
    omega_m = omega_M_0*(1.+z)**3 / ( omega_M_0*(1.+z)**3+(1.-omega_M_0))
    f = omega_m**0.55
    return f

def update_system_status_values(path, section, system, value):

    ### Make ConfigParser case sensitive
    class CaseConfigParser(configparser.ConfigParser):
        def optionxform(self, optionstr):
            return optionstr
    cp = CaseConfigParser()
    cp.read(path)
    cf = open(path, 'w')
    cp.set(section, system, value)
    cp.write(cf)
    cf.close()

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data',type=str,default=None,required=True,
        help='input file')
    
    parser.add_argument('--chi2',type=str,default=None,required=True,
        help='modif file')
    
    args = parser.parse_args()

    h = fitsio.FITS(args.data)

    we = h['COR']['CO'][:]
    we = np.diagonal(we)
    
    z = h['COR']['Z'][:]
    rp = h['COR']['RP'][:]
    rt = h['COR']['RT'][:]
    r = np.sqrt(rp**2. + rt**2.)
    w = (r>80.) & (r<120.)
    h.close()
    
    weTot = 0.
    zeffTot = 0.
    
    if np.sum( we[w] )!=0.:
        zeff = np.sum( z[w]*we[w] )/np.sum( we[w] )
        zeffTot = zeffTot*weTot + np.sum( z[w]*we[w] )
        weTot += np.sum( we[w] )
        zeffTot /= weTot

    f = growthRateStructure(zeff)
    biasQSO = 3.7 * bias_vs_z_std(zeff, zref=2.33, alpha=1.7)
    betaQSO = f/biasQSO
    print('zeff =',zeff)
    update_system_status_values(args.chi2, 'data sets', 'zeff', str(zeff))
    
    conf = args.data.replace('.fits','.ini')
    update_system_status_values(conf, 'parameters', 'growth_rate', str(f)+' 0.1 None None fixed')
    if 'xcf' in args.data:
        update_system_status_values(conf, 'parameters', 'beta_QSO', str(betaQSO)+' 0.1 None None fixed')

                
                
    