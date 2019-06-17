#!/usr/bin/env python

import os
import sys
import fitsio
import copy
import functools
import argparse
import h5py
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
if (sys.version_info > (3, 0)):
    import configparser as ConfigParser
else:
    import ConfigParser

from picca.fitter2 import parser as fit_parser


labels = {
    'ap':'\\alpha_{\parallel}',
    'at':'\\alpha_{\perp}',
    'beta_LYA':'\\beta_{\mathrm{Ly}\\alpha}',
    'bias_LYA':'b_{\mathrm{Ly}\\alpha}',
    'bias_hcd':'b_{\mathrm{HCD}}',
    'beta_hcd':'\\beta_{\mathrm{HCD}}',
    'bias_SiII(1190)':'b_{\mathrm{SiII(1190)}}',
    'bias_SiII(1193)':'b_{\mathrm{SiII(1193)}}',
    'bias_SiIII(1207)':'b_{\mathrm{SiIII(1207)}}',
    'bias_SiII(1260)':'b_{\mathrm{SiII(1260)}}',
}


def derivative(f,x):
    eps = 10**(-5)
    der = (f(x+eps)-f(x))/eps
    return der

def extract_h5file(fname,cor_name):
    f = h5py.File(os.path.expandvars(fname),'r')
    free_p = [ el.decode('UTF-8') for el in f['best fit'].attrs['list of free pars'] ]
    fixed_p = [ el.decode('UTF-8') for el in f['best fit'].attrs['list of fixed pars'] ]
    pars = {}
    err_pars = {}
    for i in free_p:
        pars[i] = f['best fit'].attrs[i][0]
        err_pars[i] = f['best fit'].attrs[i][1]
    for i in fixed_p:
        pars[i] = f['best fit'].attrs[i][0]
        err_pars[i] = 0.
    xi = sp.array(f[cor_name]['fit'])
    f.close()

    return free_p,fixed_p,pars,err_pars,xi

def extract_data(chi2file,dic_init):
    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(os.path.expandvars(chi2file))
    dic_data = fit_parser.parse_data(os.path.expandvars(cp.get('data sets','ini files')),
        cp.get('data sets','zeff'),dic_init['fiducial'])
    data_file = dic_data['data']['filename']
    f = fitsio.FITS(data_file)
    cov = f[1]['CO'][:]
    rp = f[1]['RP'][:]
    rt = f[1]['RT'][:]
    z = f[1]['Z'][:]
    ico = linalg.inv(cov)

    head = f[1].read_header()
    nt = head['NT']
    np = head['NP']
    rt_min = 0.
    rt_max = head['RTMAX']
    rp_min = head['RPMIN']
    rp_max = head['RPMAX']

    r_min = dic_data['cuts']['r-min']
    r_max = dic_data['cuts']['r-max']
    f.close()

    return ico,rp,rt,np,nt,rp_min,rp_max,rt_min,rt_max,r_min,r_max,z

def apply_mask(dm,ico,z,dic):
    for d in dic['data sets']['data']:
        for i in dm:
            dm_dp[i][~d.mask] = 0.
        ico[:,~d.mask] = 0.
        ico[~d.mask,:] = 0.
        z[~d.mask] = 0.

def xi_mod(pars,dic_init):
    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pksb_lin = dic_init['fiducial']['pksb']
    for d in dic_init['data sets']['data']:
        pars['SB'] = False
        xi_best_fit = pars['bao_amp']*d.xi_model(k, pk_lin-pksb_lin, pars)

        pars['SB'] = True
        snl_par = pars['sigmaNL_par']
        snl_per = pars['sigmaNL_per']
        pars['sigmaNL_par'] = 0.
        pars['sigmaNL_per'] = 0.
        xi_best_fit += d.xi_model(k, pksb_lin, pars)

        pars['SB'] = False
        pars['sigmaNL_par'] = snl_par
        pars['sigmaNL_per'] = snl_per

    return xi_best_fit

def plot_xi(xi,title=' '):
    plt.figure()
    plt.imshow(xi.reshape((np,nt)),origin=0,interpolation='nearest',
        extent=[rt_min,rt_max,rp_min,rp_max],cmap='seismic',
        vmin=-max(abs(xi.min()),abs(xi.max())),vmax=max(abs(xi.min()),abs(xi.max())))
    plt.colorbar()
    plt.ylabel(r"$r_{\parallel}$",size=20)
    plt.xlabel(r"$r_{\perp}$",size=20)
    plt.title(r'$'+title+'$',size = 20)

def xi_mod_p(x,pname,pars):
    pars2 = copy.deepcopy(pars)
    pars2[pname] = x
    return xi_mod(pars2,dic_init)

def compute_dm_dp(freep,pars):
    dm = {}
    for p in freep :
        g = functools.partial(xi_mod_p,pname=p,pars=pars)
        dm[p] = derivative(g,pars[p])
    return dm

def compute_M(dm_dp,icov):
    M = sp.zeros(icov.shape)
    for i in range(icov.shape[0]):
        for j in range(icov.shape[0]):
            M[i,j] = dm_dp[i] * icov[i,j] * dm_dp[j]
    return M

def compute_z0(M,z):
    res = 0.
    den = 0.
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            res += M[i,j]*z[i]
            den += M[i,j]
    return res,den

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Measure the bins contribution to the fitted parameters and  compute their effective redshift')

    parser.add_argument('--chi2-file', type=str, required=True,
        help = "Path to the config 'chi2.ini' file used in fitter2")

    parser.add_argument('--cor-name', type=str, required=True,
        help = "Name of the correlation in 'config.ini' to look at")

    parser.add_argument('--params', type=str,default=[], required=False, nargs='*',
        help="List of the fitted parameters, if 'all' in list compute all")

    parser.add_argument('--plot-effective-bins', action='store_true',
        help='Display an image with the bins involved in the fit of each selected parameter')

    args = parser.parse_args()
    chi2_file = args.chi2_file

    ### Open files
    dic_init = fit_parser.parse_chi2(chi2_file)
    h5_file = dic_init['outfile']
    free_pars,fixed_pars,best_fit_pars,err_best_fit_pars,xi_best_fit = extract_h5file(h5_file,args.cor_name)
    if 'all' in args.params:
        args.params = free_pars
    if sp.any(~sp.in1d(args.params,free_pars)):
        print('ERROR: Some parameters are not fitted {}, the list is {}'.format(args.params,free_pars))
        sys.exit(12)
    ico,rp,rt,np,nt,rp_min,rp_max,rt_min,rt_max,r_min,r_max,z = extract_data(chi2_file,dic_init)

    ### Computation of the effective bins
    dm_dp = compute_dm_dp(free_pars,best_fit_pars)
    apply_mask(dm_dp,ico,z,dic_init)

    for p in args.params:
        print('Parameter {}'.format(p))
        M = compute_M(dm_dp[p],ico)
        res,den = compute_z0(M,z)
        print('<z> = %2.3f/%2.3f = %2.3f'%(res,den,res/den))
        print('\n')

    if args.plot_effective_bins:
        for p in args.params:
            lab = '\partial m/ \partial '
            if p in labels:
                lab += labels[p]
            else:
                lab += p
            plot_xi(dm_dp[p],lab)
        plt.show()
