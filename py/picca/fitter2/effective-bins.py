#!/usr/bin/env python

import os
import sys
import fitsio
import copy
import functools
import argparse
import h5py
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
if (sys.version_info > (3, 0)):
    import configparser as ConfigParser
else:
    import ConfigParser

from ..utils import userprint
from . import parser as fit_parser


labels = {
    'ap':'\\alpha_{\parallel}',
    'at':'\\alpha_{\perp}',
    'beta_LYA':'\\beta_{\mathrm{Ly}\\alpha}',
    'bias_eta_LYA':'b_{\mathrm{Ly}\\alpha}',
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

def extract_h5file(fname):
    '''

    '''

    f = h5py.File(os.path.expandvars(fname),'r')

    free_p = [ el.decode('UTF-8') for el in f['best fit'].attrs['list of free pars'] ]
    fixed_p = [ el.decode('UTF-8') for el in f['best fit'].attrs['list of fixed pars'] ]
    pars = { el:f['best fit'].attrs[el][0] for el in free_p }
    err_pars = { el:f['best fit'].attrs[el][1] for el in free_p }
    pars.update({ el:f['best fit'].attrs[el][0] for el in fixed_p })
    err_pars.update({ el:0. for el in fixed_p })

    f.close()

    return free_p, fixed_p, pars, err_pars

def extract_data(chi2file,dic_init):

    data = {}

    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(os.path.expandvars(chi2file))
    zeff = cp.get('data sets','zeff')

    for d in cp.get('data sets','ini files').split():
        dic_data = fit_parser.parse_data(os.path.expandvars(d),zeff,dic_init['fiducial'])
        name = dic_data['data']['name']
        data[name] = {}

        f = fitsio.FITS(dic_data['data']['filename'])

        head = f[1].read_header()
        data[name]['nt'] = head['NT']
        data[name]['np'] = head['NP']
        data[name]['rt_min'] = 0.
        data[name]['rt_max'] = head['RTMAX']
        data[name]['rp_min'] = head['RPMIN']
        data[name]['rp_max'] = head['RPMAX']

        data[name]['r_min'] = dic_data['cuts']['r-min']
        data[name]['r_max'] = dic_data['cuts']['r-max']
        f.close()

    return data

def apply_mask(data):
    for k in data.dm_dp.keys():
        data.dm_dp[k][~data.mask] = 0.
    data.ico = linalg.inv(data.co)
    data.ico[:,~data.mask] = 0.
    data.ico[~data.mask,:] = 0.
    data.z[~data.mask] = 0.

    return

def xi_mod(pars, data, dic_init):

    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pksb_lin = dic_init['fiducial']['pksb']

    pars['SB'] = False
    xi_best_fit = pars['bao_amp']*data.xi_model(k, pk_lin-pksb_lin, pars)

    pars['SB'] = True & (not dic_init['fiducial']['full-shape'])
    snl_par = pars['sigmaNL_par']
    snl_per = pars['sigmaNL_per']
    pars['sigmaNL_par'] = 0.
    pars['sigmaNL_per'] = 0.
    xi_best_fit += data.xi_model(k, pksb_lin, pars)

    pars['SB'] = False
    pars['sigmaNL_par'] = snl_par
    pars['sigmaNL_per'] = snl_per

    return xi_best_fit

def xi_mod_p(x, data, dic_init, pname, pars):

    pars2 = copy.deepcopy(pars)
    pars2[pname] = x

    return xi_mod(pars2, data=data, dic_init=dic_init)

def compute_dm_dp(data, dic_init, freep, pars):

    dm = {}
    for p in freep:
        userprint('Parameter {}'.format(p))
        g = functools.partial(xi_mod_p, data=data, dic_init=dic_init, pname=p, pars=pars)
        dm[p] = derivative(g,pars[p])

    return dm

def compute_M(dm_dp,ico):
    M = ico*(dm_dp*dm_dp[:,None])
    return M

def compute_z0(M,z):
    res = (M*z[:,None]).sum()
    den = M.sum()
    return res,den

def plot_xi(xi,data,title=' '):
    plt.figure()
    plt.imshow(xi.reshape((data['np'],data['nt'])),origin=0,interpolation='nearest',
        extent=[data['rt_min'],data['rt_max'],data['rp_min'],data['rp_max']],cmap='seismic',
        vmin=-max(abs(xi.min()),abs(xi.max())),vmax=max(abs(xi.min()),abs(xi.max())))
    plt.colorbar()
    plt.ylabel(r"$r_{\parallel}$",size=20)
    plt.xlabel(r"$r_{\perp}$",size=20)
    plt.title(r'$'+title+'$',size = 20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Measure the bins contribution to the fitted parameters and  compute their effective redshift')

    parser.add_argument('--chi2-file', type=str, required=True,
        help = "Path to the config 'chi2.ini' file used in fitter2")

    parser.add_argument('--params', type=str,default=['all'], required=False, nargs='*',
        help="List of the fitted parameters, if 'all' in list compute all")

    parser.add_argument('--plot-effective-bins', action='store_true',
        help='Display an image with the bins involved in the fit of each selected parameter')

    args = parser.parse_args()

    ### Open files
    dic_init = fit_parser.parse_chi2(args.chi2_file)

    free_pars, fixed_pars, best_fit_pars, err_best_fit_pars = extract_h5file(dic_init['outfile'])
    if 'all' in args.params:
        args.params = free_pars.copy()
    if np.any(~np.in1d(args.params,free_pars)):
        print('ERROR: Some parameters are not fitted {}, the list is {}'.format(args.params,free_pars))
        sys.exit(12)

    ### Computing derivatives for each parameter, in each correlation
    for data in dic_init['data sets']['data']:
        userprint('\n data: {}\n'.format(data.name))
        data.dm_dp = compute_dm_dp(data,dic_init,free_pars,best_fit_pars)
        apply_mask(data)

    ### Computation of the effective bins
    userprint("\n")
    for p in args.params:
        userprint('\n\nParameter {}'.format(p))

        res = []
        den = []
        for data in dic_init['data sets']['data']:
            M = compute_M(data.dm_dp[p],data.ico)
            tres, tden = compute_z0(M,data.z)
            res += [tres]
            den += [tden]
            userprint('{}, <z> = {}/{} = {}'.format(data.name,tres,tden,tres/tden))

        if len(dic_init['data sets']['data'])>1:
            res = np.array(res).sum()
            den = np.array(den).sum()
            print('Combined')
            print('<z> = {}/{} = {}'.format(res,den,res/den))

    ### Plot
    if args.plot_effective_bins:
        datap = extract_data(args.chi2_file, dic_init)
        for p in args.params:
            lab = '\partial m/ \partial '
            if p in labels:
                lab += labels[p]
            else:
                lab += p.replace('_','-')
            for data in dic_init['data sets']['data']:
                plot_xi(data.dm_dp[p],datap[data.name],data.name.replace('_','-')+'\,'+lab)
        plt.show()
