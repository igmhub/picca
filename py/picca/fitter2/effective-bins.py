#!/usr/bin/env python
import h5py
import sys
import matplotlib.pyplot as plt
import scipy as sp
import fitsio
import copy
import functools
import argparse
from scipy import linalg
if (sys.version_info > (3, 0)):
    import configparser as ConfigParser
else:
    import ConfigParser
from picca.fitter2 import parser as fit_parser

def derivative(f,x):
    eps = 10**(-5)
    der = (f(x+eps)-f(x))/eps
    return der

def extract_h5file(fname):
    f = h5py.File(fname)
    free_p  = []
    for i in f['best fit'].attrs['list of free pars']: free_p.append(i.decode('UTF-8'))
    fixed_p  = []
    for i in f['best fit'].attrs['list of fixed pars']: free_p.append(i.decode('UTF-8'))
    pars = {}
    err_pars = {}
    for i in free_p:
        pars[i] = f['best fit'].attrs[i][0]
        err_pars[i] = f['best fit'].attrs[i][1]
    for i in fixed_p:
        pars[i] = f['best fit'].attrs[i][0]
        err_pars[i] = 0
    xi= sp.array(f['LYA(LYA)-LYA(LYA)']['fit'])
    return free_p,fixed_p,pars,err_pars,xi

def extract_data(chi2file,dic_init):
    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(chi2file)
    dic_data = fit_parser.parse_data(cp.get('data sets','ini files'),cp.get('data sets','zeff'),dic_init['fiducial'])
    data_file = dic_data['data']['filename']
    f = fitsio.FITS(data_file)
    cov = f[1]["CO"][:]
    rp  = f[1]["RP"][:]
    rt  = f[1]["RT"][:]
    z   = f[1]["Z"][:]
    ico = linalg.inv(cov)

    head = f[1].read_header()
    nt = head['NT']
    np = head['NP']
    rt_min = 0
    rt_max = head['RTMAX']
    rp_min = head['RPMIN']
    rp_max = head['RPMAX']

    r_min = dic_data['cuts']['r-min']
    r_max = dic_data['cuts']['r-max']
    return ico,rp,rt,np,nt,rp_min,rp_max,rt_min,rt_max,r_min,r_max,z

def apply_mask(dm,ico,z,dic):
    for d in dic['data sets']['data']:
        for i in dm:
            dm_dp[i][d.mask==0] = 0
        ico[:,d.mask==0] = 0
        ico[d.mask==0,:] = 0
        z[d.mask==0] = 0
    
def xi_mod(pars,dic_init):
    k = dic_init['fiducial']['k']
    pk_lin = dic_init['fiducial']['pk']
    pksb_lin = dic_init['fiducial']['pksb']
    for d in dic_init['data sets']['data']:
        pars['SB'] = False
        xi_best_fit = pars['bao_amp']*d.xi_model(k, pk_lin-pksb_lin, pars)
        pars['SB'] = True
        snl = pars['sigmaNL_per']
        pars['sigmaNL_per'] = 0
        xi_best_fit += d.xi_model(k, pksb_lin, pars)
        pars['sigmaNL_per'] = snl
        pars['SB'] = False
        
    return xi_best_fit
    
def plot_xi(xi,title = ' '):
    plt.figure()
    plt.imshow(xi.reshape((np,nt)),origin=0,interpolation='nearest',extent=[rt_min,rt_max,rp_min,rp_max],cmap='seismic')
    plt.colorbar()
    plt.ylabel(r"$r_{\parallel}$",size=20)
    plt.xlabel(r"$r_{\perp}$",size=20)
    plt.title(title,size = 20)

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
    res = 0
    den = 0
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            res += M[i,j]*z[i]
            den += M[i,j]
    return res,den

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--chi2-file', type = str, default = None, required=True,
                        help = 'chi2 input file name')
    
    parser.add_argument('--params', type=str,default=[], required=False,nargs='*',
                        help='List of the fitted parameters')
    
    parser.add_argument('--plot-effective-bins', action='store_true', required=True,
                        help='display an image with the bins involved in the fit of each selected parameter')

    args = parser.parse_args()
    chi2_file = args.chi2_file

    if not args.params:
        print('ERROR : empty parameter list')
        sys.exit(12)

    ######### Open files
    dic_init = fit_parser.parse_chi2(chi2_file)
    h5_file = dic_init['outfile']
    free_pars,fixed_pars,best_fit_pars,err_best_fit_pars,xi_best_fit = extract_h5file(h5_file)
    for p in args.params:
        if p not in free_pars:
            print('ERROR : parameter %s is not fitted'%(p))
            sys.exit(12)
    ico,rp,rt,np,nt,rp_min,rp_max,rt_min,rt_max,r_min,r_max,z = extract_data(chi2_file,dic_init)
    ######### Computation of the effective bins
    dm_dp = compute_dm_dp(free_pars,best_fit_pars)
    apply_mask(dm_dp,ico,z,dic_init)

    for p in args.params:
        print("Parameter %s"%(p))
        M = compute_M(dm_dp[p],ico)
        res,den = compute_z0(M,z)
        print("res = ",res,"den = ",den)
        print("<z> = ",res/den)
        print(" ")

    if args.plot_effective_bins:
        labels = {}
        labels['ap'] = r'$\alpha_{\parallel}$'
        labels['at'] = r'$\alpha_{\perp}$'
        labels['beta_LYA'] = r'$\beta_{Ly\alpha}$'
        labels['bias_LYA'] = r'$b_{Ly\alpha}$'
        labels['bias_hcd'] = r'$b_{HCD}$'
        labels['beta_hcd'] = r'$\beta_{HCD}$'
        labels['bias_SiII(1190)'] = r'$b_{SiII(1190)}$'
        labels['bias_SiII(1193)'] = r'$b_{SiII(1193)}$'
        labels['bias_SiIII(1207)'] = r'$b_{SiIII(1207)}$'
        labels['bias_SiII(1260)'] = r'$b_{SiII(1260)}$'

        for p in args.params:
            if p in labels:
                lab = r'$\partial m/ \partial$'+labels[p]
            else :
                lab = r'$\partial m/ \partial$'+p
            plot_xi(dm_dp['ap'],lab)
        plt.show()
