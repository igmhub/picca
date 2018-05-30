from __future__ import print_function

from functools import partial
import sys
import scipy as sp
import os.path
from pkg_resources import resource_filename
if (sys.version_info > (3, 0)):
    # Python 3 code in this block
    import configparser as ConfigParser
else:
    import ConfigParser

import fitsio
from . import data, utils, priors

def parse_chi2(filename):
    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(filename)

    dic_init = {}

    dic_init['fiducial'] = {}

    if cp.has_section('fiducial'):
        p = cp.get('fiducial','filename')
        p = os.path.expandvars(p)
        print('INFO: reading input Pk {}'.format(p))
    else:
        p = resource_filename('picca', 'fitter2/models/PlanckDR12/PlanckDR12.fits')
        p = os.path.expandvars(p)
        print('INFO: reading default Pk {}'.format(p))

    h = fitsio.FITS(p)
    zref = h[1].read_header()['ZREF']
    dic_init['fiducial']['zref'] = zref
    dic_init['fiducial']['Om'] = h[1].read_header()['OM']
    dic_init['fiducial']['OL'] = h[1].read_header()['OL']
    dic_init['fiducial']['k'] = h[1]['K'][:]
    dic_init['fiducial']['pk'] = h[1]['PK'][:]
    dic_init['fiducial']['pksb'] = h[1]['PKSB'][:]

    zeff = float(cp.get('data sets','zeff'))
    dic_init['data sets'] = {}
    dic_init['data sets']['zeff'] = zeff
    dic_init['data sets']['data'] = [data.data(parse_data(os.path.expandvars(d),zeff,dic_init['fiducial'])) for d in cp.get('data sets','ini files').split()]

    utils.cosmo_fit_func = getattr(utils, cp.get('cosmo-fit type','cosmo fit func'))

    dic_init['outfile'] = cp.get('output','filename')

    if 'verbosity' in cp.sections():
        dic_init['verbosity'] = int(cp.get('verbosity','level'))

    if 'fast mc' in cp.sections():
        dic_init['fast mc'] = {}
        for item, value in cp.items('fast mc'):
            if item=='covscaling':
                value = value.split()
                dic_init['fast mc'][item] = sp.array(value).astype(float)
                if not len(dic_init['fast mc'][item])==len(dic_init['data sets']['data']):
                    raise AssertionError()
            else:
                dic_init['fast mc'][item] = int(value)

    if cp.has_section('minos'):
        dic_init['minos'] = {}
        for item, value in cp.items('minos'):
            if item=='sigma':
                value = float(value)
            elif item=='parameters':
                value = value.split()
            dic_init['minos'][item] = value

    if cp.has_section('chi2 scan'):
        dic_init['chi2 scan'] = parse_chi2scan(cp.items('chi2 scan'))

    return dic_init

def parse_data(filename,zeff,fiducial):
    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(filename)

    dic_init = {}
    dic_init['data'] = {}
    print("INFO: reading {}".format(filename))
    for item, value in cp.items('data'):
        if item == "rp_binsize" or value == "rt_binsize":
            value = float(value)
        if item == "ell-max":
            value = int(value)
        dic_init['data'][item] = value

    dic_init['cuts'] = {}
    for item, value in cp.items('cuts'):
        dic_init['cuts'][item] = float(value)

    dic_init['model'] = {}
    dic_init['model']['zeff'] = zeff
    dic_init['model']['zref'] = fiducial['zref']
    dic_init['model']['Om'] = fiducial['Om']
    dic_init['model']['OL'] = fiducial['OL']
    for item, value in cp.items('model'):
        dic_init['model'][item] = value

    dic_init['parameters'] = {}
    dic_init['parameters']['values'] = {}
    dic_init['parameters']['errors'] = {}
    dic_init['parameters']['limits'] = {}
    dic_init['parameters']['fix'] = {}
    for item, value in cp.items('parameters'):
        value = value.split()
        dic_init['parameters']['values'][item] = float(value[0])
        dic_init['parameters']['errors']['error_'+item] = float(value[1])
        lim_inf = None
        lim_sup = None
        if value[2] != 'None': lim_inf = float(value[2])
        if value[3] != 'None': lim_sup = float(value[3])
        dic_init['parameters']['limits']['limit_'+item]=(lim_inf,lim_sup)
        assert value[4] == 'fixed' or value[4] == 'free'
        dic_init['parameters']['fix']['fix_'+item] = value[4] == 'fixed'

    if 'metals' in cp.sections():
        dic_init['metals']={}
        for item, value in cp.items('metals'):
            dic_init['metals'][item] = value
        if 'in tracer1' in dic_init['metals']:
            dic_init['metals']['in tracer1'] = dic_init['metals']['in tracer1'].split()
        if 'in tracer2' in dic_init['metals']:
            dic_init['metals']['in tracer2'] = dic_init['metals']['in tracer2'].split()

    if 'priors' in cp.sections():
        for item, value in cp.items('priors'):
            if item in priors.prior_dic.keys():
                print("WARNING: prior on {} will be overwritten".format(item))
            value = value.split()
            priors.prior_dic[item] = partial(getattr(priors, value[0]), prior_pars=sp.array(value[1:]).astype(float), name=item)

    return dic_init

def parse_chi2scan(items):

    assert len(items)==1 or len(items)==2

    dic_init = {}
    for item, value in items:
        dic = {}
        value = value.split()
        dic['min']    = float(value[0])
        dic['max']    = float(value[1])
        dic['nb_bin'] = int(value[2])
        dic['grid']   = sp.linspace(dic['min'],dic['max'],num=dic['nb_bin'],endpoint=True)
        dic_init[item] = dic

    return dic_init
