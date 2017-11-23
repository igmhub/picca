from __future__ import print_function

import ConfigParser
import fitsio
from . import data

def parse_chi2(filename):
    cp = ConfigParser.ConfigParser()
    cp.optionxform=str
    cp.read(filename)

    dic_init = {}

    dic_init['data sets'] = [data.data(parse_data(d)) for d in cp.get('data sets','ini files').split()]

    dic_init['fiducial'] = {}
    h = fitsio.FITS(cp.get('fiducial','filename'))
    dic_init['fiducial']['k'] = h[1]['K'][:]
    dic_init['fiducial']['pk'] = h[1]['PK'][:]
    dic_init['fiducial']['pksb'] = h[1]['PKSB'][:]

    dic_init['outfile'] = cp.get('output','filename')
    if 'fast mc' in cp.sections():
        dic_init['fast mc'] = {}
        for item, value in cp.items('fast mc'):
            dic_init['fast mc'][item] = int(value)

    return dic_init

def parse_data(filename):
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
        if item == 'zref':
            value = float(value)
        dic_init['data'][item] = value

    dic_init['cuts'] = {}
    for item, value in cp.items('cuts'):
        dic_init['cuts'][item] = float(value)

    dic_init['model'] = {}
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

    return dic_init
