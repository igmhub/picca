#!/usr/bin/env python
from __future__ import print_function
import iminuit
import sys
import configargparse

from picca.utils import print
from picca.fitter import parameters, cosmo, Chi2, metals

### Get the parser
param = parameters.parameters()
parser = configargparse.ArgParser()

parser.add('-c', '--config_file', required=False, is_config_file=True, help='config file path')
for i in param.dic_init_float:
    parser.add('--'+i,type=float,required=False,help=param.help[i], default=param.dic_init_float[i])
for i in param.dic_init_int:
     parser.add('--'+i,type=int,required=False,help=param.help[i], default=param.dic_init_int[i])
for i in param.dic_init_bool:
    parser.add('--'+i,action='store_true',required=False,help=param.help[i], default=param.dic_init_bool[i])
for i in param.dic_init_string:
    parser.add('--'+i,type=str,required=False,help=param.help[i], default=param.dic_init_string[i])
for i in param.dic_init_list_string:
    parser.add('--'+i,type=str,required=False,help=param.help[i],default=param.dic_init_list_string[i],nargs="*")

args, unknown = parser.parse_known_args()
param.set_parameters_from_parser(args=args,unknown=unknown)
param.test_init_is_valid()
dic_init = param.dic_init

kw=dict()

met=None
if not dic_init['metals'] is None:
    met=metals.model(dic_init)
    met.templates = not dic_init['metal_dmat']
    met.grid = not dic_init['metal_xdmat']
    if not dic_init['data_auto'] is None:met.add_auto()
    if not dic_init['data_cross'] is None:met.add_cross(dic_init)

    fix_met = met.fix
    for i in fix_met:
        kw['fix_'+i]=True

    for n,v in zip(met.pname,met.pinit):
        print(n,v)
        kw[n]=v
        kw['error_'+n]=0.005
    print()

m=cosmo.model(dic_init)
if not dic_init['data_auto'] is None:
    m.add_auto(dic_init)
if not dic_init['data_cross'] is None:
    m.add_cross(dic_init)
if not dic_init['data_autoQSO'] is None:
    m.add_autoQSO(dic_init)

chi2=Chi2.Chi2(dic_init,cosmo=m,met=met)

for n,v in zip(m.pall,m.pinit):
    #print(n,v)
    kw[n]=v
    if abs(v)!=0:
        kw['error_'+n]=abs(v)/10.
    else:
        kw['error_'+n]=0.1

fix=m.fix
for i in fix:
    kw['fix_'+i]=True

if not dic_init['fix'] is None:
    for i in dic_init['fix']:
        if any(i==el for el in chi2.pname):
            kw['fix_'+i]=True
        else:
            print()
            print('  fit/bin/fit:: Unknown parameter = ', i)
            print('  Exit')
            sys.exit(0)

if not dic_init['free'] is None:
    for i in dic_init['free']:
        if any(i==el for el in chi2.pname):
            kw['fix_'+i]=False
        else:
            print()
            print('  fit/bin/fit:: Unknown parameter = ', i)
            print('  Exit')
            sys.exit(0)

if not dic_init['limit'] is None:
    for i in range(0,len(dic_init['limit']),3):

        ###
        limit_lower = dic_init['limit'][i+1]
        if (limit_lower=='None'):
            limit_lower = None
        else:
            limit_lower = float(limit_lower)

        ###
        limit_upper = dic_init['limit'][i+2]
        if (limit_upper=='None'):
            limit_upper = None
        else:
            limit_upper = float(limit_upper)

        kw['limit_'+dic_init['limit'][i]]=(limit_lower,limit_upper)

### Set to nan the error of fixed parameters.
for i in kw:
    if ( len(i)<len('fix_') ): continue
    if ( i[:4]!='fix_' ): continue
    if not kw[i]: continue
    kw[ 'error_'+i[4:] ] = 0.

mig = iminuit.Minuit(chi2, throw_nan=True ,forced_parameters=chi2.pname,print_level=dic_init['verbose'],errordef=1,**kw)
mig.tol = dic_init['migrad_tol']
if not dic_init['debug']:
    try:
        mig.migrad()
        if not dic_init['no_hesse']:
            mig.hesse()
    except Exception as error :
        print()
        print('  fit/bin/fit:: error in minimization = ', error)
        print('  Exit')
        sys.exit(0)

if not dic_init['minos'] is None and mig.migrad_ok():
    if any('_all_'==el for el in dic_init['minos']):
        mig.minos(var=None,sigma=dic_init['sigma_minos'])
    else:
        for i in dic_init['minos']:
            if any(i==el for el in mig.list_of_vary_param()):
                mig.minos(var=i,sigma=dic_init['sigma_minos'])

chi2.export(mig,param)

### Scan of chi2
if not dic_init['chi2Scan'] is None:
    chi2.chi2Scan(mig,kw)
### Realisation of fast Monte-Carlo
elif not dic_init['fastMonteCarlo'] is None:
    chi2.fastMonteCarlo(mig,kw)
