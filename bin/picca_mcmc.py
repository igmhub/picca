#!/usr/bin/env python
from __future__ import print_function
import iminuit
import types
import configargparse
import emcee
import numpy as np
from numpy import random

from picca.utils import print
from picca.fitter import parameters, cosmo, Chi2, metals


param = parameters.parameters()
parser = configargparse.ArgParser()

### Get the parser
parser.add('-c', '--config_file', required=False, is_config_file=True, help='config file path')

for i in param.dic_init_float:
    parser.add('--'+i,type=types.FloatType,required=False,help=param.help[i], default=param.dic_init_float[i])
for i in param.dic_init_int:
     parser.add('--'+i,type=types.IntType,required=False,help=param.help[i], default=param.dic_init_int[i])
for i in param.dic_init_bool:
    parser.add('--'+i,action='store_true',required=False,help=param.help[i], default=param.dic_init_bool[i])
for i in param.dic_init_string:
    parser.add('--'+i,type=types.StringType,required=False,help=param.help[i], default=param.dic_init_string[i])
for i in param.dic_init_list_string:
    parser.add('--'+i,type=types.StringType,required=False,help=param.help[i],default=param.dic_init_list_string[i],nargs="*")

parser.add_argument('--nwalkers',type=int,required=False,help='number of mcmc walkers', default=100)
parser.add_argument('--nburn',type=int,required=False,help='number of samples to burn', default=10)
parser.add_argument('--nsamples',type=int,required=False,help='number of samples', default=10)
parser.add_argument('--nthreads',type=int,required=False,help='number of threads', default=1)

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
        if any(i in el for el in chi2.pname):
            kw['fix_'+i]=True

if not dic_init['free'] is None:
    for i in dic_init['free']:
        if any(i in el for el in chi2.pname):
            kw['fix_'+i]=False

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
mig.migrad()
chi2.export(mig,param)

p0 = [mig.values[p] for p in chi2.pname]
dp0 = [mig.errors[p] for p in chi2.pname]
def lnprob(p):
    lnp = -0.5*chi2(*p)+lnpriors(p)
    return lnp

def lnpriors(p):
    if dic_init['limit'] is not None:
        for i in range(0,len(dic_init['limit']),3):
            pname = dic_init['limit'][i]
            val=p[chi2.pname.index(pname)]
            p_min = dic_init['limit'][i+1]
            if p_min=="None":p_min=-np.inf
            p_max = dic_init['limit'][i+2]
            if p_max=="None":p_max=np.inf
            p_min=float(p_min)
            p_max=float(p_max)
            if val<p_min or val>p_max:
                return -np.inf
    return 0.



nwalkers = args.nwalkers
p=p0+1e-3*np.array(dp0*random.rand(nwalkers*len(dp0)).reshape(nwalkers,len(dp0)))
sampler = emcee.EnsembleSampler(nwalkers, len(chi2.pname), lnprob,threads=args.nthreads)
pos, prob, state = sampler.run_mcmc(p, args.nburn)

def output_chain(sampler,ofile):
    f=open(ofile,"w")
    f.write("walker iteration ")
    for p in chi2.pname:
        f.write(p+" ")
    f.write("\n")

    for i in range(nwalkers):
        for j in range(len(sampler.chain[i,:,0])):
            f.write(str(i)+" "+str(j)+" ")
            for k in range(len(chi2.pname)):
                f.write(str(sampler.chain[i,j,k])+" ")
            f.write("\n")
    f.close()

output_chain(sampler,args.output_prefix+"mcmc.burn.dat")

sampler.reset()
sampler.run_mcmc(pos, args.nsamples)
output_chain(sampler,args.output_prefix+"mcmc.samples.dat")
