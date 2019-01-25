import scipy as sp
from picca.fitter.data import data
from scipy import stats
import iminuit
import types
import copy
from picca.fitter import broadband
from picca.fitter import broadband_cross
import sys


class Chi2:

    def __init__(self,dic_init,cosmo=None,met=None):

        self.auto = None
        if not dic_init['data_auto'] is None:
            self.auto = data(kind='auto',dic_init=dic_init)
        self.cross = None
        if not dic_init['data_cross'] is None:
            self.cross = data(kind='cross',dic_init=dic_init)
        self.autoQSO = None
        if not dic_init['data_autoQSO'] is None:
            self.autoQSO = data(kind='autoQSO',dic_init=dic_init)

        self.dic_init = dic_init
        self.cosmo = cosmo

        self.pname=[]
        self.pname.extend(cosmo.pall)
        self.met = met
        if not met is None:
            self.pname.extend(met.pname)


        self.bb = None
        self.bb_cross = None
        if dic_init['bb_rmu'] or dic_init['bb_rPerp_rParal']:
            imin = dic_init['bb_i_min']
            imax = dic_init['bb_i_max']
            istep = dic_init['bb_i_step']

            ellmin = dic_init['bb_ell_min']
            ellmax = dic_init['bb_ell_max']
            ellstep = dic_init['bb_ell_step']

            if not dic_init['data_auto'] is None:
                self.bb = broadband.model(self.auto,imin,imax,istep,ellmin,ellmax,ellstep,dic_init['distort_bb_auto'])
            if not dic_init['data_cross'] is None:
                self.bb_cross = broadband_cross.model(self.cross,imin,imax,istep,ellmin,ellmax,ellstep,dic_init['distort_bb_cross'],dic_init['bb_rPerp_rParal'])

        self.verbose = dic_init['verbose']

    def chi2_auto(self,pars):
        model = self.cosmo.valueAuto(self.auto.rp,self.auto.rt,self.auto.z,{pcosmo: pars[pcosmo] for pcosmo in self.cosmo.pauto+self.cosmo.pglob})
        if not self.met is None:
            model += self.met.valueAuto(pars)


        v=sp.dot(self.auto.dm,model)

        v=v[self.auto.cuts]
        v=self.auto.da-v

        if not self.bb is None:
            p,b = self.bb.value(v)
            v -= b

        return sp.dot(v,sp.dot(self.auto.ico,v))


    def chi2_cross(self,pars):
        model =  self.cosmo.valueCross(self.cross.rp,self.cross.rt,self.cross.z,{pcosmo: pars[pcosmo] for pcosmo in self.cosmo.pcross+self.cosmo.pglob})

        if not self.met is None:
            model += self.met.valueCross(pars)

        v=sp.dot(self.cross.dm,model)
        v=v[self.cross.cuts]
        v=self.cross.da-v

        if not self.bb_cross is None:
            p,b = self.bb_cross.value(v,pars['drp'])
            v -= b

        return sp.dot(v,sp.dot(self.cross.ico,v))

    def chi2_autoQSO(self,pars):
        model =  self.cosmo.valueAutoQSO(self.autoQSO.rp,self.autoQSO.rt,self.autoQSO.z,{pcosmo: pars[pcosmo] for pcosmo in self.cosmo.pautoQSO+self.cosmo.pglob})

        v=sp.dot(self.autoQSO.dm,model)
        v=v[self.autoQSO.cuts]
        v=self.autoQSO.da-v
        return sp.dot(v,sp.dot(self.autoQSO.ico,v))

    def __call__(self,*p):
        pars=dict()
        for i,name in enumerate(self.pname):
            pars[name]=p[i]

        chi2 = 0

        if not self.auto is None:
            chi2 += self.chi2_auto(pars)

        if not self.cross is None:
            chi2 += self.chi2_cross(pars)

        if not self.autoQSO is None:
            chi2 += self.chi2_autoQSO(pars)

        if not self.dic_init['gaussian_prior'] is None:
            dic_gaussian_prior={}
            dic_init = self.dic_init
            nb_prior = len(dic_init['gaussian_prior'])//3
            for i in range(nb_prior):
                par_name = dic_init['gaussian_prior'][i*3]
                par_mean = float(dic_init['gaussian_prior'][i*3+1])
                par_sigma = float(dic_init['gaussian_prior'][i*3+2])

                if (self.verbose):
                    print("adding prior ",par_name,par_mean,par_sigma)
                    sys.stdout.flush()

                chi2+=(pars[par_name]-par_mean)**2/par_sigma**2

        if (self.verbose):
            print("---")
            for pname in self.pname:
                print(pname,pars[pname])

            print("Chi2: ",chi2)
            sys.stdout.flush()

        return chi2

    def fastMonteCarlo(self,mig,kw):

        dic_init = self.dic_init

        ### Get parameters
        try:
            if sp.remainder(len(dic_init['fastMonteCarlo']),2)!=0: raise

            nb_fMC   = int(dic_init['fastMonteCarlo'][0])
            seed_fMC = int(dic_init['fastMonteCarlo'][1])
            sp.random.seed(seed=seed_fMC)

            nb_expected_values = sp.floor_divide(len(dic_init['fastMonteCarlo'])-2,2)
            for i in range(nb_expected_values):
                key = dic_init['fastMonteCarlo'][2*(i+1)]
                val = dic_init['fastMonteCarlo'][2*(i+1)+1]
                if len(key)>4 and key[:4]=='fix_':
                    kw[key] = bool(val)
                    kw['error_'+key[4:]] = 0.
                elif len(key)>5 and key[:5]=='free_':
                    kw[key] = bool(val)
                else:
                    val = float(val)
                    mig.values[key] = val
                    kw[key] = val
        except Exception as error :
            print('  ERROR::picca/py/picca/fitter/Chi2.py:: error in fast Monte-Carlo = ', error)
            print('  Exit')
            sys.exit(0)

        ### Get bes fit
        if not self.auto is None:
            rp  = self.auto.rp
            rt  = self.auto.rt
            z   = self.auto.z
            bestFit_auto = self.cosmo.valueAuto(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pauto})
            ### Metals
            met = None
            if not self.met is None:
                met = self.met.valueAuto(mig.values)
                bestFit_auto += met
            bestFit_auto = sp.dot(self.auto.dm,bestFit_auto)
            ### Broadband
            bb = None
            if not self.bb is None:
                p,b = self.bb.value(self.auto.da-bestFit_auto[self.auto.cuts])
                bb = self.bb(rt,rp,p)
                if self.dic_init['distort_bb_auto']:
                    bb=sp.dot(self.auto.dm,bb)
                bestFit_auto += bb
        if not self.cross is None:
            rp  = self.cross.rp
            rt  = self.cross.rt
            z   = self.cross.z
            bestFit_cross = self.cosmo.valueCross(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pcross})
            ### Metals
            met = None
            if not self.met is None:
                met = self.met.valueCross(mig.values)
                bestFit_cross += met
            bestFit_cross = sp.dot(self.cross.dm,bestFit_cross)
            ### Broadband
            bb = None
            if not self.bb_cross is None:
                p,b = self.bb_cross.value(self.cross.da-bestFit_cross[self.cross.cuts],mig.values['drp'])
                bb = self.bb_cross(rt,rp,mig.values['drp'],p)
                if self.dic_init['distort_bb_cross']:
                    bb = sp.dot(self.cross.dm,bb)
                bestFit_cross += bb
        if not self.autoQSO is None:
            rp  = self.autoQSO.rp
            rt  = self.autoQSO.rt
            z   = self.autoQSO.z
            bestFit_autoQSO = self.cosmo.valueAutoQSO(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pautoQSO})
            bestFit_autoQSO = sp.dot(self.autoQSO.dm,bestFit_autoQSO)

        ### File to save into
        output_name = dic_init['output_prefix']
        output_name += "save.pars.fastMonteCarlo"
        f = open(output_name,"w",0)
        f.write("#\n#seed = {}\n#\n".format(seed_fMC))
        for p in mig.parameters:
            f.write("{} ".format(p))
        f.write("chi2 ")
        for p in mig.parameters:
            f.write("{} ".format("err_"+p))
        f.write("err_chi2\n")

        ### Get realisation fastMonteCarlo
        for i in range(nb_fMC):

            print('  fastMonteCarlo: ', i, ' over ', nb_fMC)
            sys.stdout.flush()

            ### Get realisation fastMonteCarlo
            if not dic_init['data_auto'] is None:
                self.auto.get_realisation_fastMonteCarlo(bestFit=bestFit_auto)
            if not dic_init['data_cross'] is None:
                self.cross.get_realisation_fastMonteCarlo(bestFit=bestFit_cross)
            if not dic_init['data_autoQSO'] is None:
                self.autoQSO.get_realisation_fastMonteCarlo(bestFit=bestFit_autoQSO)

            ### Fit
            mig_fMC = iminuit.Minuit(self,forced_parameters=self.pname,errordef=1,print_level=0,**kw)
            try:
                mig_fMC.migrad()
                chi2_result = mig_fMC.get_fmin().fval
            except:
                chi2_result = sp.nan

            ### Save
            for p in mig_fMC.parameters:
                f.write("{} ".format(mig_fMC.values[p]))
            f.write("{} ".format(chi2_result))
            for p in mig_fMC.parameters:
                f.write("{} ".format(mig_fMC.errors[p]))
            f.write("{}\n".format(0.))

        f.close()

        return

    def chi2Scan(self,mig,kw):
        dic_init = self.dic_init

        if len(dic_init['chi2Scan'])%4 != 0:
            print('ERROR::bin/fit:: chi2 scan syntax is incorrect')
            return

        ### Get the parameters of the scan
        dic_chi2Scan = {}
        nb_param = len(dic_init['chi2Scan'])//4
        for i in range(nb_param):
            dic_param = {}
            if not any(dic_init['chi2Scan'][i*4+0] in el for el in self.pname):
                print('  ERROR::bin/fit:: Param not fitted: ', dic_init['chi2Scan'][i*4+0])
                continue

            par_name   = dic_init['chi2Scan'][i*4+0]
            par_min    = float(dic_init['chi2Scan'][i*4+1])
            par_max    = float(dic_init['chi2Scan'][i*4+2])
            par_nb_bin = int(dic_init['chi2Scan'][i*4+3])

            dic_param['name']   = par_name
            dic_param['min']    = par_min
            dic_param['max']    = par_max
            dic_param['nb_bin'] = par_nb_bin
            dic_param['grid']   = sp.linspace(par_min,par_max,num=par_nb_bin,endpoint=True)

            dic_chi2Scan[str(i)] = dic_param
            kw['fix_'+dic_param['name']] = True
            kw['error_'+dic_param['name']] = 0.

        output_name = dic_init['output_prefix']
        for i in range(nb_param):
            output_name += "."+dic_init['chi2Scan'][i*4]
        output_name += ".scan.dat"
        f = open(output_name,"w",0)
        for p in mig.parameters:
            f.write("{} ".format(p))
        f.write("chi2\n")
        ### 1D
        if (nb_param==1):
            par_name   = dic_chi2Scan[str(0)]['name']
            grid       = dic_chi2Scan[str(0)]['grid']
            par_nb_bin = dic_chi2Scan[str(0)]['nb_bin']

            for i in range(par_nb_bin):
                kw[par_name] = grid[i]
                mig = iminuit.Minuit(self,forced_parameters=self.pname,errordef=1,print_level=0,**kw)
                try:
                    mig.migrad()
                    chi2_result = mig.get_fmin().fval
                except:
                    chi2_result = sp.nan
                for p in mig.parameters:
                    f.write("{} ".format(mig.values[p]))
                f.write("{}\n".format(chi2_result))

        ### 2D
        if (nb_param==2):
            par_min1    = dic_chi2Scan[str(0)]['min']
            par_max1    = dic_chi2Scan[str(0)]['max']
            par_name1   = dic_chi2Scan[str(0)]['name']
            grid1       = dic_chi2Scan[str(0)]['grid']
            par_nb_bin1 = dic_chi2Scan[str(0)]['nb_bin']
            par_step1   = (par_max1-par_min1)/par_nb_bin1

            par_name2   = dic_chi2Scan[str(1)]['name']
            grid2       = dic_chi2Scan[str(1)]['grid']
            par_nb_bin2 = dic_chi2Scan[str(1)]['nb_bin']

            idx1 = []
            for i in range(par_nb_bin1):
                idx1 = sp.append(idx1,(par_min1+(i+0.5)*par_step1)*sp.ones(par_nb_bin2))
            idx2 = []
            for i in range(par_nb_bin1):
                idx2 = sp.append(idx2,grid2)

            for i in range(par_nb_bin1):
                for j in range(par_nb_bin2):
                    kw[par_name1] = grid1[i]
                    kw[par_name2] = grid2[j]
                    mig = iminuit.Minuit(self,forced_parameters=self.pname,errordef=1,print_level=self.verbose,**kw)
                    try:
                        mig.migrad()
                        chi2_result = mig.get_fmin().fval
                    except:
                        chi2_result = sp.nan
                    for p in mig.parameters:
                        f.write("{} ".format(mig.values[p]))
                    f.write("{}\n".format(chi2_result))

        f.close()
        return
    def export(self,mig,param):
        prefix=param.dic_init['output_prefix']
        if self.auto != None:

            rp = self.auto.rp
            rt = self.auto.rt
            z  = self.auto.z

            ### Save all bins
            fit = self.cosmo.valueAuto(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pauto})

            ### Metals
            met = None
            if not self.met is None:
                met = self.met.valueAuto(mig.values)
                fit += met
                met = sp.dot(self.auto.dm,met)
            fit = sp.dot(self.auto.dm,fit)

            ### Broadband
            bb = None
            if not self.bb is None:
                p,b = self.bb.value(self.auto.da-fit[self.auto.cuts])
                bb = self.bb(rt,rp,p)
                if self.dic_init['distort_bb_auto']:
                    bb=sp.dot(self.auto.dm,bb)

                fit += bb

            ## Side bands
            pars_sb={p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pauto}
            pars_sb['bao_amp']=0.
            sb=self.cosmo.valueAuto(rp,rt,z,pars_sb)
            sb=sp.dot(self.auto.dm,sb)

            ### Other attributes
            index = sp.arange(len(rp))
            da    = self.auto.da_all
            err   = sp.sqrt(sp.diagonal(self.auto.co_all))

            ### Save all bins
            self._exp_res(prefix+"auto_all",index,rp,rt,z,da,err,fit,met=met,bb=bb,sb=sb)

            ### Apply cuts
            cuts  = self.auto.cuts
            index = index[cuts]
            rp    = rp[cuts]
            rt    = rt[cuts]
            z     = z[cuts]
            da    = da[cuts]
            err   = sp.sqrt(sp.diagonal(self.auto.co))
            fit   = fit[cuts]
            if not met is None:
                met = met[cuts]
            if not bb is None:
                bb = bb[cuts]
            sb    = sb[cuts]

            ### Save only fitted bins
            self._exp_res(prefix+"auto",index,rp,rt,z,da,err,fit,bb=bb,met=met,sb=sb)
        if self.cross != None:

            rp = self.cross.rp
            rt = self.cross.rt
            z  = self.cross.z

            ### Fit
            fit = self.cosmo.valueCross(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pcross})

            ### Metals
            met = None
            if not self.met is None:
                met = self.met.valueCross(mig.values)
                fit += met
            fit = sp.dot(self.cross.dm,fit)

            ### Broadband
            bb = None
            if not self.bb_cross is None:
                p,b = self.bb_cross.value(self.cross.da-fit[self.cross.cuts],mig.values['drp'])
                bb = self.bb_cross(rt,rp,mig.values['drp'],p)
                if self.dic_init['distort_bb_cross']:
                    bb = sp.dot(self.cross.dm,bb)
                fit += bb

            ### Side_bands
            pars_sb = {p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pcross}
            pars_sb['bao_amp']=0.
            sb = self.cosmo.valueCross(rp,rt,z,pars_sb)
            sb = sp.dot(self.cross.dm,sb)

            ### Other attributes
            index = sp.arange(len(rp))
            da    = self.cross.da_all
            err   = sp.sqrt(sp.diagonal(self.cross.co_all))

            ### Save all bins
            self._exp_res(prefix+"cross_all",index,rp,rt,z,da,err,fit,met=met,bb=bb,sb=sb)

            ### Apply cuts
            cuts  = self.cross.cuts
            index = index[cuts]
            rp    = rp[cuts]
            rt    = rt[cuts]
            z     = z[cuts]
            da    = da[cuts]
            err   = sp.sqrt(sp.diagonal(self.cross.co))
            fit   = fit[cuts]
            if not met is None:
                met = met[cuts]
            if not bb is None:
                bb = bb[cuts]
            sb    = sb[cuts]

            ### Save only fitted bins
            self._exp_res(prefix+"cross",index,rp,rt,z,da,err,fit,met=met,bb=bb,sb=sb)
        if self.autoQSO != None:
            rp=self.autoQSO.rp
            rt=self.autoQSO.rt
            z= self.autoQSO.z

            ### Save all bins
            index=sp.arange(len(rp))
            da=self.autoQSO.da_all
            err=sp.sqrt(sp.diagonal(self.autoQSO.co_all))
            fit=self.cosmo.valueAutoQSO(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pautoQSO})
            fit=sp.dot(self.autoQSO.dm,fit)
            self._exp_res(prefix+"autoQSO_all",index,rp,rt,z,da,err,fit)

            ### Save only fitted bins
            cuts=self.autoQSO.cuts
            index=sp.arange(len(cuts))[cuts]
            da=self.autoQSO.da
            err=sp.sqrt(sp.diagonal(self.autoQSO.co))
            fit=self.cosmo.valueAutoQSO(rp,rt,z,{p:mig.values[p] for p in self.cosmo.pglob+self.cosmo.pautoQSO})
            fit=sp.dot(self.autoQSO.dm,fit)
            rp=rp[cuts]
            rt=rt[cuts]
            z=z[cuts]
            fit=fit[cuts]
            self._exp_res(prefix+"autoQSO",index,rp,rt,z,da,err,fit)


        self._exp_pars(prefix,mig)


        ### Save <prefix>fit.config.input
        self._exp_config('fit.config.input',param,param.dic_init)

        ### Save <prefix>fit.config.output
        dic_param = copy.deepcopy(param.dic_init)
        for i in self.pname:
            dic_param[i] = mig.values[i]
        self._exp_config('fit.config',param,dic_param)

        ### Save correlation matrix of parameters
        self._exp_correlation_matrix_parameters(mig,param)

        ### Save the minos errors if they were calculated.
        self._exp_minos(mig,param)

        ### Save the parameters of minuit at the minimum
        self._exp_fmin(mig,param)

    def _exp_minos(self,mig,param):
        if param.dic_init['minos'] is None: return

        ## minos keys. Spell them out to have them in this order
        minos_keys = ['lower','upper','lower_valid','upper_valid','min','is_valid',
            'lower_new_min','upper_new_min',
            'at_lower_limit','at_upper_limit',
            'at_lower_max_fcn', 'at_upper_max_fcn','nfcn']
        minos_values_default = [0.,0.,False,False,0.,False,True,True,True,True,True,True,0]
        dic_default = {}
        for i in range(len(minos_keys)):
            dic_default[minos_keys[i]] = minos_values_default[i]
        minos_errors = mig.get_merrors()

        ### Get list of parameters with minos errors
        ###    and parameters in dic_init['minos']
        list_parameters_minos = minos_errors.keys()
        for i in param.dic_init['minos']:
            if (i=='_all_'): continue
            if not any(i in el for el in list_parameters_minos):
                list_parameters_minos += [i]

        f=open(param.dic_init['output_prefix']+'minos_save.pars','w')
        for key in minos_keys:
            f.write(' '+key)
        f.write('\n')
        for par in list_parameters_minos:
            f.write(par+' ')
            if any(par in el for el in minos_errors): dic = minos_errors[par]
            else: dic = dic_default
            for key in minos_keys:
                if isinstance(dic[key],types.BooleanType): f.write('%i ' % (dic[key]))
                elif isinstance(dic[key],types.IntType): f.write('%i ' % (dic[key]))
                else: f.write('{:f}'.format(dic[key])+' ')
            f.write('\n')
        f.close()

        return

    def _exp_fmin(self,mig,param):

        keys = ['hesse_failed',
        'has_reached_call_limit',
        'has_accurate_covar',
        'has_posdef_covar',
        'up',
        'edm',
        'is_valid',
        'is_above_max_edm',
        'has_covariance',
        'has_made_posdef_covar',
        'has_valid_parameters',
        'fval',
        'nfcn']

        max_len = 0
        for k in keys:
            max_len = max(max_len,len(k))

        fmin = mig.get_fmin()
        path_to_save = param.dic_init['output_prefix']+'minuit.fmin'
        f = open(path_to_save,'w')

        for k in keys:
            f.write(k.ljust(max_len)+' ')
            f.write(str(fmin[k])+'\n')
        f.close()

        return

    def _exp_res(self,prefix,index,rp,rt,z,da,err,fit,met=None,bb=None,sb=None):

        f=open(prefix+"_residuals.dat","w")
        nbins=len(da)

        r=sp.sqrt(rp**2+rt**2)
        mu = rp/r

        for i in range(nbins):
            f.write(str(index[i])+" "+str(rp[i])+" "+str(rt[i])+" "+str(z[i])+" "+str(r[i])+" "+str(mu[i])+" "+str(z[i])+" "+str(fit[i])+" "+str(da[i])+" "+str(err[i]))
            if not bb is None:
                f.write(" "+str(bb[i]))
            else:
                f.write(" 0")
            if not met is None:
                f.write(" "+str(met[i]))
            else:
                f.write(" 0")
            if not sb is None:
                f.write(" "+str(sb[i]))
            else:
                f.write(" 0")

            f.write("\n")

        f.close()

    def _exp_pars(self,prefix,mig):
        f=open(prefix+"save.pars","w")
        for i,p in enumerate(self.pname):
            err = mig.errors[p]
            if mig.is_fixed(p): err=0.
            f.write(str(i)+" "+p+" "+str(mig.values[p])+" "+str(err)+"\n")
        f.close()

        f=open(prefix+"save.broadband.pars","w")
        i = 0
        if not self.bb is None:
            for i in range(self.bb.npar):
                f.write(str(i)+' '+self.bb.par_name[i]+' '+str(self.bb.pars[i])+" 0.\n")
        if not self.bb_cross is None:
            for j in range(self.bb_cross.npar):
                f.write(str(i+j)+' '+self.bb_cross.par_name[j]+' '+str(self.bb_cross.pars[j])+" 0.\n")
        f.close()

        nb_fit = 0
        if not self.auto is None:
            nb_fit    += 1
            f=open(prefix+"auto_fit.chisq","w")
            chi2=self.chi2_auto(mig.values)
            ndata=len(self.auto.da)
            npars=len({p for p in self.cosmo.pglob+self.cosmo.pauto if p in mig.list_of_vary_param()})
            if not self.met is None:
                npars+=len([p for p in self.met.pname if p in mig.list_of_vary_param()])
            if not self.bb is None:
                npars+= self.bb.npar
            f.write(str(chi2)+" "+str(ndata)+" "+str(npars)+" "+str(1-stats.chi2.cdf(chi2,ndata-npars))+"\n")
            f.close()

        if not self.cross is None:
            nb_fit   += 1
            f=open(prefix+"cross_fit.chisq","w")
            chi2=self.chi2_cross(mig.values)
            ndata=len(self.cross.da)
            npars=len([p for p in self.cosmo.pglob+self.cosmo.pcross if p in mig.list_of_vary_param()])
            if not self.met is None:
                npars+=len([p for p in self.met.pname if p in mig.list_of_vary_param()])
            if not self.bb_cross is None:
                npars+= self.bb_cross.npar
            f.write(str(chi2)+" "+str(ndata)+" "+str(npars)+" "+str(1-stats.chi2.cdf(chi2,ndata-npars))+"\n")
            f.close()

        if not self.autoQSO is None:
            nb_fit += 1
            f=open(prefix+"autoQSO_fit.chisq","w")
            chi2=self.chi2_autoQSO(mig.values)
            ndata=len(self.autoQSO.da)
            npars=len([p for p in self.cosmo.pglob+self.cosmo.pautoQSO if p in mig.list_of_vary_param()])
            f.write(str(chi2)+" "+str(ndata)+" "+str(npars)+" "+str(1-stats.chi2.cdf(chi2,ndata-npars))+"\n")
            f.close()

        f=open(prefix+"combined_fit.chisq","w")
        chi2  = mig.get_fmin()['fval']
        ndata = 0
        if not self.auto is None: ndata    += len(self.auto.da)
        if not self.cross is None: ndata   += len(self.cross.da)
        if not self.autoQSO is None: ndata += len(self.autoQSO.da)
        npars = len(mig.list_of_vary_param())
        if not self.bb is None:
            npars+= self.bb.npar
        if not self.bb_cross is None:
            npars+= self.bb_cross.npar
        f.write(str(chi2)+" "+str(ndata)+" "+str(npars)+" "+str(1-stats.chi2.cdf(chi2,ndata-npars))+"\n")
        f.close()

    def _exp_correlation_matrix_parameters(self,mig,param):

        path_to_save = param.dic_init['output_prefix']+'save.pars.cor'

        matrix = sp.array(mig.matrix(correlation=True))
        fitted_parameters = sp.array(mig.list_of_vary_param())

        f = open(path_to_save,'w')
        f.write(' --- ')
        for i in range(fitted_parameters.size):
            f.write(fitted_parameters[i]+' ')
        f.write("\n")
        for i in range(fitted_parameters.size):
            f.write(fitted_parameters[i]+' ')
            for j in range(fitted_parameters.size):
                f.write(str(matrix[i,j])+' ')
            f.write("\n")
        f.close()

        return

    def _exp_config(self,suffix,param,dic_param):

        max_len = 0
        for i in dic_param:
            max_len = max(max_len,len(i))

        with open(dic_param['output_prefix']+suffix, 'w') as configfile:

            configfile.write('\n[PARAMETER]\n')

            configfile.write('\n#String\n')
            for i in sorted(param.dic_init_string):
                configfile.write(str(i).ljust(max_len) + ' = ' + str(dic_param[i]) + '\n')

            configfile.write('\n#List of strings\n')
            for i in sorted(param.dic_init_list_string):
                value = dic_param[i]
                if not value is None and value != 'None':
                    value = ' '.join(value)
                configfile.write(str(i).ljust(max_len) + ' = ' + str(value) + '\n')

            configfile.write('\n#Bool\n')
            for i in sorted(param.dic_init_bool):
                if (dic_param[i]):
                    configfile.write(str(i).ljust(max_len) + ' = ' + str(dic_param[i]) + '\n')
                else:
                    configfile.write('#' + str(i).ljust(max_len) + ' = ' + str(dic_param[i]) + '\n')

            configfile.write('\n#Int\n')
            for i in sorted(param.dic_init_int):
                configfile.write(str(i).ljust(max_len) + ' = ' + str(dic_param[i]) + '\n')

            configfile.write('\n#Float\n')
            for i in sorted(param.dic_init_float):
                if (dic_param[i]>=0.):
                    configfile.write(str(i).ljust(max_len) + ' = ' + str(dic_param[i]) + '\n')
                else:
                    str_param = '{:.16f}'.format(dic_param[i])
                    configfile.write(str(i).ljust(max_len) + ' = ' + str_param + '\n')

            configfile.write('\n')








