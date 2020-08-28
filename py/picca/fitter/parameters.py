import numpy as np
import copy
import sys
import subprocess

from picca.utils import userprint


class parameters:

    def __init__(self):

        self.dic_init_float = {
            'bias_lya*(1+beta_lya)' : -0.336,
            'beta_lya'          : 1.4,
            'ap'                : 1.,
            'at'                : 1.,
            'aiso'                : 1.,
            '1+epsilon'         : 1.,
            'bin_size'          : 4.,
            'alpha_lya'             : 2.9,
            'Lpar_auto'         : 4.,
            'Lper_auto'         : 4.,
            'SigmaNL_perp'      : 0.,
            '1+f'               : 1.,
            'bias_qso'          : 3.,
            'growth_rate'       : 0.962524,
            'drp'               : 0.,
            'Lpar_cross'        : 4.,
            'Lper_cross'        : 4.,
            'qso_evol_0'        : 0.53,
            'qso_evol_1'        : 0.289,
            'bias_lls'          : -0.01,
            'beta_lls'          : 0.6,
            'L0_lls'            : 20.,
            'bias_gamma'        : 0.13,
            'bias_prim'         : -2./3,
            'lambda_uv'            : 300.,
            'qso_metal_boost'   : 1.,
            'sigma_minos'       : 1.,
            'Lpar_autoQSO'      : 4.,
            'Lper_autoQSO'      : 4.,
            'migrad_tol'        : 0.1,
            'sigma_velo_gauss'  : 0.0,
            'sigma_velo_lorentz' : 0.0,
            'bao_amp'          : 1.0,
            'bias_lya_peak'    : -0.14,
            'beta_lya_peak'    : 1.4,

            'qso_rad_strength'  : 0.42,
            'qso_rad_asymmetry' : 0.5,
            'qso_rad_lifetime'  : 29.0,
            'qso_rad_decrease'  : 244.0,

            'rmin'             : 10.,
            'rmax'             : 180.,
            'mumin'             : -1.,
            'mumax'             : 1.,
            'r_per_min'        : np.nan,
            'r_per_max'        : np.nan,
            'r_par_min'        : np.nan,
            'r_par_max'        : np.nan,
        }
        self.dic_init_int = {
            'ell_max'           : 6,

            'bb_ell_min'        : 0,
            'bb_ell_max'        : 6,
            'bb_ell_step'       : 2,

            'bb_i_max'          : 3,
            'bb_i_min'          : 0,
            'bb_i_step'         : 1,

        }
        self.dic_init_bool = {
            'hcds'              : False,
            'uv'                : False,
            'debug'             : False,
            'verbose'           : False,
            'bb_rmu'            : False,
            'bb_rPerp_rParal'   : False,
            '2d'                : False,
            'different_drp'     : False,
            'velo_gauss'        : False,
            'velo_lorentz'      : False,
            'distort_bb_auto'   : False,
            'distort_bb_cross'  : False,
            'fix_bias_beta_peak' : False,
            'fit_aiso'          : False,
            'fit_qso_radiation_model' : False,
            'metal_dmat'        : False,
            'metal_xdmat'        : False,
            'no_hesse'          : False,
            'hcds_mets'          : False,
        }
        self.dic_init_string = {
            'model'             : None,
            'dnl_model'         : None,
            'metal_prefix'      : None,
            'output_prefix'     : "./",
            'data_auto'         : None,
            'data_cross'        : None,
            'data_autoQSO'      : None,
            'QSO_evolution'     : None,
        }
        self.dic_init_list_string = {
            'metals'            : None,
            'fix'               : None,
            'free'              : None,
            'minos'             : None,
            'chi2Scan'          : None,
            'gaussian_prior'    : None,
            'limit'             : None,
            'fastMonteCarlo'    : None,
        }

        self.dic_init = {}
        self.dic_init.update(self.dic_init_float)
        self.dic_init.update(self.dic_init_int)
        self.dic_init.update(self.dic_init_string)
        self.dic_init.update(self.dic_init_list_string)
        self.dic_init.update(self.dic_init_bool)

        help_bool = {
            'hcds'              : "use hcds in the fit",
            'uv'                : "use uv in the fit",
            'debug'             : "debug",
            'verbose'           : "verbose",
            'different_drp'     : "All metals-QSO correlation have different 'drp'",
            'velo_gauss'        : "Use a Gaussian model for the QSO velocity dispersion",
            'velo_lorentz'      : "Use a Lorentzian model for the QSO velocity dispersion",
            'fit_qso_radiation_model' : "Fit for the effect of the quasar lifetime and UV emission of the neutral hydrogen density",
            'metal_dmat' : "use metal distorsion matrix instead of templates",
        }
        help_string = {
            'model'             : "prefix to the fiducial P(k) file",
            'dnl_model'         : "Type of non-linear correction model ('mcdonald' or 'arinyo')",
            'metal_prefix'      : "prefix to the metal template files",
            'output_prefix'     : "prefix for the output",
            'data_auto'         : "prefix to the data file (auto Lya)",
            'data_cross'        : "prefix to the data file (cross Lya-QSO)",
            'data_autoQSO'      : "prefix to the data file (auto QSO)",
            'QSO_evolution'     : "Type of evolution for the QSO bias (nothing or 'croom' )",
        }
        help_list_string = {
            'metals'            : "prefix to the metal template files",
            'fix'               : "list of variables to fix to their initial values",
            'free'              : "list of variables to free, overwrites fix",
            'minos'             : "list of variables to get minos error from. Setting to '_all_' gets minos errors for all free parameters.",
            'chi2Scan'          : "performs a scan on the given parameters, syntaxe is 'parameter_name min max nb_bin' for each parameters",
            'gaussian_prior'    : "<var> <mean> <sigma> adds a gaussian prior to the variable var of the given mean and sigma",
            'fastMonteCarlo'    : "<nb_real> <seed> ['optional' <expected_value> <value>, ...] for realisation of fast monte-carlo",
            'no_hesse'          : 'do not do Hesse',
        }

        self.help = copy.deepcopy(self.dic_init)
        for i in self.dic_init:
            self.help[i] = i
        for i in help_bool:
            self.help[i] = help_bool[i]
        for i in help_string:
            self.help[i] = help_string[i]
        for i in help_list_string:
            self.help[i] = help_list_string[i]

        return
    def set_parameters_from_parser(self,args,unknown):

        dic_arg = vars(args)
        for i in dic_arg:
            value = dic_arg[i]
            if value=='None' or value=='' or value is None: value = None
            elif (i in self.dic_init_list_string):
                if len(value)==0 or value[0]=='' or value[0]=='None':value=None
                else:
                    tmp = []
                    for j in range(len(value)):
                        tmp += [ str(el) for el in value[j].split() ]
                    value = tmp
            self.dic_init[i] = value

        dic_unknown = self._format_unknown(unknown=unknown)
        self._set_metals(dic_unknown=dic_unknown)

        return
    def _format_unknown(self,unknown):

        dic_unknown = {}
        i = 0
        while (i<len(unknown)):
            try:
                if (unknown[i][:2]=='--'):
                    dic_unknown[unknown[i][2:]] = float(unknown[i+1])
                    i += 2
                else:
                    userprint('  picca/py/picca/fitter/parameters.py:: unknown entry = ', unknown[i])
                    userprint('  Exit')
                    sys.exit(0)
                    i += 1
            except:
                userprint('  picca/py/picca/fitter/parameters.py:: unknown entry = ', unknown[i])
                userprint('  Exit')
                sys.exit(0)
                i += 1

        return dic_unknown
    def _set_metals(self,dic_unknown):

        metals_default = {}
        metals_default['bias'] = -0.01
        metals_default['beta'] = 0.5
        metals_default['alpha'] = 1.
        metals_default['drp']  = 0.

        if self.dic_init['metals'] is None:
            if (len(dic_unknown)!=0):
                userprint('  picca/py/picca/fitter/parameters.py:: entries not metal = ', dic_unknown)
                userprint('  Exit')
                sys.exit(0)
            return

        ### what is in dic_unknow ?
        for i in dic_unknown:
            is_a_metal = False
            for met in self.dic_init['metals']:
                for par in metals_default:
                    if (par+'_'+met == i):
                        is_a_metal = True
            if not is_a_metal:
                userprint('  picca/py/picca/fitter/parameters.py:: entry not metal = ', i)
                userprint('  Exit')
                sys.exit(0)

        ### Set values
        for i in self.dic_init['metals']:
            if not any(i in el for el in self.dic_init):

                for par in metals_default:

                    if any(par+'_'+i in el for el in dic_unknown):
                        self.dic_init_float[par+'_'+i] = dic_unknown[par+'_'+i]
                        self.dic_init[par+'_'+i] = dic_unknown[par+'_'+i]
                    else:
                        self.dic_init_float[par+'_'+i] = metals_default[par]
                        self.dic_init[par+'_'+i] = metals_default[par]

        return
    def test_init_is_valid(self):

        if self.dic_init['model'] is None:
            userprint('  picca/py/picca/fitter/parameters.py::  No model file.')
            userprint('  Exit')
            sys.exit(0)

        if self.dic_init['data_auto'] is None and self.dic_init['data_cross'] is None and self.dic_init['data_autoQSO'] is None:
            userprint('  picca/py/picca/fitter/parameters.py::  No data file.')
            userprint('  Exit')
            sys.exit(0)

        try:
            path_to_save = self.dic_init['output_prefix']+'test_if_output_possible'
            np.savetxt(path_to_save,np.ones(10))
            command = 'rm '+path_to_save
            subprocess.call(command, shell=True)
        except:
            userprint('  picca/py/picca/fitter/parameters.py::  Impossible to save in : ', self.dic_init['output_prefix'])
            userprint('  Exit')
            sys.exit(0)

        return
