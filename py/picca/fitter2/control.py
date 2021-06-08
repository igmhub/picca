from __future__ import print_function, division

from . import parser, chi2

class fitter2:
    '''
    Main interface for the fitter2. Creates cf models and runs the chi2.
    All of the functionality is single core.
    '''

    def __init__(self, chi2_file):
        ''' Read the config and initialize run settings '''
        self.dic_init = parser.parse_chi2(chi2_file)
        self.control = self.dic_init['control']
        self.run_chi2 = self.control.getboolean('chi2', False)

        # Initialize the required objects
        if self.run_chi2:
            self.chi2 = chi2.chi2(self.dic_init)

    def run(self):
        ''' Run the fitter. This function only runs single core options '''

        if self.run_chi2:
            self.chi2.minimize()
            self.chi2.minos()
            self.chi2.chi2scan()
            self.chi2.fastMC()
            self.chi2.export()
        else:
            raise ValueError('You called "fitter.run()" without \
                asking for chi2. Set "chi2 = True" in [control]')
