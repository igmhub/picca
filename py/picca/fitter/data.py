import astropy.io.fits as pyfits
import numpy as np
from scipy import linalg
import copy

class data:
    def __init__(self,kind,dic_init):

        self.kind=kind
        if (self.kind=='auto'):
            fname = dic_init['data_auto']
        elif (self.kind=='cross'):
            fname = dic_init['data_cross']
        elif (self.kind=='autoQSO'):
            fname = dic_init['data_autoQSO']

        rmin     = dic_init['rmin']
        rmax     = dic_init['rmax']
        mumin     = dic_init['mumin']
        mumax     = dic_init['mumax']
        bin_size = dic_init['bin_size']

        h = pyfits.open(fname)
        da = h[1].data.DA
        co = h[1].data.CO

        self.dm = h[1].data.DM
        self.rt = h[1].data.RT
        self.rp = h[1].data.RP
        self.z = h[1].data.Z
        self.da_all = copy.deepcopy(da)
        self.co_all = copy.deepcopy(co)

        ### Get the center of the bins from the regular grid
        bin_center_rt = np.zeros(self.rt.size)
        bin_center_rp = np.zeros(self.rp.size)
        for i in np.arange(-self.rt.size-1,self.rt.size+1,1).astype('int'):
            bin_center_rt[ np.logical_and( self.rt>=bin_size*i, self.rt<bin_size*(i+1.) ) ] = bin_size*(i+0.5)
            bin_center_rp[ np.logical_and( self.rp>=bin_size*i, self.rp<bin_size*(i+1.) ) ] = bin_size*(i+0.5)

        r = np.sqrt(bin_center_rt**2 + bin_center_rp**2)
        mu=bin_center_rp/r

        cuts = (r>rmin) & (r<rmax) & (mu>=mumin) & (mu<=mumax)
        if np.isfinite(dic_init['r_per_min']):
            cuts = cuts & (bin_center_rt > dic_init['r_per_min'])
        if np.isfinite(dic_init['r_per_max']):
            cuts = cuts & (bin_center_rt < dic_init['r_per_max'])
        if np.isfinite(dic_init['r_par_min']):
            cuts = cuts & (bin_center_rp > dic_init['r_par_min'])
        if np.isfinite(dic_init['r_par_max']):
            cuts = cuts & (bin_center_rp < dic_init['r_par_max'])

        co=co[:,cuts]
        co=co[cuts,:]
        da=da[cuts]

        self.cuts = cuts
        self.da=da
        self.co=co
        self.ico=np.linalg.inv(co)

    def get_realisation_fastMonteCarlo(self,bestFit):

        if not hasattr(self,'cho_co'):
            self.cho_co = np.linalg.cholesky(self.co_all,lower=True)

        self.da_all[:] = bestFit
        rand = np.random.normal(loc=0.0, scale=1.0, size=self.da_all.size)
        self.da_all += np.dot(self.cho_co,rand)
        self.da = self.da_all[(self.cuts)]

        return
