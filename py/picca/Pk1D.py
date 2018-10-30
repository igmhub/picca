from __future__ import print_function

import scipy as sp
from scipy.fftpack import fft

from picca import constants
from picca.utils import print


def split_forest(nb_part,dll,ll,de,diff,iv,first_pixel):

    ll_limit=[ll[first_pixel]]
    nb_bin= (len(ll)-first_pixel)//nb_part

    m_z_arr = []
    ll_arr = []
    de_arr = []
    diff_arr = []
    iv_arr = []

    ll_c = ll.copy()
    de_c = de.copy()
    diff_c = diff.copy()
    iv_c = iv.copy()

    for p in range(1,nb_part) :
        ll_limit.append(ll[nb_bin*p+first_pixel])

    ll_limit.append(ll[len(ll)-1]+0.1*dll)

    for p in range(nb_part) :

        selection = (ll_c>= ll_limit[p]) & (ll_c<ll_limit[p+1])

        ll_part = ll_c[selection]
        de_part = de_c[selection]
        diff_part = diff_c[selection]
        iv_part = iv_c[selection]

        lam_lya = constants.absorber_IGM["LYA"]
        m_z = (sp.power(10.,ll_part[len(ll_part)-1])+sp.power(10.,ll_part[0]))/2./lam_lya -1.0

        m_z_arr.append(m_z)
        ll_arr.append(ll_part)
        de_arr.append(de_part)
        diff_arr.append(diff_part)
        iv_arr.append(iv_part)

    return m_z_arr,ll_arr,de_arr,diff_arr,iv_arr

def rebin_diff_noise(dll,ll,diff):

    crebin = 3
    if (diff.size < crebin):
        print("Warning: diff.size too small for rebin")
        return diff
    dll2 = crebin*dll

    # rebin not mixing pixels separated by masks
    bin2 = sp.floor((ll-ll.min())/dll2+0.5).astype(int)

    # rebin regardless of intervening masks
    # nmax = diff.size//crebin
    # bin2 = sp.zeros(diff.size)
    # for n in range (1,nmax +1):
    #     bin2[n*crebin:] += sp.ones(diff.size-n*crebin)

    cdiff2 = sp.bincount(bin2.astype(int),weights=diff)
    civ2 = sp.bincount(bin2.astype(int))
    w = (civ2>0)
    if (len(civ2) == 0) :
        print( "Error: diff size = 0 ",diff)
    diff2 = cdiff2[w]/civ2[w]*sp.sqrt(civ2[w])
    diffout = sp.zeros(diff.size)
    nmax = len(diff)//len(diff2)
    for n in range (nmax+1) :
        lengthmax = min(len(diff),(n+1)*len(diff2))
        diffout[n*len(diff2):lengthmax] = diff2[:lengthmax-n*len(diff2)]
        sp.random.shuffle(diff2)

    return diffout


def fill_masked_pixels(dll,ll,delta,diff,iv,no_apply_filling):


    if no_apply_filling : return ll,delta,diff,iv,0


    ll_idx = ll.copy()
    ll_idx -= ll[0]
    ll_idx /= dll
    ll_idx += 0.5
    index =sp.array(ll_idx,dtype=int)
    index_all = range(index[-1]+1)
    index_ok = sp.in1d(index_all, index)

    delta_new = sp.zeros(len(index_all))
    delta_new[index_ok]=delta

    ll_new = sp.array(index_all,dtype=float)
    ll_new *= dll
    ll_new += ll[0]

    diff_new = sp.zeros(len(index_all))
    diff_new[index_ok]=diff

    iv_new = sp.ones(len(index_all))
    iv_new *=0.0
    iv_new[index_ok]=iv

    nb_masked_pixel=len(index_all)-len(index)

    return ll_new,delta_new,diff_new,iv_new,nb_masked_pixel

def compute_Pk_raw(dll,delta,ll):

    #   Length in km/s
    length_lambda = dll*constants.speed_light/1000.*sp.log(10.)*len(delta)

    # make 1D FFT
    nb_pixels = len(delta)
    nb_bin_FFT = nb_pixels//2 + 1
    fft_a = fft(delta)

    # compute power spectrum
    fft_a = fft_a[:nb_bin_FFT]
    Pk = (fft_a.real**2+fft_a.imag**2)*length_lambda/nb_pixels**2
    k = sp.arange(nb_bin_FFT,dtype=float)*2*sp.pi/length_lambda

    return k,Pk


def compute_Pk_noise(dll,iv,diff,ll,run_noise):

    nb_pixels = len(iv)
    nb_bin_FFT = nb_pixels//2 + 1

    nb_noise_exp = 10
    Pk = sp.zeros(nb_bin_FFT)
    err = sp.zeros(nb_pixels)
    w = iv>0
    err[w] = 1.0/sp.sqrt(iv[w])

    if (run_noise) :
        for iexp in range(nb_noise_exp):
            delta_exp= sp.zeros(nb_pixels)
            delta_exp[w] = sp.random.normal(0.,err[w])
            k_exp,Pk_exp = compute_Pk_raw(dll,delta_exp,ll)
            Pk += Pk_exp

        Pk /= float(nb_noise_exp)

    k_diff,Pk_diff = compute_Pk_raw(dll,diff,ll)

    return Pk,Pk_diff

def compute_cor_reso(delta_pixel,mean_reso,k):

    nb_bin_FFT = len(k)
    cor = sp.ones(nb_bin_FFT)

    sinc = sp.ones(nb_bin_FFT)
    sinc[k>0.] =  (sp.sin(k[k>0.]*delta_pixel/2.0)/(k[k>0.]*delta_pixel/2.0))**2

    cor *= sp.exp(-(k*mean_reso)**2)
    cor *= sinc
    return cor


class Pk1D :

    def __init__(self,ra,dec,zqso,mean_z,plate,mjd,fiberid, msnr, mreso,
                 k,Pk_raw,Pk_noise,cor_reso,Pk, nb_mp, Pk_diff=None):

        self.ra = ra
        self.dec = dec
        self.zqso = zqso
        self.mean_z = mean_z
        self.mean_snr = msnr
        self.mean_reso = mreso
        self.nb_mp = nb_mp

        self.plate = plate
        self.mjd = mjd
        self.fid = fiberid
        self.k = k
        self.Pk_raw = Pk_raw
        self.Pk_noise = Pk_noise
        self.cor_reso = cor_reso
        self.Pk = Pk
        self.Pk_diff = Pk_diff


    @classmethod
    def from_fitsio(cls,hdu):

        """
        read Pk1D from fits file
        """

        hdr = hdu.read_header()

        ra = hdr['RA']
        dec = hdr['DEC']
        zqso = hdr['Z']
        mean_z = hdr['MEANZ']
        mean_reso = hdr['MEANRESO']
        mean_SNR = hdr['MEANSNR']
        plate = hdr['PLATE']
        mjd = hdr['MJD']
        fid = hdr['FIBER']
        nb_mp = hdr['NBMASKPIX']

        data = hdu.read()
        k = data['k'][:]
        Pk = data['Pk'][:]
        Pk_raw = data['Pk_raw'][:]
        Pk_noise = data['Pk_noise'][:]
        cor_reso = data['cor_reso'][:]
        Pk_diff = data['Pk_diff'][:]

        return cls(ra,dec,zqso,mean_z,plate,mjd,fid, mean_SNR, mean_reso, k,Pk_raw,Pk_noise,cor_reso, Pk, nb_mp, Pk_diff)
