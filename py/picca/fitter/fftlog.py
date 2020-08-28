import numpy as np
import sys
from scipy.special import gamma
from scipy.interpolate import RegularGridInterpolator
import numpy.fft as fft
import time

from picca.utils import userprint

def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y = np.where(x<xp[0], yp[0]+(x-xp[0])*(yp[0]-yp[1])/(xp[0]-xp[1]), y)
    y = np.where(x>xp[-1], yp[-1]+(x-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2]), y)
    return y

def extrapolate_pk_logspace(ko, ki, pki):
    """
    Linear interpolation of pk in log,log,
    taking care of extrapolation (also linear in log,log)
    and numerical precision
    """
    logpk=extrap(np.log(ko),np.log(ki[pki>0]),np.log(pki[pki>0]))
    pk=np.zeros(logpk.shape)
    minlogpk=-np.log(np.finfo(np.float64).max)
    pk[logpk>minlogpk]=np.exp(logpk[logpk>minlogpk])
    return pk

def HankelTransform(k,a,q,mu,r0=10.,transformed_axis=0,output_r=None,output_r_power=0,n=None) :
    """
    Hankel transform, with power law biasing
    a'(r) = r**(output_r_power) integral_0^infty a(k) (kr)^q J_mu(kr) r dk
    with k logathmically spaced
    based on Hamilton 2000 FFTLog algorithm.

    Args:
         k : 1D numpy array, log spacing defining the rectangular k-grid of Pk
         a : 1D or 2D numpy array, axis to be transformed must have same size as k
         q : float or int, power of kr in transformation
         mu : float or int, parameter of Bessel function
    Options:
         r0 : float, default is 10 (units of 1/k)
         transformed_axis : int, if a is 2D, can transform along axis 0 or 1,
                            do the transform on all rows of other axis.
         output_r : 1D numpy array or None. if set, output is interpolated to
                            this array of coordinates
         output_r_power : multiply output by r**output_r_power
         n : default n=None and set to n=q , do not play with this
    Returns:
         r : 1D numpy array
         a'(r) : numpy array, 1D or 2D, depending on input a
    """
    if len(a.shape)>2 :
        userprint("not implemented for a of more than 2D")
        sys.exit(12)

    k0 = k[0]
    N  = len(k)
    L  = np.log(k.max()/k0)* N/(N-1.) ## this is important, need to have the right scale !!
    emm = N*np.fft.fftfreq(N)

    # ??? empirically I need n=q
    if n is None :
        n=q

    nout=n+output_r_power

    x=(q-n)+2*np.pi*1j*emm/L # Eq. 174

    if 1 : # choose r0 to limit ringing with the condition u(-N/2)=u(N/2), see Hamilton 2000, Eq. 186
        x0     = (q-n)+np.pi*1j*N/L
        tmp    = 1./np.pi*np.angle(2**x0*gamma((mu+1+x0)/2.)/gamma((mu+1-x0)/2.))
        number = int(np.log(k0*r0)*N/L - tmp)
        lowringing_r0 = np.exp(L/N*(tmp+number))/k0
        r0  = lowringing_r0

    um=(k0*r0)**(-2*np.pi*1j*emm/L)*2**x*(gamma((mu+1+x)/2.)/gamma((mu+1-x)/2.)) # Eq. 174
    um[0]=um[0].real

    r=r0*np.exp(-emm*L/N)
    s=np.argsort(r)
    rs=r[s] # sorted

    if len(a.shape)==1 :

        transformed=(fft.ifft(um*fft.fft(a*(k**n)))*(r**nout)).real[s]

        if output_r is not None :
            transformed=extrap(output_r,rs,transformed)

    else :


        if transformed_axis==0 :
            if output_r is  None :
                transformed=np.zeros_like(a)
                for i in range(a.shape[1]) : # I don't know how to do this at once
                    transformed[:,i]=(fft.ifft(um*fft.fft(a[:,i]*(k**n)))*(r**nout)).real[s]
            else :
               transformed=np.zeros(shape=(output_r.size,a.shape[1]),dtype=a.dtype)
               for i in range(a.shape[1]) :
                  transformed[:,i]=extrap(output_r,rs,(fft.ifft(um*fft.fft(a[:,i]*(k**n)))*(r**nout)).real[s])
        else :
            if output_r is  None :
                transformed=np.zeros_like(a)
                for i in range(a.shape[0]) :
                    transformed[i,:]=(fft.ifft(um*fft.fft(a[i,:]*(k**n)))*(r**nout)).real[s]
            else :
                transformed=np.zeros(shape=(a.shape[0],output_r.size),dtype=a.dtype)
                for i in range(a.shape[0]) :
                    transformed[i,:]=extrap(output_r,rs,(fft.ifft(um*fft.fft(a[i,:]*(k**n)))*(r**nout)).real[s])

    if output_r is None :
        return rs,transformed
    else :
        return output_r,transformed

def Pk2XiR(k,pk2d,rp,rt) :
    """
    Transforms Pk in 2D (square grid with k in log-space)
    to xi in 2D (Rectangular grid).

    Tested with nk=1024 kmin=1e-7 kmax=100. Preserves isotropy
    to a precision of 0.9% up to r=200Mpc for linear matter Pk.

    Args:
         k : 1D numpy array, log spacing defining the rectangular k-grid of Pk
         pk2d : 2D numpy array of shape (k.size,k.size)
         rp : 1D numpy array, arbitrary spacing, defining one axis of output RECTANGULAR grid, size=50 for grid 50x50
         rt : 1D numpy array, arbitrary spacing, defining one axis of output RECTANGULAR grid, size=50 for grid 50x50
    Returns:
         xi : 2D numpy array of shape (rp.size,rt.size)
    """

    start=time.time()
    # kt -> rt
    # int pk k dk J_{mu=0}
    # HankelTransform is int pk (kr)**q J_{mu}(kr) r dk
    # mu=0, q=1 , and I need to multiply the result by r**(-2)
    junk,pkxi=HankelTransform(k=k,a=pk2d,q=1,mu=0,transformed_axis=1,output_r=rt,output_r_power=-2)

    if 1 :  # to avoid aliasing
        a=np.log(k[1]/k[0])
        nk=k.size
        k=np.append(k,k[:nk//2]*np.exp(a*k.size))
        tmp=np.zeros(shape=(k.size,pkxi.shape[1]),dtype=pkxi.dtype)
        tmp[:pkxi.shape[0]]=pkxi
        pkxi=tmp


    # pk dk cos(k*r) = int pk dk J_{mu=-1/2}(k*r) * (kr)**(1/2) (pi/2)**(1/2)
    # mu=-1/2, q=1/2 , and I need to multiply the result by r**(-1) * (pi/2)**(1/2)
    junk,xi=HankelTransform(k=k,a=pkxi,q=0.5,mu=-0.5,transformed_axis=0,output_r=rp,output_r_power=-1)
    xi *= 2*np.sqrt(np.pi/2)/(2*np.pi)**2

    stop=time.time()
    userprint("fftlog.Pk2XiR done in %f sec"%(stop-start))
    return xi

def Pk2XiA(k,pk2d,rp,rt) :
    """
    Transforms Pk in 2D (square grid with k in log-space)
    to xi in 2D (on Arbitrary grid).
    Uses Pk2Xi on rectangular grid and then does a 2D interpolation.
    Use directly Pk2Xi if result is to be on rectangular grid
    Args:
         k : 1D numpy array, log spacing defining the rectangular k-grid of Pk
         pk2d : 2D numpy array of shape (k.size,k.size)
         rp : 1D numpy array, defining full set of arb. coordinates on grid, size=2500 for grid 50x50
         rt : 1D numpy array, defining full set of arb. coordinates on grid, size=2500 for grid 50x50
    Returns:
         xi : 2D numpy array of shape (rp.size,rt.size)
    """


    # define a rectangular grid
    rpr=np.linspace(np.min(rp),np.max(rp),np.sqrt(rp.size))
    rtr=np.linspace(np.min(rt),np.max(rt),np.sqrt(rp.size))
    xi=Pk2XiR(k,pk2d,rpr,rtr)


    func=RegularGridInterpolator(points=(rpr,rtr),values=xi,method="linear")
    xia=func(np.array([rp,rt]).T)
    return xia
