"""This module defines some functions to blind data using strategy A"""
import numpy as np
from picca.constants import Cosmo, ABSORBER_IGM
import sys, os  # requiered to mute Cosmo and dont print the
                #   cosmology change on screen


def calcMaps(scale=.95, Om=0.315):
    """Initialize class and compute the conversion maps

    Args:
        scale: float - default: .95
            Scaling of ommega matter with respect to the fiducial cosmology
        Om: float - default: 0.315
            Omega matter in the fiducial cosmology
    Returns:
        z_grid, z_map: input and output values of mapping functions
            to interpolate and blind the qso redshift value: z' = z+zmap(z)
        lambda_grid, lambda_map: input and output values of mapping functions
            to interpolate and blind the forest lambda values: 
            lambda'= lambda*lambda_map(lambda)
    """

    ###### Definition of the fiducial cosmological model
    
    fid_Om = Om
    fid_Or = 0
    fid_Ok = 0
    fid_wl = -1
    
    cosmo_m = Cosmo(Om=fid_Om,Or=fid_Or,
                   Ok=fid_Ok,wl=fid_wl, blinding=False)
    lambda_abs = ABSORBER_IGM['LYA']
    
    ###### Definition of blind cosmological model
    
    fid_Om = Om*scale
    fid_Or = 0
    fid_Ok = 0
    fid_wl = -1
    
    cosmo_m2 = Cosmo(Om=fid_Om,Or=fid_Or,
                    Ok=fid_Ok,wl=fid_wl, blinding=False)
    lambda_abs = ABSORBER_IGM['LYA']
    
    #######
        
    z_max = 10.
    lambda_max = ( lambda_abs * (z_max + 1) ) - 1.3
    log_lambda = np.log10( np.linspace(lambda_abs + .5, lambda_max, 10000) )
    z = 10**log_lambda/lambda_abs-1.
    
    r_comov = cosmo_m.get_r_comov(z)
    r_comov2 = cosmo_m2.get_r_comov(z)
    
    z_prime = resample_flux(r_comov, r_comov2, z)
    z_mask = ( z <= 10 )

    z_grid = z[z_mask]
    z_prime = z_prime[z_mask]
    r_comov = r_comov[z_mask]
    r_comov2 = r_comov2[zmask]
    log_lambda_grid = log_lambda[zmask]
    
    log_lambda_prime = np.log10( lambda_abs * ( z_prime + 1 ) )
    z_map = -z_prime + z
    lambda_grid = 10**log_lambda_grid
    lambda_map = lambda_grid / (10**log_lambda_prime)
    
    return z_grid, z_map, lambda_grid, lambda_map

def resample_flux(x_out, x, flux, ivar=None, extrapolate=False):
    """Function to do a weighted resampling of fluxes into a new
        absissa array
        Taken from desispec.interpolation::

    Args:
        x_out: abscissa array for which the flux will be resampled
        x: original abscissa
        flux: original flux
        ivar: inverse variance weighting of the original flux,
            if not given the output flux will an interpolation
        extrapolate: if true borders will be extrapolated to 
            "improve" the edge resampling
    Returns:
        out_flux: new flux resampled to x_out, i.e. flux(x_out)
        out_ivar: -only if ivar given as arg- new ivar values 
            for out_flux

    """
    if ivar is None:
        return _unweighted_resample(x_out, x, flux, extrapolate=extrapolate)
    else:
        if extrapolate :
            raise ValueError("Cannot extrapolate ivar. Either set ivar=None and extrapolate=True or the opposite")
        a = _unweighted_resample(x_out, x, flux*ivar, extrapolate=False)
        b = _unweighted_resample(x_out, x, ivar, extrapolate=False)
        mask = (b>0)
        out_flux = np.zeros(a.shape)
        out_flux[mask] = a[mask] / b[mask]
        dx = np.gradient(x)
        dx_out = np.gradient(x_out)
        out_ivar = _unweighted_resample(x_out, x, ivar/dx)*dx_out

        return out_flux, out_ivar

def _unweighted_resample(x_out, x, flux_density, extrapolate=False) :
    """Function to do a unweighted resampling of fluxes into a new
        absissa array, works as a interpolation
        Taken from desispec.interpolation

    Args:
        x_out: abscissa array for which the flux will be resampled
        x: original abscissa
        flux_density: original flux density
        extrapolate: if true borders will be extrapolated to 
            "improve" the edge resampling
    Returns:
        histogram: new flux resampled to x_out, i.e. 
            flux_density(x_out)
    """
    ix=x
    iy=flux_density
    ox=x_out

    # boundary of output bins
    bins=np.zeros(ox.size+1)
    bins[1:-1]=(ox[:-1]+ox[1:])/2.
    bins[0]=1.5*ox[0]-0.5*ox[1]     # = ox[0]-(ox[1]-ox[0])/2
    bins[-1]=1.5*ox[-1]-0.5*ox[-2]  # = ox[-1]+(ox[-1]-ox[-2])/2

    tx=bins.copy()

    if not extrapolate :
        # note we have to keep the array sorted here because we are going to use it for interpolation
        ix = np.append( 2*ix[0]-ix[1] , ix)
        iy = np.append(0.,iy)
        ix = np.append(ix, 2*ix[-1]-ix[-2])
        iy = np.append(iy, 0.)

    ty=np.interp(tx,ix,iy)

    #  add input nodes which are inside the node array
    k=np.where((ix>=tx[0])&(ix<=tx[-1]))[0]
    if k.size :
        tx=np.append(tx,ix[k])
        ty=np.append(ty,iy[k])

    # sort this node array
    p = tx.argsort()
    tx=tx[p]
    ty=ty[p]

    trapeze_integrals=(ty[1:]+ty[:-1])*(tx[1:]-tx[:-1])/2.

    trapeze_centers=(tx[1:]+tx[:-1])/2.
    binsize = bins[1:]-bins[:-1]

    if np.any(binsize<=0)  :
        raise ValueError("Zero or negative bin size")

    return np.histogram(trapeze_centers, bins=bins, weights=trapeze_integrals)[0] / binsize

def blindData(data, z_grid, z_map, lambda_grid, lambda_map):
    """Ads AP blinding (strategy A) to the deltas data

    Args:
        data: dict
            A dictionary with the forests in each healpix
        z_grid, z_map: input and output values of mapping functions
            to interpolate and blind the qso redshift value: z' = z+zmap(z)
        lambda_grid, lambda_map: input and output values of mapping functions
            to interpolate and blind the forest lambda values: 
            lambda' = lambda*lambda_map(lambda)

    Returns:
        The following variables:
        data: dict
            A dictionary with the blinded forests and in each healpix
    """
    lambda_pix = []

    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            # Z QSO ap shift
            z_qso = forest.z_qso # may  be used later depending on the new grid
            z_rebin = np.interp( z_qso, z_grid, z_map )
            forest.z_qso = z_qso + z_rebin
            
            # QSO forest ap shift with interval conservation
            lambda_ = 10**( forest.log_lambda )
            lambda_map_rebin = resample_flux( lambda_, lambda_grid, lambda_map )

            lambda_rebin = lambda_map_rebin*lambda_     # this new lambda' vector is not linearly spaced in lambda
            # creating new linear loglam interval to save lambda'
            lambda_prime = 10**( np.arange(np.log10( np.min(lambda_rebin)), np.log10(np.max(lambda_rebin)) , forest.delta_log_lambda)   )
            # lambda_prime = lambda-( lambda[0]-lambda_rebin[0] )  #to use original picca interval, just shifting it to store lambda' (more data loss)
            flux, ivar = resample_flux(lambda_prime, lambda_rebin, forest.flux, ivar=forest.ivar )   #rebining to lambda' while conserving flux
            cont = resample_flux(lambda_prime, lambda_rebin, forest.cont )     # rebining continuum to lambda'
            

            # 1 to mask data: first two conditionals to cut new data to restframe lya range, 
            #  last two conditionals to cut data for prep_del stacks in observed frame, only needed 
            #  when blinding is before/during continuum fitting, will induce pixel loss
            if 0:
               # to calc pixel loss
               #lambda_rebin_rest = np.log10( lambda_rebin / ( 1 + forest.z_qso ) )
               #lambda_rebin_mask = ( lambda_rebin_rest > forest.log_lambda_min_rest_frame) & ( lambda_rebin_rest < forest.log_lambda_max_rest_frame) & (np.log10(lambda_rebin) > forest.log_lambda_min ) & (  (lambda_rebin) < ( 10**forest.log_lambda_max - 1) )
               lambda_prime_rest = np.log10(   lambda_prime / (1+forest.z_qso)   )   # using new z_qso value
               lambda_mask = (lambda_prime_rest > forest.log_lambda_min_rest_frame) & (lambda_prime_rest < forest.log_lambda_max_rest_frame) & (np.log10(lambda_prime) > forest.log_lambda_min ) & (  (lambda_prime) < ( 10**forest.log_lambda_max - 1) )
               forest.log_lambda = np.log10( lambda_prime[lambda_mask]  )
               forest.flux = flux[lambda_mask]
               forest.ivar = ivar[lambda_mask]
            else:
               forest.log_lambda = np.log10( lambda_prime )
               forest.flux = flux
               forest.ivar = ivar
               forest.cont = cont
               if forest.mean_expected_flux_frac is not None:
                  forest.mean_expected_flux_frac = resample_flux(lambda_prime, lambda_rebin, forest.mean_expected_flux_frac  )
            
            # to save pixel loss
            #pixel_diff = len(lambda_rebin) - sum(lambda_rebin_mask)
            #lambda_pix.append( np.hstack(( z_qso, forest.z_qso, pixel_diff, len(lambda_rebin) ))  )
    # to save pixel loss
    #np.save('~/losspixels_picca', lambda_pix)        
    return data
