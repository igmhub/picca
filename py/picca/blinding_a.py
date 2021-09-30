"""This module defines some functions to blind data using strategy A"""
import numpy as np
from picca.constants import Cosmo, ABSORBER_IGM
import sys, os  # requiered to mute Cosmo and dont print the
                #   cosmology change on screen


# disable print to null to avout cosmo function outputs with O_m values
def blockPrint():
    """Redirect stdout to null (mute print to screen functions)
        until enablePrint() is called

    """
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    """Redirect stdout to __stdout__ (Unmute print to screen functions)
        after enablePrint() is called

    """
    sys.stdout = sys.__stdout__

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
        l_grid, l_map: input and output values of mapping functions
            to interpolate and blind the forest lambda values: l'= l*lmap(l)
    """


    blockPrint() # mute print output for Cosmo function
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
    
    enablePrint()# Unmute print output for Cosmo function
    
    zmax = 10.
    l_max = ( lambda_abs * (zmax + 1) ) - 1.3
    ll = np.log10( np.linspace(lambda_abs + .5, l_max, 10000) )
    z = 10**ll/lambda_abs-1.
    
    r_comov = cosmo_m.get_r_comov(z)
    r_comov2 = cosmo_m2.get_r_comov(z)
    
    znew = resample_flux(r_comov, r_comov2, z)
    zmask = ( z <= 10 )

    z_grid = z[zmask]
    znew = znew[zmask]
    r_comov = r_comov[zmask]
    r_comov2 = r_comov2[zmask]
    ll = ll[zmask]
    
    llnew = np.log10( lambda_abs * ( znew + 1 ) )
    z_map = -znew + z
    l_grid = 10**ll
    l_map = l_grid / (10**llnew)
    
    return z_grid, z_map, l_grid, l_map

def resample_flux(xout, x, flux, ivar=None, extrapolate=False):
    """Function to do a weighted resampling of fluxes into a new
        absissa array
        Taken from desispec.interpolation::

    Args:
        xout: abscissa array for which the flux will be resampled
        x: original abscissa
        flux: original flux
        ivar: inverse variance weighting of the original flux,
            if not given the output flux will an interpolation
        extrapolate: if true borders will be extrapolated to 
            "improve" the edge resampling
    Returns:
        outflux: new flux resampled to xout, i.e. flux(xout)
        outivar: -only if ivar given as arg- new ivar values 
            for outflux

    """
    if ivar is None:
        return _unweighted_resample(xout, x, flux, extrapolate=extrapolate)
    else:
        if extrapolate :
            raise ValueError("Cannot extrapolate ivar. Either set ivar=None and extrapolate=True or the opposite")
        a = _unweighted_resample(xout, x, flux*ivar, extrapolate=False)
        b = _unweighted_resample(xout, x, ivar, extrapolate=False)
        mask = (b>0)
        outflux = np.zeros(a.shape)
        outflux[mask] = a[mask] / b[mask]
        dx = np.gradient(x)
        dxout = np.gradient(xout)
        outivar = _unweighted_resample(xout, x, ivar/dx)*dxout

        return outflux, outivar

def _unweighted_resample(output_x,input_x,input_flux_density, extrapolate=False) :
    """Function to do a unweighted resampling of fluxes into a new
        absissa array, works as a interpolation
        Taken from desispec.interpolation

    Args:
        output_x: abscissa array for which the flux will be resampled
        input_x: original abscissa
        input_flux_density: original flux
        extrapolate: if true borders will be extrapolated to 
            "improve" the edge resampling
    Returns:
        histogram: new flux resampled to output_x, i.e. 
            input_flux_density(output_x)
    """
    ix=input_x
    iy=input_flux_density
    ox=output_x

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

def blindData(data, z_grid, z_map, l_grid, l_map):
    """Ads AP blinding (strategy A) to the deltas data

    Args:
        data: dict
            A dictionary with the forests in each healpix
        z_grid, z_map: input and output values of mapping functions
            to interpolate and blind the qso redshift value: z' = z+zmap(z)
        l_grid, l_map: input and output values of mapping functions
            to interpolate and blind the forest lambda values: l' = l*lmap(l)

    Returns:
        The following variables:
        data: dict
            A dictionary with the blinded forests and in each healpix
    """
    lpix = []

    for healpix in sorted(list(data.keys())):
        for forest in data[healpix]:
            # Z QSO ap shift
            Za = forest.z_qso
            Z_rebin = np.interp( Za, z_grid, z_map )
            forest.z_qso = Za + Z_rebin
            
            # QSO forest ap shift with interval conservation
            l = 10**( forest.log_lambda )
            l_map_rebin = resample_flux( l, l_grid, l_map )

            l_rebin = l_map_rebin*l     #this new vector is not linearly spaced in lambda
            # creating new linear loglam interval to save all data
            l2 = 10**( np.arange(np.log10( np.min(l_rebin)), np.log10(np.max(l_rebin)) , forest.delta_log_lambda)   )
            # l2 = l-( l[0]-l_rebin[0] )  #to use original picca interval, just shifting it (more data loss)
            flux, ivar = resample_flux(l2, l_rebin, forest.flux, ivar=forest.ivar )   #rebining while conserving flux
            cont = resample_flux(l2, l_rebin, forest.cont )         #rebining continuum
            
            # to calc pixel loss
            #lrebinrest = np.log10( l_rebin / ( 1 + forest.z_qso ) )
            #lrebinmask = ( lrebinrest > forest.log_lambda_min_rest_frame) & ( lrebinrest < forest.log_lambda_max_rest_frame) & (np.log10(l_rebin) > forest.log_lambda_min ) & (  (l_rebin) < ( 10**forest.log_lambda_max - 1) )

            # 1 to mask data: first two for restframe lya range, 
            #   last two for prep_del obs frame stacks, only needed when
            #   blinding is before/during continuum fitting, will induce pixel loss
            if 0:
               lnrest = np.log10(   l2 / (1+forest.z_qso)   )
               lmask = (lnrest > forest.log_lambda_min_rest_frame) & (lnrest < forest.log_lambda_max_rest_frame) & (np.log10(l2) > forest.log_lambda_min ) & (  (l2) < ( 10**forest.log_lambda_max - 1) )
               forest.log_lambda = np.log10( l2[lmask]  )
               forest.flux = flux[lmask]
               forest.ivar = ivar[lmask]
            else:
               forest.log_lambda = np.log10( l2 )
               forest.flux = flux
               forest.ivar = ivar
               forest.cont = cont
               if forest.mean_expected_flux_frac is not None:
                  forest.mean_expected_flux_frac = resample_flux(l2, l_rebin, forest.mean_expected_flux_frac  )
            
            # pixel loss
            #diff = len(l_rebin) - sum(lrebinmask)
            #lpix.append( np.hstack(( Za, forest.z_qso, diff, len(l_rebin) ))  )
    # pixel loss
    #np.save('~/losspixels_picca', lpix)        
    return data
