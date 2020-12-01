"""This module defines the abstract class SdssForest to represent
SDSS forests
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_objects.drq_object import DrqObject
from picca.delta_extraction.astronomical_objects.forest import Forest

class SdssForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__

    Class Attributes
    ----------------
    delta_log_lambda: float
    Variation of the logarithm of the wavelength (in Angs) between two pixels.

    log_lambda_max: float
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.

    log_lambda_min: float
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.

    log_lambda_max_rest_frame: float
    As log_lambda_max but for rest-frame wavelength.

    log_lambda_min_rest_frame: float
    As log_lambda_min but for rest-frame wavelength.

    Attributes
    ----------
    dec: float (from AstronomicalObject)
    Declination (in rad)

    healpix: int (from AstronomicalObject)
    Healpix number associated with (ra, dec)

    los_id: longint (from AstronomicalObject)
    Line-of-sight id. Same as thingid

    ra: float (from AstronomicalObject)
    Right ascention (in rad)

    z: float (from AstronomicalObject)
    Redshift

    continuum: array of float or None (from Forest)
    Quasar continuum. None for no information

    deltas: array of float or None (from Forest)
    Flux-transmission field (delta field). None for no information

    flux: array of float (from Forest)
    Flux

    ivar: array of float (from Forest)
    Inverse variance

    mask_fields: list of str (from Forest)
    Names of the fields that are affected by masking. In general it will
    be "flux" and "ivar" but some child classes might add more.

    mean_snf: float (from Forest)
    Mean signal-to-noise of the forest

    fiberid: int
    Fiberid of the observation

    mjd: int
    Modified Julian Date of the observation

    plate: int
    Plate of the observation

    thingid: int
    Thingid of the object

    log_lambda: array of float
    Logarithm of the wavelengths (in Angstroms)
    """
    delta_log_lambda = None
    log_lambda_max = None
    log_lambda_max_rest_frame = None
    log_lambda_min = None
    log_lambda_min_rest_frame = None

    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        if SdssForest.delta_log_lambda is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Class variable 'delta_log_lambda' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if SdssForest.log_lambda_max is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Class variable 'log_lambda_max' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if SdssForest.log_lambda_max_rest_frame is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Class variable 'log_lambda_max_rest_frame' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if SdssForest.log_lambda_min is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Class variable 'log_lambda_min' "
                                          "must be set prior to initialize "
                                          "instances of this type")
        if SdssForest.log_lambda_min_rest_frame is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Class variable 'log_lambda_min_rest_frame' "
                                          "must be set prior to initialize "
                                          "instances of this type")

        self.fiberid = kwargs.get("fiberid")
        if self.fiberid is None:
            raise AstronomicalObjectError("Error constructing DrqObject. "
                                          "Missing variable 'fiberid'")
        del kwargs["fiberid"]

        self.flux = kwargs.get("flux")
        if self.flux is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'flux'")
        del kwargs["flux"]

        self.ivar = kwargs.get("ivar")
        if self.ivar is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'ivar'")
        del kwargs["ivar"]

        self.mjd = kwargs.get("mjd")
        if self.mjd is None:
            raise AstronomicalObjectError("Error constructing DrqObject. "
                                          "Missing variable 'mjd'")
        del kwargs["mjd"]

        self.log_lambda = kwargs.get("log_lambda")
        if self.log_lambda is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'log_lambda'")
        del kwargs["log_lambda"]

        self.plate = kwargs.get("plate")
        if self.plate is None:
            raise AstronomicalObjectError("Error constructing DrqObject. "
                                          "Missing variable 'plate'")
        del kwargs["plate"]

        self.thingid = kwargs.get("thingid")
        if self.thingid is None:
            raise AstronomicalObjectError("Error constructing DrqObject. "
                                          "Missing variable 'thingid'")
        del kwargs["thingid"]

        z = kwargs.get("z")
        if z is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'z'")


        ## cut to specified range
        bins = (np.floor((log_lambda - SdssForest.log_lambda_min) /
                         SdssForest.delta_log_lambda + 0.5).astype(int))
        log_lambda = SdssForest.log_lambda_min + bins * SdssForest.delta_log_lambda
        w = (log_lambda >= SdssForest.log_lambda_min)
        w = w & (log_lambda < SdssForest.log_lambda_max)
        w = w & (log_lambda - np.log10(1. + z) >
                 SdssForest.log_lambda_min_rest_frame)
        w = w & (log_lambda - np.log10(1. + z) <
                 SdssForest.log_lambda_max_rest_frame)
        w = w & (ivar > 0.)
        if w.sum() == 0:
            return
        bins = bins[w]
        log_lambda = log_lambda[w]
        flux = flux[w]
        ivar = ivar[w]

        # rebin arrays
        rebin_log_lambda = (SdssForest.log_lambda_min +
                            np.arange(bins.max() + 1) * SdssForest.delta_log_lambda)
        rebin_flux = np.zeros(bins.max() + 1)
        rebin_ivar = np.zeros(bins.max() + 1)
        rebin_flux_aux = np.bincount(bins, weights=ivar * flux)
        rebin_ivar_aux = np.bincount(bins, weights=ivar)
        rebin_flux[:len(rebin_flux_aux)] += rebin_flux_aux
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux
        w = (rebin_ivar > 0.)
        if w.sum() == 0:
            return
        log_lambda = rebin_log_lambda[w]
        flux = rebin_flux[w] / rebin_ivar[w]
        ivar = rebin_ivar[w]

        # keep the rebinned arrays
        self.log_lambda = log_lambda
        kwargs["flux"] = flux
        kwargs["ivar"] = ivar

        # call parent constructor
        kwargs["los_id"] = self.thingid
        super().__init__(**kwargs)

    def coadd(self, other):
        """Coadds the information of another forest.

        Forests are coadded by using inverse variance weighting.

        Arguments
        ---------
        other: SdssForest
        The forest instance to be coadded.
        """
        log_lambda = np.append(self.log_lambda, other.log_lambda)
        flux = np.append(self.flux, other.flux)
        ivar = np.append(self.ivar, other.ivar)

        # coadd the deltas by rebinning
        # log lambda & ivar
        bins = np.floor((log_lambda - SdssForest.log_lambda_min) /
                        SdssForest.delta_log_lambda + 0.5).astype(int)
        rebin_log_lambda = SdssForest.log_lambda_min + (np.arange(bins.max() + 1) *
                                                        SdssForest.delta_log_lambda)
        rebin_ivar = np.zeros(bins.max() + 1)
        rebin_ivar_aux = np.bincount(bins, weights=ivar)
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux
        w = (rebin_ivar > 0.)
        self.log_lambda = rebin_log_lambda[w]

        self.ivar = rebin_ivar[w]
        # flux
        rebin_flux = np.zeros(bins.max() + 1)
        rebin_flux_aux = np.bincount(bins, weights=ivar * flux)
        rebin_flux[:len(rebin_flux_aux)] += rebin_flux_aux
        self.flux = rebin_flux[w] / rebin_ivar[w]
