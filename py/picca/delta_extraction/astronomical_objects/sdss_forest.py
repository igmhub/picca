"""This module defines the abstract class SdssForest to represent
SDSS forests
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_objects.forest import Forest

class SdssForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__
    coadd

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
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'fiberid'")
        del kwargs["fiberid"]

        self.mjd = kwargs.get("mjd")
        if self.mjd is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'mjd'")
        del kwargs["mjd"]

        self.log_lambda = kwargs.get("log_lambda")
        if self.log_lambda is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'log_lambda'")
        del kwargs["log_lambda"]

        self.plate = kwargs.get("plate")
        if self.plate is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'plate'")
        del kwargs["plate"]

        self.thingid = kwargs.get("thingid")
        if self.thingid is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'thingid'")
        del kwargs["thingid"]

        z = kwargs.get("z")
        if z is None:
            raise AstronomicalObjectError("Error constructing SdssForest. "
                                          "Missing variable 'z'")

        # call parent constructor
        kwargs["los_id"] = self.thingid
        super().__init__(**kwargs)
        self.mask_fields.append("log_lambda")

        # consistency check
        if self.log_lambda.size != self.flux.size:
            raise AstronomicalObjectError("Error constructing SdssForest. 'flux' "
                                          " and 'log_lambda' don't have the same "
                                          " size")

        # rebin arrays
        # this needs to happen after flux and ivar arrays are initialized by
        # Forest constructor
        bins = (np.floor((self.log_lambda - SdssForest.log_lambda_min) /
                         SdssForest.delta_log_lambda + 0.5).astype(int))
        self.log_lambda = SdssForest.log_lambda_min + bins * SdssForest.delta_log_lambda
        w = (self.log_lambda >= SdssForest.log_lambda_min)
        w = w & (self.log_lambda < SdssForest.log_lambda_max)
        w = w & (self.log_lambda - np.log10(1. + z) >
                 SdssForest.log_lambda_min_rest_frame)
        w = w & (self.log_lambda - np.log10(1. + z) <
                 SdssForest.log_lambda_max_rest_frame)
        w = w & (self.ivar > 0.)
        if w.sum() == 0:
            return
        bins = bins[w]
        self.log_lambda = self.log_lambda[w]
        self.flux = self.flux[w]
        self.ivar = self.ivar[w]
        self.transmission_correction = self.transmission_correction[w]

        self.rebin(bins)

    def coadd(self, other):
        """Coadds the information of another forest.

        Forests are coadded by using inverse variance weighting.

        Arguments
        ---------
        other: SdssForest
        The forest instance to be coadded.
        """
        self.log_lambda = np.append(self.log_lambda, other.log_lambda)
        self.flux = np.append(self.flux, other.flux)
        self.ivar = np.append(self.ivar, other.ivar)
        self.transmission_correction = np.append(self.transmission_correction,
                                                 other.transmission_correction)

        # coadd the deltas by rebinning
        bins = np.floor((self.log_lambda - SdssForest.log_lambda_min) /
                        SdssForest.delta_log_lambda + 0.5).astype(int)
        self.rebin(bins)

    def rebin(self, bins):
        """Rebin log_lambda, flux and ivar arrays

        Flux and ivar are rebinned using the Forest version of rebin

        Arguments
        ---------
        bins: array of floats
        The binning solution
        """
        # flux and ivar are rebinned by super()
        w = super().rebin(bins)
        # rebin log lambda
        rebin_lambda = (SdssForest.log_lambda_min +
                        np.arange(bins.max() + 1) * SdssForest.delta_log_lambda)
        self.log_lambda = rebin_lambda[w]
