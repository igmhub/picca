"""This module defines the abstract class SdssForest to represent
SDSS forests
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_objects.forest import Forest

class DesiForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__
    coadd

    Class Attributes
    ----------------
    delta_lambda: float
    Variation of the wavelength (in Angs) between two pixels.

    lambda_max: float
    Maximum wavelength (in Angs) to be considered in a forest.

    lambda_min: float
    Minimum wavelength (in Angs) to be considered in a forest.

    lambda_max_rest_frame: float
    As wavelength_max but for rest-frame wavelength.

    lambda_min_rest_frame: float
    As wavelength_min but for rest-frame wavelength.

    Attributes
    ----------
    dec: float (from AstronomicalObject)
    Declination (in rad)

    healpix: int (from AstronomicalObject)
    Healpix number associated with (ra, dec)

    los_id: longint (from AstronomicalObject)
    Line-of-sight id. Same as targetid

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

    lambda_: array of float
    Wavelength (in Angstroms)

    night: int or None
    Identifier of the night where the observation was made. None for no info

    petal: int or None
    Identifier of the spectrograph used in the observation. None for no info

    targetid: int
    Targetid of the object

    tile: int or None
    Identifier of the tile used in the observation. None for no info

    """
    delta_lambda = None
    lambda_max = None
    lambda_max_rest_frame = None
    lambda_min = None
    lambda_min_rest_frame = None

    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        self.lambda_ = kwargs.get("lambda")
        if self.lambda_ is None:
            raise AstronomicalObjectError("Error constructing DesiForest. "
                                          "Missing variable 'wavelength'")
        del kwargs["lambda"]

        self.night = kwargs.get("night")
        if self.night is not None:
            del kwargs["night"]

        self.petal = kwargs.get("petal")
        if self.petal is not None:
            del kwargs["petal"]

        self.targetid = kwargs.get("targetid")
        if self.targetid is None:
            raise AstronomicalObjectError("Error constructing DesiForest. "
                                          "Missing variable 'thingid'")
        del kwargs["targetid"]

        self.tile = kwargs.get("tile")
        if self.tile is not None:
            del kwargs["tile"]

        z = kwargs.get("z")
        if z is None:
            raise AstronomicalObjectError("Error constructing DesiForest. "
                                          "Missing variable 'z'")

        # call parent constructor
        kwargs["los_id"] = self.targetid
        super().__init__(**kwargs)

        # rebin arrays
        # this needs to happen after flux and ivar arrays are initialized by
        # Forest constructor
        bins = (np.floor((self.lambda_ - DesiForest.lambda_min) /
                         DesiForest.delta_lambda + 0.5).astype(int))
        self.lambda_ = DesiForest.lambda_min + bins * DesiForest.delta_lambda
        w = (self.lambda_ >= DesiForest.lambda_min)
        w = w & (self.lambda_ < DesiForest.lambda_max)
        w = w & (self.lambda_ / (1. + z) >
                 DesiForest.lambda_min_rest_frame)
        w = w & (self.lambda_ / (1. + z) <
                 DesiForest.lambda_max_rest_frame)
        w = w & (self.ivar > 0.)
        if w.sum() == 0:
            return
        bins = bins[w]
        self.lambda_ = self.lambda_[w]
        self.flux = self.flux[w]
        self.ivar = self.ivar[w]

        self.rebin(bins)


    def coadd(self, other):
        """Coadds the information of another forest.

        Forests are coadded by using inverse variance weighting

        Arguments
        ---------
        other: DesiForest
        The forest instance to be coadded.
        """
        self.lambda_ = np.append(self.lambda_, other.lambda_)
        self.flux = np.append(self.flux, other.flux)
        self.ivar = np.append(self.ivar, other.ivar)

        # coadd the deltas by rebinning
        bins = np.floor((self.lambda_ - DesiForest.lambda_min) /
                        DesiForest.delta_lambda + 0.5).astype(int)
        self.rebin(bins)

    def rebin(self, bins):
        """Rebin the lambda_, flux and ivar arrays.

        Flux and ivar are rebinned using the Forest version of rebin

        Arguments
        ---------
        bins: array of floats
        The binning solution
        """
        # flux and ivar are rebinned by super()
        w = super().rebin(bins)
        # rebin wavelength
        rebin_lambda = (DesiForest.lambda_min +
                        np.arange(bins.max() + 1) * DesiForest.delta_lambda)
        self.lambda_ = rebin_lambda[w]
