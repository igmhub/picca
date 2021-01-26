"""This module defines the abstract class Forest from which all
objects representing a forest must inherit from
"""
import numpy as np

from picca.delta_extraction.errors import AstronomicalObjectError

from picca.delta_extraction.astronomical_object import AstronomicalObject

class Forest(AstronomicalObject):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__
    rebin

    Attributes
    ----------
    bad_continuum_reason: str or None
    Reason as to why the continuum fit is not acceptable. None for acceptable
    contiuum.

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

    continuum: array of float or None
    Quasar continuum. None for no information

    deltas: array of float or None
    Flux-transmission field (delta field). None for no information

    flux: array of float
    Flux

    ivar: array of float
    Inverse variance

    mask_fields: list of str
    Names of the fields that are affected by masking. In general it will
    be "flux", "ivar" and "transmission_correction" but some child classes might
    add more.

    mean_snf: float
    Mean signal-to-noise of the forest

    transmission_correction: array of float
    Transmission correction.
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information
        """
        self.bad_continuum_reason = None
        self.continuum = kwargs.get("continuum")
        if kwargs.get("continuum") is not None:
            del kwargs["continuum"]

        self.deltas = kwargs.get("deltas")
        if kwargs.get("deltas") is not None:
            del kwargs["deltas"]

        self.flux = kwargs.get("flux")
        if self.flux is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Missing variable 'flux'")
        del kwargs["flux"]

        self.ivar = kwargs.get("ivar")
        if self.ivar is None:
            raise AstronomicalObjectError("Error constructing Forest. "
                                          "Missing variable 'ivar'")
        del kwargs["ivar"]

        self.mask_fields = kwargs.get("mask_fields")
        if self.mask_fields is None:
            self.mask_fields = ["flux", "ivar", "transmission_correction"]
        else:
            del kwargs["mask_fields"]

        self.transmission_correction = np.ones_like(self.flux)

        # compute mean quality variables
        error = 1.0 / np.sqrt(self.ivar)
        snr = self.flux / error
        self.mean_snr = sum(snr) / float(len(snr))

        # call parent constructor
        super().__init__(**kwargs)

    def rebin(self, bins):
        """Rebin the flux and ivar arrays.

        Arguments
        ---------
        bins: array of float
        The binning solution

        Returns
        -------
        w: array of bool
        Mask used in the rebinning
        """
        rebin_flux = np.zeros(bins.max() + 1)
        rebin_ivar = np.zeros(bins.max() + 1)
        rebin_flux_aux = np.bincount(bins, weights=self.ivar * self.flux)
        rebin_ivar_aux = np.bincount(bins, weights=self.ivar)
        rebin_flux[:len(rebin_flux_aux)] += rebin_flux_aux
        rebin_ivar[:len(rebin_ivar_aux)] += rebin_ivar_aux

        w = (rebin_ivar > 0.)
        if w.sum() == 0:
            raise AstronomicalObjectError("Attempting to rebin arrays flux and "
                                          "ivar in class Forest, but ivar seems "
                                          "to contain only zeros")
        self.flux = rebin_flux[w] / rebin_ivar[w]
        self.ivar = rebin_ivar[w]

        return w
