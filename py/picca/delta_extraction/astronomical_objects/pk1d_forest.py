"""This module defines the abstract class Pk1dForest from which all
objects representing a forest in the Pk1D analysis must inherit from
"""
import numpy as np

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import AstronomicalObjectError


class Pk1dForest(Forest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    __init__
    class_variable_check
    consistency_check
    rebin

    Class Attributes
    ----------------
    delta_log_lambda: float (from Forest)
    Variation of the logarithm of the wavelength (in Angs) between two pixels.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_max: float or None (from Forest)
    Logarithm of the maximum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_max_rest_frame: float or None (from Forest)
    As log_lambda_max but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    log_lambda_min: float or None (from Forest)
    Logarithm of the minimum wavelength (in Angs) to be considered in a forest.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    log_lambda_min_rest_frame: float or None (from Forest)
    As log_lambda_min but for rest-frame wavelength. This should not be None if
    wave_solution is "log". Ignored if wave_solution is "lin".

    mask_fields: list of str (from Forest)
    Names of the fields that are affected by masking. In general it will
    be "flux", "ivar", "transmission_correction", "exposures_diff", "reso" and
    either "log_lambda" if Forest.wave_solution is "log" or "lambda_" if
    Forests.wave_solution is "lin", but some child classes might add more.

    wave_solution: "lin" or "log" (from Forest)
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    lambda_abs_igm: float
    Wavelength of the IGM absorber

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

    bad_continuum_reason: str or None (from Forest)
    Reason as to why the continuum fit is not acceptable. None for acceptable
    contiuum.

    continuum: array of float or None (from Forest)
    Quasar continuum. None for no information

    deltas: array of float or None (from Forest)
    Flux-transmission field (delta field). None for no information

    flux: array of float (from Forest)
    Flux

    ivar: array of float (from Forest)
    Inverse variance

    log_lambda: array of float or None
    Logarithm of the wavelength (in Angstroms)

    mean_snr: float (from Forest)
    Mean signal-to-noise of the forest

    transmission_correction: array of float (from Forest)
    Transmission correction.

    weights: array of float or None (from Forest)
    Weights associated to the delta field. None for no information

    exposures_diff: array of floats
    Difference between exposures

    mean_z: float
    Mean redshift of the forest

    reso: array of floats
    Resolution of the forest
    """

    lambda_abs_igm = None

    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary contiaing the information

        Raise
        -----
        AstronomicalObjectError if there are missing variables
        """
        Pk1dForest.class_variable_check()

        self.exposures_diff = kwargs.get("exposures_diff")
        if self.exposures_diff is None:
            raise AstronomicalObjectError("Error constructing Pk1dForest. "
                                          "Missing variable 'exposures_diff'")
        del kwargs["exposures_diff"]

        self.reso = kwargs.get("reso")
        if self.reso is None:
            raise AstronomicalObjectError("Error constructing Pk1dForest. "
                                          "Missing variable 'reso'")
        del kwargs["reso"]

        # call parent constructor
        super().__init__(**kwargs)

        # compute mean quality variables
        self.mean_reso = self.reso.mean()
        self.mean_z = (
            (np.power(10., self.log_lambda[len(self.log_lambda) - 1]) +
             np.power(10., self.log_lambda[0])) / 2. /
            Pk1dForest.lambda_abs_igm - 1.0)

        self.consistency_check()

    @classmethod
    def class_variable_check(cls):
        """Check that class variables have been correctly initialized"""
        if cls.lambda_abs_igm is None:
            raise AstronomicalObjectError("Error constructing Pk1DForest. "
                                          "Class variable 'lambda_abs_igm' "
                                          "must be set prior to initialize "
                                          "instances of this type")

    def consistency_check(self):
        """Consistency checks after __init__"""
        super().consistency_check()
        if self.flux.size != self.exposures_diff.size:
            raise AstronomicalObjectError(
                "Error constructing Pk1dForest. 'flux', "
                "and 'exposures_diff' don't have the "
                "same size")
        if "exposures_diff" not in Forest.mask_fields:
            Forest.mask_fields += ["exposures_diff"]
        if "reso" not in Forest.mask_fields:
            Forest.mask_fields += ["reso"]

    def coadd(self, other):
        """Coadd the information of another forest.

        Extends the coadd method of Forest to also include information
        about the exposures_diff and reso arrays

        Arguments
        ---------
        other: Pk1dForest
        The forest instance to be coadded.

        Raise
        -----
        AstronomicalObjectError if other is not a Pk1dForest instance
        """
        if not isinstance(other, Pk1dForest):
            raise AstronomicalObjectError(
                "Error coadding Pk1dForest. Expected "
                "Pk1dForest instance in other. Found: "
                f"{type(other)}")
        self.exposures_diff = np.append(self.exposures_diff,
                                        other.exposures_diff)
        self.reso = np.append(self.reso, other.reso)

        # coadd the deltas by rebinning
        super().coadd(other)

    def get_data(self):
        """Get the data to be saved in a fits file.

        Extends the get_data method of Forest to also include data for
        ivar and exposures_diff.

        Return
        ------
        cols: list of arrays
        Data of the different variables

        names: list of str
        Names of the different variables

        units: list of str
        Units of the different variables

        comments: list of str
        Comments attached to the different variables
        """
        cols, names, units, comments = super().get_data()

        cols += [self.ivar, self.exposures_diff]
        names += ["IVAR", "DIFF"]
        comments += [
            "Inverse variance. Check input spectra for units",
            "Difference. Check input spectra for units"
        ]
        units += ["Flux units", "Flux units"]

        return cols, names, units, comments

    def get_header(self):
        """Return line-of-sight data to be saved as a fits file header

        Adds to specific Pk1dForest keys to general header (defined in class
        Forsest)

        Return
        ------
        header : list of dict
        A list of dictionaries containing 'name', 'value' and 'comment' fields
        """
        header = super().get_header()
        header += [
            {
                'name': 'MEANZ',
                'value': self.mean_z,
                'comment': 'Mean redshift'
            },
            {
                'name': 'MEANRESO',
                'value': self.mean_reso,
                'comment': 'Mean resolution'
            },
        ]

        return header

    def rebin(self):
        """Rebin the arrays and update control variables

        Extends the rebon method of Forest to also rebin exposures_diff and compute
        the control variable mean_reso.

        Rebinned arrays are flux, ivar, lambda_ or log_lambda,
        transmission_correctionm, exposures_diff, and reso. Control variables
        are mean_snr and mean_reso.

        Return
        ------
        bins: array of float
        Binning solution to be used for the rebinning

        rebin_ivar: array of float
        Rebinned version of ivar

        orig_ivar: array of float
        Original version of ivar (before applying the function)

        w1: array of bool
        Masking array for the bins solution

        w2: array of bool
        Masking array for the rebinned ivar solution

        Raise
        -----
        AstronomicalObjectError if Forest.wave_solution is not 'lin' or 'log'
        """
        bins, rebin_ivar, orig_ivar, w1, w2 = super().rebin()
        if len(rebin_ivar) == 0:
            self.exposures_diff = np.array([])
            self.reso = np.array([])
            return [], [], [], [], []

        # apply mask due to cuts in bin
        self.exposures_diff = self.exposures_diff[w1]
        self.reso = self.reso[w1]

        # rebin exposures_diff and reso
        rebin_exposures_diff = np.zeros(bins.max() + 1)
        rebin_reso = np.zeros(bins.max() + 1)
        rebin_exposures_diff_aux = np.bincount(bins,
                                               weights=orig_ivar[w1] *
                                               self.exposures_diff)
        rebin_reso_aux = np.bincount(bins, weights=orig_ivar[w1] * self.reso)
        rebin_exposures_diff[:len(rebin_exposures_diff_aux
                                 )] += rebin_exposures_diff_aux
        rebin_reso[:len(rebin_reso_aux)] += rebin_reso_aux

        # apply mask due to rebinned inverse vairane
        self.exposures_diff = rebin_exposures_diff[w2] / rebin_ivar[w2]
        self.reso = rebin_reso[w2] / rebin_ivar[w2]

        # finally update control variables
        self.mean_reso = self.reso.mean()
        self.mean_z = (
            (np.power(10., self.log_lambda[len(self.log_lambda) - 1]) +
             np.power(10., self.log_lambda[0])) / 2. /
            Pk1dForest.lambda_abs_igm - 1.0)

        # return weights and binning solution to be used by child classes if
        # required
        return bins, rebin_ivar, orig_ivar, w1, w2
