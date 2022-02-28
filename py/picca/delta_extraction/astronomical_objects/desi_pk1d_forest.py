"""This module defines the class DesiPk1dForest to represent DESI forests
in the Pk1D analysis
"""
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.pk1d_forest import Pk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import AstronomicalObjectError


class DesiPk1dForest(DesiForest, Pk1dForest):
    """Forest Object

    Methods
    -------
    __gt__ (from AstronomicalObject)
    __eq__ (from AstronomicalObject)
    class_variable_check (from Forest, Pk1dForest)
    consistency_check (from Forest, Pk1dForest)
    get_data (from Forest, Pk1dForest)
    rebin (from Forest)
    coadd (from DesiForest, Pk1dForest)
    get_header (from DesiForest, Pk1dForest)
    __init__


    Class Attributes
    ----------------
    delta_lambda: float or None (from Forest)
    Variation of the wavelength (in Angs) between two pixels. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    delta_log_lambda: float or None (from Forest)
    Variation of the logarithm of the wavelength (in Angs) between two pixels.
    This should not be None if wave_solution is "log". Ignored if wave_solution
    is "lin".

    lambda_max: float or None (from Forest)
    Maximum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_max_rest_frame: float or None (from Forest)
    As wavelength_max but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min: float or None (from Forest)
    Minimum wavelength (in Angs) to be considered in a forest. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

    lambda_min_rest_frame: float or None (from Forest)
    As wavelength_min but for rest-frame wavelength. This should not
    be None if wave_solution is "lin". Ignored if wave_solution is "log".

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
    be "flux" and "ivar" but some child classes might add more.

    wave_solution: "lin" or "log" (from Forest)
    Determines whether the wavelength solution has linear spacing ("lin") or
    logarithmic spacing ("log").

    lambda_abs_igm: float (from Pk1dForest)
    Wavelength of the IGM absorber

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

    bad_continuum_reason: str or None
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

    lambda_: array of float or None (from Forest)
    Wavelength (in Angstroms)

    log_lambda: array of float or None (from Forest)
    Logarithm of the wavelength (in Angstroms)

    mean_snr: float (from Forest)
    Mean signal-to-noise of the forest

    transmission_correction: array of float (from Forest)
    Transmission correction.

    weights: array of float or None (from Forest)
    Weights associated to the delta field. None for no information

    night: list of int (from DesiForest)
    Identifier of the night where the observation was made. None for no info

    petal: list of int (from DesiForest)
    Identifier of the spectrograph used in the observation. None for no info

    targetid: int (from DesiForest)
    Targetid of the object

    tile: list of int (from DesiForest)
    Identifier of the tile used in the observation. None for no info

    exposures_diff: array of floats (from Pk1dForest)
    Difference between exposures

    mean_z: float
    Mean redshift of the forest (from Pk1dForest)

    reso: array of floats or None (from Pk1dForest)
    Resolution of the forest
    
    resolution_matrix: array of floats or None (from Pk1dForest)
    Resolution matrix of the forests
    """
    def __init__(self, **kwargs):
        """Initialize instance

        Arguments
        ---------
        **kwargs: dict
        Dictionary containing the information

        Raise
        -----
        AstronomicalObjectError if there are missing variables
        """

        self.resolution_matrix = kwargs.get("resolution_matrix")
        if self.resolution_matrix is None:
            raise AstronomicalObjectError(
                "Error constructing DesiPk1dForest. "
                "Missing variable 'resolution_matrix'")
        if "resolution_matrix" in kwargs:
            del kwargs["resolution_matrix"]

        # call parent constructors
        super().__init__(**kwargs)

        self.consistency_check()

        super().rebin()

    def consistency_check(self):
        """Consistency checks after __init__"""
        super().consistency_check()
        if self.resolution_matrix.shape[1] != self.flux.shape[0]:
            raise AstronomicalObjectError(
                "Error constructing DesiPk1dForest. 'resolution_matrix', "
                "and 'flux' don't have the "
                "same size")
        if "resolution_matrix" not in Pk1dForest.mask_fields:
            Pk1dForest.mask_fields += ["resolution_matrix"]

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
        if not isinstance(other, DesiPk1dForest):
            raise AstronomicalObjectError(
                "Error coadding DesiPk1dForest. Expected "
                "DesiPk1dForest instance in other. Found: "
                f"{type(other)}")

        if other.resolution_matrix.size > 0 and self.resolution_matrix.size > 0:
            self.resolution_matrix = np.append(self.resolution_matrix,
                                               other.resolution_matrix,
                                               axis=1)
        elif self.resolution_matrix.size == 0:
            self.resolution_matrix = other.resolution_matrix

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

        #transposing here is necessary to store in fits file
        cols += [self.resolution_matrix.T]
        names += ["RESOMAT"]
        comments += ["Transposed Masked resolution matrix"]
        units += [""]
        return cols, names, units, comments

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
            self.resolution_matrix = np.array([[]])
            return [], [], [], [], []

        # apply mask due to cuts in bin
        self.resolution_matrix = self.resolution_matrix[:, w1]

        # rebin exposures_diff and reso
        rebin_reso_matrix_aux = np.zeros(
            (self.resolution_matrix.shape[0], bins.max() + 1))
        for index, reso_matrix_col in enumerate(self.resolution_matrix):
            rebin_reso_matrix_aux[index, :] = np.bincount(
                bins, weights=orig_ivar[w1] * reso_matrix_col)

        # apply mask due to rebinned inverse vairane
        self.resolution_matrix = rebin_reso_matrix_aux[:, w2] / rebin_ivar[
            np.newaxis, w2]

        # return weights and binning solution to be used by child classes if
        # required
        return bins, rebin_ivar, orig_ivar, w1, w2
