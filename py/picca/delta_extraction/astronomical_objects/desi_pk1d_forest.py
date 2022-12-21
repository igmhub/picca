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

    Class Methods
    -------------
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    update_class_variables

    Methods
    -------
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)
    __init__
    consistency_check
    coadd
    get_data
    rebin

    Class Attributes
    ----------------
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    Attributes
    ----------
    (see DesiForest in py/picca/delta_extraction/astronomical_objects/desi_forest.py)
    (see Pk1dForest in py/picca/delta_extraction/astronomical_objects/pk1d_forest.py)

    resolution_matrix: 2d-array of floats or None
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
        #potentially change this in case we ever want log-binning with DESI Pk1d data
        #then would need a check of self.wave_solution
        if self.resolution_matrix is None:
            raise AstronomicalObjectError(
                "Error constructing DesiPk1dForest. "
                "Missing variable 'resolution_matrix'")
        del kwargs["resolution_matrix"]

        # call parent constructors
        super().__init__(**kwargs)
        self.consistency_check()

    def consistency_check(self):
        """Consistency checks after __init__"""
        super().consistency_check()
        if self.resolution_matrix.shape[1] != self.flux.shape[0]:
            raise AstronomicalObjectError(
                "Error constructing DesiPk1dForest. 'resolution_matrix' "
                "and 'flux' don't have the "
                "same size")
        if "resolution_matrix" not in Forest.mask_fields:
            Forest.mask_fields += ["resolution_matrix"]

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
        AstronomicalObjectError if other is not a DesiPk1dForest instance
        """
        if not isinstance(other, DesiPk1dForest):
            raise AstronomicalObjectError(
                "Error coadding DesiPk1dForest. Expected "
                "DesiPk1dForest instance in other. Found: "
                f"{type(other).__name__}")

        if other.resolution_matrix.size > 0 and self.resolution_matrix.size > 0:
            if self.resolution_matrix.shape[0] != other.resolution_matrix.shape[
                    0]:
                largershape = np.max([
                    self.resolution_matrix.shape[0],
                    other.resolution_matrix.shape[0]
                ])
                smallershape = np.min([
                    self.resolution_matrix.shape[0],
                    other.resolution_matrix.shape[0]
                ])
                shapediff = largershape - smallershape
                if self.resolution_matrix.shape[0] == smallershape:
                    self.resolution_matrix = np.append(np.zeros(
                        [shapediff // 2, self.resolution_matrix.shape[1]]),
                                                       self.resolution_matrix,
                                                       axis=0)
                    self.resolution_matrix = np.append(
                        self.resolution_matrix,
                        np.zeros(
                            [shapediff // 2, self.resolution_matrix.shape[1]]),
                        axis=0)
                if other.resolution_matrix.shape[0] == smallershape:
                    other.resolution_matrix = np.append(np.zeros(
                        [shapediff // 2, other.resolution_matrix.shape[1]]),
                                                        other.resolution_matrix,
                                                        axis=0)
                    other.resolution_matrix = np.append(
                        other.resolution_matrix,
                        np.zeros(
                            [shapediff // 2, other.resolution_matrix.shape[1]]),
                        axis=0)

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
        rebin_ivar, orig_ivar, w1, w2, wslice_inner = super().rebin()
        if len(rebin_ivar) == 0 or np.sum(w2) == 0:
            self.resolution_matrix = np.array([[]])
            return [], [], [], np.array([]), np.array([])

        # apply mask due to cuts in bin
        self.resolution_matrix = self.resolution_matrix[:, w1]

        # Find non-empty bins
        binned_arr_size = self.log_lambda_index.max() + 1

        # Do a simple average when ivar=0
        orig_ivar_2 = orig_ivar[w1]
        w__ = orig_ivar_2>0
        orig_ivar_2[~w__] = 1
        rebin_reso_ivar = np.bincount(self.log_lambda_index,
                                      weights=orig_ivar_2, minlength=binned_arr_size)

        # rebin resolution_matrix
        rebin_reso_matrix_aux = np.zeros(
            (self.resolution_matrix.shape[0], binned_arr_size))
        for index, reso_matrix_col in enumerate(self.resolution_matrix):
            rebin_reso_matrix_aux[index, :] = np.bincount(
                self.log_lambda_index, weights=orig_ivar_2 * reso_matrix_col)

        # apply mask due to rebinned inverse vairane
        self.resolution_matrix = rebin_reso_matrix_aux[:, wslice_inner] / rebin_reso_ivar[
            np.newaxis, wslice_inner]

        # return weights and binning solution to be used by child classes if
        # required
        return rebin_ivar, orig_ivar, w1, w2, wslice_inner

    @classmethod
    def update_class_variables(cls):
        """Update class variable mask_fields (from Forest) to also contain the
        necessary fields for this class to work properly.
        """
        cls.class_variable_check()
        for field in ["exposures_diff", "reso", "reso_pix", "resolution_matrix"]:
            if field not in Forest.mask_fields:
                cls.mask_fields.append(field)
