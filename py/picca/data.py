"""This module defines data structure to deal with line of sight data.

This module provides with three classes (QSO, Forest, Delta)
to manage the line-of-sight data.
See the respective docstrings for more details
"""
import numpy as np
from itertools import repeat
import warnings

from . import constants
from .utils import userprint

class QSO(object):
    """Class to represent quasar objects.

    Attributes:
        ra: float
            Right-ascension of the quasar (in radians).
        dec: float
            Declination of the quasar (in radians).
        z_qso: float
            Redshift of the quasar.
        plate: integer
            Plate number of the observation.
        fiberid: integer
            Fiberid of the observation.
        mjd: integer
            Modified Julian Date of the observation.
        thingid: integer
            Thingid of the observation.
        x_cart: float
            The x coordinate when representing ra, dec in a cartesian
            coordinate system.
        y_cart: float
            The y coordinate when representing ra, dec in a cartesian
            coordinate system.
        z_cart: float
            The z coordinate when representing ra, dec in a cartesian
            coordinate system.
        cos_dec: float
            Cosine of the declination angle.
        weights: float
            Weight assigned to object
        r_comov: float or None
            Comoving distance to the object
        dist_m: float or None
            Angular diameter distance to object
        log_lambda: float or None
            Wavelength associated with the quasar redshift

    Note that plate-fiberid-mjd is a unique identifier
    for the quasar.

    Methods:
        __init__: Initialize class instance.
        get_angle_between: Computes the angular separation between two quasars.
    """

    def __init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid):
        """Initializes class instance.

        Args:
            thingid: integer
                Thingid of the observation.
            ra: float
                Right-ascension of the quasar (in radians).
            dec: float
                Declination of the quasar (in radians).
            z_qso: float
                Redshift of the quasar.
            plate: integer
                Plate number of the observation.
            mjd: integer
                Modified Julian Date of the observation.
            fiberid: integer
                Fiberid of the observation.
        """
        self.ra = ra
        self.dec = dec

        self.plate = plate
        self.mjd = mjd
        self.fiberid = fiberid

        ## cartesian coordinates
        self.x_cart = np.cos(ra) * np.cos(dec)
        self.y_cart = np.sin(ra) * np.cos(dec)
        self.z_cart = np.sin(dec)
        self.cos_dec = np.cos(dec)

        self.z_qso = z_qso
        self.los_id = los_id
        #this is for legacy purposes only
        self.thingid = los_id
        warnings.warn("currently a thingid entry is created in QSO.__init__, this feature will be removed", DeprecationWarning)

        # variables computed in function io.read_objects
        self.weight = None
        self.r_comov = None
        self.dist_m = None

        # variables computed in modules bin.picca_xcf_angl and bin.picca_xcf1d
        self.log_lambda = None

    def get_angle_between(self, data):
        """Computes the angular separation between two quasars.

        Args:
            data: QSO or list of QSO
                Objects with which the angular separation will
                be computed.

        Returns
            A float or an array (depending on input data) with the angular
            separation between this quasar and the object(s) in data.
        """
        # case 1: data is list-like
        try:
            x_cart = np.array([d.x_cart for d in data])
            y_cart = np.array([d.y_cart for d in data])
            z_cart = np.array([d.z_cart for d in data])
            ra = np.array([d.ra for d in data])
            dec = np.array([d.dec for d in data])

            cos = x_cart * self.x_cart + y_cart * self.y_cart + z_cart * self.z_cart
            w = cos >= 1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos>=1.'.format(w.sum()))
                cos[w] = 1.
            w = cos <= -1.
            if w.sum() != 0:
                userprint('WARNING: {} pairs have cos<=-1.'.format(w.sum()))
                cos[w] = -1.
            angl = np.arccos(cos)

            w = ((np.absolute(ra - self.ra) < constants.SMALL_ANGLE_CUT_OFF) &
                 (np.absolute(dec - self.dec) < constants.SMALL_ANGLE_CUT_OFF))
            if w.sum() != 0:
                angl[w] = np.sqrt((dec[w] - self.dec)**2 +
                                  (self.cos_dec * (ra[w] - self.ra))**2)
        # case 2: data is a QSO
        except TypeError:
            x_cart = data.x_cart
            y_cart = data.y_cart
            z_cart = data.z_cart
            ra = data.ra
            dec = data.dec

            cos = x_cart * self.x_cart + y_cart * self.y_cart + z_cart * self.z_cart
            if cos >= 1.:
                userprint('WARNING: 1 pair has cosinus>=1.')
                cos = 1.
            elif cos <= -1.:
                userprint('WARNING: 1 pair has cosinus<=-1.')
                cos = -1.
            angl = np.arccos(cos)
            if ((np.absolute(ra - self.ra) < constants.SMALL_ANGLE_CUT_OFF) &
                    (np.absolute(dec - self.dec) < constants.SMALL_ANGLE_CUT_OFF)):
                angl = np.sqrt((dec - self.dec)**2 + (self.cos_dec *
                                                      (ra - self.ra))**2)
        return angl


class Forest(QSO):
    """Class to represent a Lyman alpha (or other absorption) forest

    This is not a proper forest class anymore, it's just the minimal set of variables to not make the corr_function, stacking etc fail
    """
    log_lambda_min = None
    log_lambda_max = None
    delta_log_lambda = None

    @classmethod
    def get_var_lss(cls, log_lambda):
        """Interpolates the pixel variance due to the Large Scale Strucure on
        the wavelength array.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_eta(cls, log_lambda):
        """Interpolates the correction factor to the contribution of the
        pipeline estimate of the instrumental noise to the variance on the
        wavelength array.

        See equation 4 of du Mas des Bourboux et al. 2020 for details.

        Empty function to be loaded at run-time.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

    @classmethod
    def get_fudge(cls, log_lambda):
        """Interpolates the fudge contribution to the variance on the
        wavelength array.

        See function epsilon in equation 4 of du Mas des Bourboux et al.
        2020 for details.

        Args:
            log_lambda: array of float
                Array containing the logarithm of the wavelengths (in Angs)

        Returns:
            An array with the correction

        Raises:
            NotImplementedError: Function was not specified
        """
        raise NotImplementedError("Function should be specified at run-time")

   


class Delta(QSO):
    """Class to represent the mean transimission fluctuation field (delta)

    This class stores the information for the deltas for a given line of sight

    Attributes:
        ## Inherits from QSO ##
        log_lambda : array of floats
            Array containing the logarithm of the wavelengths (in Angs)
        weights : array of floats
            Weights associated to pixel. Overloaded from parent class
        cont: array of floats
            Quasar continuum
        delta: array of floats
            Mean transmission fluctuation (delta field)
        order: 0 or 1
            Order of the log10(lambda) polynomial for the continuum fit
        ivar: array of floats
            Inverse variance associated to each flux
        exposures_diff: array of floats
            Difference between exposures
        mean_snr: float
            Mean signal-to-noise ratio in the forest
        mean_reso: float
            Mean resolution of the forest in units of velocity (FWHM)
        mean_z: float
            Mean redshift of the forest
        mean_reso_pix: float
            Mean resolution of the forest in units of pixels (FWHM)
        mean_resolution_matrix: array of floats or None
            Mean (over wavelength) resolution matrix for that forest
        resolution_matrix: 2d array of floats or None
            Wavelength dependent resolution matrix for that forest
        delta_log_lambda: float
            Variation of the logarithm of the wavelength between two pixels
        z: array of floats or None
            Redshift of the abosrption
        r_comov: array of floats or None
            Comoving distance to the object. Overloaded from parent class
        dist_m: array of floats or None
            Angular diameter distance to object. Overloaded from parent
            class
        neighbours: list of Delta or QSO or None
            Neighbouring deltas/quasars
        fname: string or None
            String identifying Delta as part of a group

    Methods:
        __init__: Initializes class instances.
        from_fitsio: Initialize instance from a fits file.
        from_ascii: Initialize instance from an ascii file.
        from_image: Initialize instance from an ascii file.
        project: Project the delta field.

    """

    def __init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                 weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                 mean_reso, mean_z, resolution_matrix=None,
                 mean_resolution_matrix=None, mean_reso_pix=None):
        """Initializes class instances.

        Args:
            los_id: integer
                Thingid or Targetid of the observation.
            ra: float
                Right-ascension of the quasar (in radians).
            dec: float
                Declination of the quasar (in radians).
            z_qso: float
                Redshift of the quasar.
            plate: integer
                Plate number of the observation.
            mjd: integer
                Modified Julian Date of the observation.
            fiberid: integer
                Fiberid of the observation.
            log_lambda: array of floats
                Logarithm of the wavelengths (in Angs)
            weights: array of floats
                Pixel weights
            cont: array of floats
                Quasar continuum
            delta: array of floats
                Mean transmission fluctuation (delta field)
            order: 0 or 1
                Order of the log10(lambda) polynomial for the continuum fit
            ivar: array of floats
                Inverse variance associated to each flux
            exposures_diff: array of floats
                Difference between exposures
            mean_snr: float
                Mean signal-to-noise ratio in the forest
            mean_reso: float
                Mean resolution of the forest
            mean_z: float
                Mean redshift of the forest
            mean_reso_pix: float
                Mean resolution of the forest in units of pixels (FWHM)
            mean_resolution_matrix: array of floats or None
                Mean (over wavelength) resolution matrix for that forest
            resolution_matrix: 2d array of floats or None
                Wavelength dependent resolution matrix for that forest
            delta_log_lambda: float
                Variation of the logarithm of the wavelength between two pixels
        """
        QSO.__init__(self, los_id, ra, dec, z_qso, plate, mjd, fiberid)
        self.log_lambda = log_lambda
        self.weights = weights
        self.cont = cont
        self.delta = delta
        self.order = order
        self.ivar = ivar
        self.exposures_diff = exposures_diff
        self.mean_snr = mean_snr
        self.mean_reso = mean_reso
        self.mean_z = mean_z
        self.resolution_matrix = resolution_matrix
        self.mean_resolution_matrix = mean_resolution_matrix
        self.mean_reso_pix = mean_reso_pix

        # variables computed in function io.read_deltas
        self.z = None
        self.r_comov = None
        self.dist_m = None

        # variables computed in function cf.fill_neighs or xcf.fill_neighs
        self.neighbours = None

        # variables used in function cf.compute_wick_terms and
        # main from bin.picca_wick
        self.fname = None

    @classmethod
    def from_fitsio(cls, hdu, pk1d_type=False):
        """Initialize instance from a fits file.

        Args:
            hdu: fitsio.hdu.table.TableHDU
                A Header Data Unit opened with fitsio
            pk1d_type: bool - default: False
                Specifies if the fits file is formatted for the 1D Power
                Spectrum analysis
        Returns:
            a Delta instance
        """
        header = hdu.read_header()

        # new runs of picca_deltas should have a blinding keyword
        if "BLINDING" in header:
            blinding = header["BLINDING"]
        # older runs are not from DESI main survey and should not be blinded
        else:
            blinding = "none"

        if blinding != "none":
            delta_name = "DELTA_BLIND"
        else:
            delta_name = "DELTA"

        delta = hdu[delta_name][:].astype(float)

        if 'LOGLAM' in hdu.get_colnames():
            log_lambda = hdu['LOGLAM'][:].astype(float)
        elif 'LAMBDA' in hdu.get_colnames():
            log_lambda = np.log10(hdu['LAMBDA'][:].astype(float))
        else:
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

        if pk1d_type:
            ivar = hdu['IVAR'][:].astype(float)
            try:
                exposures_diff = hdu['DIFF'][:].astype(float)
            except (KeyError, ValueError):
                userprint('WARNING: no DIFF in hdu while pk1d_type=True, filling with zeros.')
                exposures_diff = np.zeros(delta.shape)
            mean_snr = header['MEANSNR']
            mean_reso = header['MEANRESO']
            try:
                mean_reso_pix = header['MEANRESO_PIX']
            except (KeyError, ValueError):
                mean_reso_pix = None

            mean_z = header['MEANZ']
            try:
                #transposing here gives back the actual reso matrix which has been stored transposed
                resolution_matrix = hdu['RESOMAT'][:].T.astype(float)
                if resolution_matrix is not None:
                    mean_resolution_matrix = np.mean(resolution_matrix, axis=1)
                else:
                    mean_resolution_matrix = None
            except (KeyError, ValueError):
                resolution_matrix = None
                mean_resolution_matrix = None
            weights = None
            cont = None
        else:
            ivar = None
            exposures_diff = None
            mean_snr = None
            mean_reso = None
            mean_z = None
            resolution_matrix = None
            mean_resolution_matrix = None
            mean_reso_pix = None
            weights = hdu['WEIGHT'][:].astype(float)
            cont = hdu['CONT'][:].astype(float)

        if 'THING_ID' in header:
            los_id = header['THING_ID']
            plate = header['PLATE']
            mjd = header['MJD']
            fiberid = header['FIBERID']
        elif 'LOS_ID' in header:
            los_id = header['LOS_ID']
            plate=los_id
            mjd=los_id
            fiberid=los_id
        else:
            raise Exception("Could not find THING_ID or LOS_ID")

        ra = header['RA']
        dec = header['DEC']
        z_qso = header['Z']
        try:
            order = header['ORDER']
        except KeyError:
            order = 1

        return cls(los_id, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, resolution_matrix,
                   mean_resolution_matrix, mean_reso_pix)

    @classmethod
    def from_ascii(cls, line):
        """Initialize instance from an ascii file.

        Args:
            line: string
                A line of the ascii file containing information from a line
                of sight

        Returns:
            a Delta instance
        """

        cols = line.split()
        plate = int(cols[0])
        mjd = int(cols[1])
        fiberid = int(cols[2])
        ra = float(cols[3])
        dec = float(cols[4])
        z_qso = float(cols[5])
        mean_z = float(cols[6])
        mean_snr = float(cols[7])
        mean_reso = float(cols[8])
        delta_log_lambda = float(cols[9])

        num_pixels = int(cols[10])
        delta = np.array(cols[11:11 + num_pixels]).astype(float)
        log_lambda = np.array(cols[11 + num_pixels:11 +
                                   2 * num_pixels]).astype(float)
        ivar = np.array(cols[11 + 2 * num_pixels:11 +
                             3 * num_pixels]).astype(float)
        exposures_diff = np.array(cols[11 + 3 * num_pixels:11 +
                                       4 * num_pixels]).astype(float)

        thingid = 0
        order = 0
        weights = None
        cont = None

        return cls(thingid, ra, dec, z_qso, plate, mjd, fiberid, log_lambda,
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, delta_log_lambda)

    @classmethod
    def from_image(cls, hdul, pk1d_type=False, z_min_qso=0, z_max_qso=10):
        """Initialize instance from an ascii file.

        Args:
            hdu: fitsio.hdu.table.TableHDU
                A Header Data Unit opened with fitsio
            pk1d_type: bool - default: False
                Specifies if the fits file is formatted for the 1D Power
                Spectrum analysis
            z_min_qso: float - default: 0
                Specifies the minimum redshift for QSOs
            z_max_qso: float - default: 10
                Specifies the maximum redshift for QSOs
        Returns:
            a Delta instance
        """
        if pk1d_type:
            raise ValueError("ImageHDU format not implemented for Pk1D forests.")

        header = hdul["METADATA"].read_header()
        N_forests = hdul["METADATA"].get_nrows()
        Nones = np.full(N_forests, None)

        # new runs of picca_deltas should have a blinding keyword
        if "BLINDING" in header:
            blinding = header["BLINDING"]
        else:
            blinding = "none"

        if blinding != "none":
            delta_name = "DELTA_BLIND"
        else:
            delta_name = "DELTA"

        delta = hdul[delta_name].read().astype(float)

        if "LOGLAM" in hdul:
            log_lambda = hdul["LOGLAM"][:].astype(float)
        elif "LAMBDA" in hdul:
            log_lambda = np.log10(hdul["LAMBDA"][:].astype(float))
        else:
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

        ivar = Nones
        exposures_diff = Nones
        mean_snr = Nones
        mean_reso = Nones
        mean_z = Nones
        resolution_matrix = Nones
        mean_resolution_matrix = Nones
        mean_reso_pix = Nones
        weights = hdul["WEIGHT"].read().astype(float)
        w = weights > 0
        cont = hdul["CONT"].read().astype(float)

        if "THING_ID" in hdul["METADATA"].get_colnames():
            los_id = hdul["METADATA"]["THING_ID"][:]
            plate = hdul["METADATA"]["PLATE"][:]
            mjd = hdul["METADATA"]["MJD"][:]
            fiberid=hdul["METADATA"]["FIBERID"][:]
        elif "LOS_ID" in hdul["METADATA"].get_colnames():
            los_id = hdul["METADATA"]["LOS_ID"][:]
            plate=los_id
            mjd=los_id
            fiberid=los_id
        else:
            raise Exception("Could not find THING_ID or LOS_ID")

        ra = hdul["METADATA"]["RA"][:]
        dec = hdul["METADATA"]["DEC"][:]
        z_qso = hdul["METADATA"]["Z"][:]
        try:
            order = hdul["METADATA"]["ORDER"][:]
        except (KeyError, ValueError):
            order = np.full(N_forests, 1)

        deltas = []
        for (los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda,
            weights_i, cont_i, delta_i, order_i, ivar_i, exposures_diff_i, mean_snr_i,
            mean_reso_i, mean_z_i, resolution_matrix_i,
            mean_resolution_matrix_i, mean_reso_pix_i, w_i
        ) in zip(los_id, ra, dec, z_qso, plate, mjd, fiberid, repeat(log_lambda),
                   weights, cont, delta, order, ivar, exposures_diff, mean_snr,
                   mean_reso, mean_z, resolution_matrix,
                   mean_resolution_matrix, mean_reso_pix, w):
            if z_qso_i >= z_min_qso and z_qso_i <= z_max_qso:        
                deltas.append(cls(
                    los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda[w_i],
                    weights_i[w_i] if weights_i is not None else None, 
                    cont_i[w_i], 
                    delta_i[w_i],
                    order_i, 
                    ivar_i[w_i] if ivar_i is not None else None,
                    exposures_diff_i[w_i] if exposures_diff_i is not None else None, 
                    mean_snr_i, mean_reso_i, mean_z_i,
                    resolution_matrix_i if resolution_matrix_i is not None else None,
                    mean_resolution_matrix_i if mean_resolution_matrix_i is not None else None,
                    mean_reso_pix_i,
                ))

        return deltas

    def project(self):
        """Project the delta field.

        The projection gets rid of the distortion caused by the continuum
        fitiing. See equations 5 and 6 of du Mas des Bourboux et al. 2020
        """
        # 2nd term in equation 6
        sum_weights = np.sum(self.weights)
        if sum_weights > 0.0:
            mean_delta = np.average(self.delta, weights=self.weights)
        else:
            # should probably write a warning
            return

        # 3rd term in equation 6
        res = 0
        if (self.order == 1) and self.delta.shape[0] > 1:
            mean_log_lambda = np.average(self.log_lambda, weights=self.weights)
            meanless_log_lambda = self.log_lambda - mean_log_lambda
            mean_delta_log_lambda = (
                np.sum(self.weights * self.delta * meanless_log_lambda) /
                np.sum(self.weights * meanless_log_lambda**2))
            res = mean_delta_log_lambda * meanless_log_lambda
        elif self.order == 1:
            res = self.delta

        self.delta -= mean_delta + res

    def rebin(self, factor, dwave=0.8):
        """Rebin deltas by an integer factor

        Args:
            factor: int
                Factor to rebin deltas (new_bin_size = factor * old_bin_size)
            dwave: float
                Delta lambda of original deltas
        """
        wave = 10**np.array(self.log_lambda)

        start = wave.min() - dwave / 2
        num_bins = np.ceil(((wave[-1] - wave[0]) / dwave + 1) / factor)

        edges = np.arange(num_bins) * dwave * factor + start

        new_indx = np.searchsorted(edges, wave)

        binned_delta = np.bincount(new_indx, weights=self.delta*self.weights,
                                   minlength=edges.size+1)[1:-1]
        binned_weight = np.bincount(new_indx, weights=self.weights, minlength=edges.size+1)[1:-1]

        mask = binned_weight != 0
        binned_delta[mask] /= binned_weight[mask]

        new_wave = (edges[1:] + edges[:-1]) / 2

        self.log_lambda = np.log10(new_wave[mask])
        self.delta = binned_delta[mask]
        self.weights = binned_weight[mask]
