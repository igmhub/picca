"""This module defines the class DesiData to load DESI data
"""
import logging
import os

import fitsio
import healpy
import numpy as np

from picca.delta_extraction.astronomical_objects.desi_forest import DesiForest
from picca.delta_extraction.astronomical_objects.desi_pk1d_forest import DesiPk1dForest
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data_catalogues.desi_data import DesiData, defaults, accepted_options
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.utils_pk1d import spectral_resolution_desi, exp_diff_desi

accepted_options = sorted(
    list(set(accepted_options + ["use non-coadded spectra"])))

defaults = defaults.copy()
defaults.update({
    "use non-coadded spectra": False,
})


class DesiHealpix(DesiData):
    """Reads the spectra from DESI using healpix mode and formats its data as a
    list of Forest instances.

    Methods
    -------
    filter_forests (from Data)
    set_blinding (from Data)
    __init__
    read_data
    read_file

    Attributes
    ----------
    analysis_type: str (from Data)
    Selected analysis type. Current options are "BAO 3D" or "PK 1D"

    forests: list of Forest (from Data)
    A list of Forest from which to compute the deltas.

    min_num_pix: int (from Data)
    Minimum number of pixels in a forest. Forests with less pixels will be dropped.

    blinding: str (from DesiData)
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    catalogue: astropy.table.Table (from DesiData)
    The quasar catalogue

    input_directory: str (from DesiData)
    Directory to spectra files.

    in_nside: 64 or 16
    Nside used in the folder structure (64 for data and 16 for mocks)

    logger: logging.Logger
    Logger object
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.logger = logging.getLogger(__name__)

        self.use_non_coadded_spectra = None
        self.__parse_config(config)
        #init of DesiData needs to come last, as it contains the actual data reading and thus needs all config
        super().__init__(config)



    def __parse_config(self, config):
        """Parse the configuration options

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class

        Raise
        -----
        DataError upon missing required variables
        """
        self.use_non_coadded_spectra = config.getboolean(
            "use non-coadded spectra")
        if self.use_non_coadded_spectra is None:
            raise DataError(
                "Missing argument 'use non-coadded spectra' required by DesiHealpix"
            )

    def read_data(self):
        """Read the data.

        Method used to read healpix-based survey data.

        Return
        ------
        is_mock: bool
        False as DESI data are not mocks

        is_sv: bool
        True if all the read data belong to SV. False otherwise

        Raise
        -----
        DataError if no quasars were found
        """
        in_nside = 64

        healpix = [
            healpy.ang2pix(in_nside,
                           np.pi / 2 - row["DEC"],
                           row["RA"],
                           nest=True)
            for row in self.catalogue
        ]
        self.catalogue["HEALPIX"] = healpix
        self.catalogue.sort("HEALPIX")
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        forests_by_targetid = {}
        is_sv = True
        for (index,
             (healpix,
              survey)), group in zip(enumerate(grouped_catalogue.groups.keys),
                                     grouped_catalogue.groups):

            if survey not in ["sv", "sv1", "sv2", "sv3"]:
                is_sv = False

            input_directory = f'{self.input_directory}/{survey}/dark'
            coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"
            filename = (
                f"{input_directory}/{healpix//100}/{healpix}/{coadd_name}-{survey}-"
                f"dark-{healpix}.fits")

            #TODO: not sure if we want the dark survey to be hard coded in here, probably won't run on anything else, but still

            self.logger.progress(
                f"Read {index} of {len(grouped_catalogue.groups.keys)}. "
                f"num_data: {len(forests_by_targetid)}")

            self.read_file(filename, group, forests_by_targetid)

        if len(forests_by_targetid) == 0:
            raise DataError("No Quasars found, stopping here")

        self.forests = list(forests_by_targetid.values())

        return False, is_sv

    def read_file(self, filename, catalogue, forests_by_targetid):
        """Read the spectra and formats its data as Forest instances.

        Arguments
        ---------
        filename: str
        Name of the file to read

        catalogue: astropy.table.Table
        The quasar catalogue fragment associated with this file

        forests_by_targetid: dict
        Dictionary were forests are stored. Its content is modified by this
        function with the new forests.

        Raise
        -----
        DataError if the analysis type is PK 1D and resolution data is not present
        """
        try:
            hdul = fitsio.FITS(filename)
        except IOError:
            self.logger.warning(f"Error reading  {filename}. Ignoring file")
            return
        # Read targetid from fibermap to match to catalogue later
        fibermap = hdul['FIBERMAP'].read()
        targetid_spec = fibermap["TARGETID"]
        # First read all wavelength, flux, ivar, mask, and resolution
        # from this file
        spectrographs_data = {}
        colors = ["B", "R"]
        if "Z_FLUX" in hdul:
            colors.append("Z")

        reso_from_truth = False
        no_scores_available = False
        for color in colors:
            spec = {}
            try:
                spec["WAVELENGTH"] = hdul[f"{color}_WAVELENGTH"].read()
                spec["FLUX"] = hdul[f"{color}_FLUX"].read()
                spec["IVAR"] = (hdul[f"{color}_IVAR"].read() *
                                (hdul[f"{color}_MASK"].read() == 0))
                w = np.isnan(spec["FLUX"]) | np.isnan(spec["IVAR"])
                for key in ["FLUX", "IVAR"]:
                    spec[key][w] = 0.
                if self.analysis_type == "PK 1D":
                    if self.use_non_coadded_spectra and "SCORES" in hdul:
                        # Calibration factor given in https://desi.lbl.gov/trac/browser/code/desimodel/trunk/data/tsnr/
                        spec['TEFF_LYA'] = 11.80090901380597 * hdul['SCORES'][f'TSNR2_LYA_{color}'].read()
                    else:
                        spec['TEFF_LYA'] = np.ones(spec["FLUX"].shape[0])
                        if self.use_non_coadded_spectra and not no_scores_available:
                            self.logger.warning("SCORES are missing, Teff information (and thus DIFF) will be garbage")
                        no_scores_available=True

                    if f"{color}_RESOLUTION" in hdul:
                        spec["RESO"] = hdul[f"{color}_RESOLUTION"].read()
                    else:
                        basename_truth=os.path.basename(filename).replace('spectra-','truth-')
                        pathname_truth=os.path.dirname(filename)
                        filename_truth=f"{pathname_truth}/{basename_truth}"
                        if os.path.exists(filename_truth):
                            with fitsio.FITS(filename_truth) as hdul_truth:
                                spec["RESO"] = hdul_truth[f"{color}_RESOLUTION"].read()
                            if not reso_from_truth:
                                self.logger.debug("no resolution in files, reading from truth files")
                            reso_from_truth=True
                        else:
                            raise DataError(
                                "Error while reading {color} band from "
                                "{filename}. Analysis type is  'PK 1D', "
                                "but file does not contain HDU "
                                f"'{color}_RESOLUTION' ")
                spectrographs_data[color] = spec
            except OSError:
                self.logger.warning(
                    f"Error while reading {color} band from {filename}."
                    "Ignoring color.")
        hdul.close()

        # Loop over quasars in catalogue inside this healpixel
        for row in catalogue:
            # Find which row in tile contains this quasar
            # It should be there by construction
            targetid = row["TARGETID"]
            w_t = np.where(targetid_spec == targetid)[0]
            if len(w_t) == 0:
                self.logger.warning(
                    f"Error reading {targetid}. Ignoring object")
                continue
            if len(w_t) > 1:
                self.logger.warning(
                    "Warning: more than one spectrum in this file "
                    f"for {targetid}")
            else:
                w_t = w_t[0]
            # Construct DesiForest instance
            # Fluxes from the different spectrographs will be coadded
            for spec in spectrographs_data.values():
                if self.use_non_coadded_spectra:
                    ivar = np.atleast_2d(spec['IVAR'][w_t])
                    ivar_coadded_flux = np.atleast_2d(ivar*spec['FLUX'][w_t]).sum(axis=0)
                    ivar = ivar.sum(axis=0)
                    flux = (ivar_coadded_flux / ivar)
                else:
                    flux = spec['FLUX'][w_t].copy()
                    ivar = spec['IVAR'][w_t].copy()

                args = {
                    "flux": flux,
                    "ivar": ivar,
                    "targetid": targetid,
                    "ra": row['RA'],
                    "dec": row['DEC'],
                    "z": row['Z'],
                }
                args["log_lambda"] = np.log10(spec['WAVELENGTH'])

                if self.analysis_type == "BAO 3D":
                    forest = DesiForest(**args)
                elif self.analysis_type == "PK 1D":
                    if self.use_non_coadded_spectra and not no_scores_available:
                        exposures_diff = exp_diff_desi(spec, w_t)
                    else:
                        exposures_diff = np.zeros(spec['WAVELENGTH'].shape)
                    if not reso_from_truth:
                        if len(spec['RESO'][w_t].shape)<3:
                            reso_sum = spec['RESO'][w_t].copy()
                        else:
                            reso_sum = spec['RESO'][w_t].sum(axis=0)
                    else:
                        reso_sum = spec['RESO'][:, :]
                    reso_in_pix, reso_in_km_per_s = spectral_resolution_desi(
                        reso_sum, spec['WAVELENGTH'])
                    args["exposures_diff"] = exposures_diff
                    args["reso"] = reso_in_km_per_s
                    args["resolution_matrix"] = reso_sum
                    args["reso_pix"] = reso_in_pix

                    forest = DesiPk1dForest(**args)
                else:
                    raise DataError(
                        "Unkown analysis type. Expected 'BAO 3D'"
                        f"or 'PK 1D'. Found '{self.analysis_type}'")

                # rebin arrays
                # this needs to happen after all arrays are initialized by
                # Forest constructor
                forest.rebin()

                # keep the forest
                if targetid in forests_by_targetid:
                    forests_by_targetid[targetid].coadd(forest)
                else:
                    forests_by_targetid[targetid] = forest
