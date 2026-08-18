"""This module defines the DeltaWriter class, which encapsulates
the logic for writing delta files in various formats (ImageHDU and BinTableHDU).

It writes weight components (VAR_LSS, ETA, FUDGE, VAR_PIPE) alongside
the existing WEIGHT field for backward compatibility.
"""
import logging

import numpy as np
import fitsio

from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.errors import DataError

accepted_save_format = ["BinTableHDU", "ImageHDU"]


class DeltaWriter:
    """Class to handle writing of delta files.

    This writer supports both ImageHDU and BinTableHDU formats.
    It writes weight components (VAR_LSS, ETA, FUDGE, VAR_PIPE)
    alongside the existing WEIGHT field for backward compatibility.

    Methods
    -------
    __init__
    save_deltas_one_healpix
    _save_image
    _save_table
    _write_weight_components_image

    Attributes
    ----------
    logger: logging.Logger
    Logger object
    """

    def __init__(self):
        """Initialize class instance"""
        self.logger = logging.getLogger(__name__)

    def save_deltas_one_healpix(self, out_dir, healpix, forests, save_format):
        """Save the deltas that belong to one healpix.

        Arguments
        ---------
        out_dir: str
        Parent directory to save deltas.

        healpix: int
        Healpix number

        forests: List of Forest
        List of forests to save into one file.

        save_format: str
        Format to store delta into ("ImageHDU" or "BinTableHDU")

        Returns
        ---------
        forests: List of Forest
        List of forests (for rejection log tracking).
        """
        if save_format == "BinTableHDU":
            return self._save_table(out_dir, healpix, forests)
        if save_format == "ImageHDU":
            return self._save_image(out_dir, healpix, forests)
        raise DataError("Invalid format. Expected one of " +
                        " ".join(accepted_save_format) +
                        f" Found: {save_format}")

    def _save_table(self, out_dir, healpix, forests):
        """Save deltas in BinTableHDU format.

        Forest.get_data() already includes weight components when available,
        so backward compatibility is ensured.

        Arguments
        ---------
        out_dir: str
        Parent directory to save deltas.

        healpix: int
        Healpix number

        forests: List of Forest
        List of forests to save into one file.

        Returns
        ---------
        forests: List of Forest
        """
        results = fitsio.FITS(f"{out_dir}/Delta/delta-{healpix}.fits.gz",
                              'rw',
                              clobber=True)

        for forest in forests:
            header = forest.get_header()
            cols, names, units, comments = forest.get_data()
            results.write(cols,
                          names=names,
                          header=header,
                          comment=comments,
                          units=units,
                          extname=str(forest.los_id))

        results.close()

        return forests

    def _save_image(self, out_dir, healpix, forests):
        """Save deltas in ImageHDU format.

        Writes all standard HDUs (LAMBDA, METADATA, DELTA/DELTA_BLIND,
        WEIGHT, CONT) plus weight component HDUs (VAR_LSS, ETA, FUDGE,
        VAR_PIPE) when available.

        Arguments
        ---------
        out_dir: str
        Parent directory to save deltas.

        healpix: int
        Healpix number

        forests: List of Forest
        List of forests to save into one file.

        Returns
        ---------
        forests: List of Forest
        """
        results = fitsio.FITS(f"{out_dir}/Delta/delta-{healpix}.fits.gz",
                              'rw',
                              clobber=True)

        results.write(None)  # Primary HDU

        # LAMBDA HDU
        hdr = fitsio.FITSHDR()
        hdr.add_record({
            "name": "BUNIT",
            "value": "Angstrom",
            "comment": "wavelength units",
        })
        hdr.add_record({
            "name": "WAVE_SOLUTION",
            "value": Forest.wave_solution,
            "comment": "chosen wavelength solution",
        })
        if Forest.wave_solution == "log":
            hdr.add_record({
                "name": "DELTA_LOG_LAMBDA",
                "value": round(Forest.log_lambda_grid[1] -
                               Forest.log_lambda_grid[0], 2),
                "comment": "pixel step",
            })
        elif Forest.wave_solution == "lin":
            hdr.add_record({
                "name": "DELTA_LAMBDA",
                "value": round(10**Forest.log_lambda_grid[1] -
                               10**Forest.log_lambda_grid[0], 2),
                "comment": "pixel step",
            })
        else:
            raise DataError("Error in DeltaWriter._save_image. "
                            "Class variable 'wave_solution' "
                            "must be either 'lin' or 'log'. "
                            f"Found: '{Forest.wave_solution}'")
        results.write(
            10**Forest.log_lambda_grid,
            extname="LAMBDA",
            header=hdr
        )
        results["LAMBDA"].write_comment("Wavelength grid")
        results["LAMBDA"].write_checksum()

        # METADATA HDU
        hdr = fitsio.FITSHDR()
        hdr.add_record({
            "name": "BLINDING",
            "value": Forest.blinding,
            "comment": "blinding scheme used",
        })
        results.write(
            np.array(
                [tuple(forest.get_metadata()) for forest in forests],
                dtype=forests[0].get_metadata_dtype(),
            ),
            header=hdr,
            units=forests[0].get_metadata_units(),
            extname="METADATA")
        results["METADATA"].write_comment("Per-forest metadata")
        results["METADATA"].write_checksum()

        # DELTA HDU
        delta = np.full((len(forests), len(Forest.log_lambda_grid)), np.nan)
        for i, forest in enumerate(forests):
            delta[i][forest.log_lambda_index] = forest.deltas

        hdr = fitsio.FITSHDR()
        hdr.add_record({
            "name": "BUNIT",
            "value": "",
            "comment": "delta units (unitless)",
        })
        delta_label = "DELTA" if Forest.blinding == "none" else "DELTA_BLIND"
        results.write(
            delta,
            header=hdr,
            extname=delta_label)
        results[delta_label].write_comment("Flux transmission field in "
                                           "wavelength bins")
        results[delta_label].write_checksum()

        # WEIGHT HDU (always written for backward compatibility)
        weight = np.full((len(forests), len(Forest.log_lambda_grid)), np.nan)
        for i, forest in enumerate(forests):
            weight[i][forest.log_lambda_index] = forest.weights

        hdr = fitsio.FITSHDR()
        hdr.add_record({
            "name": "BUNIT",
            "value": "",
            "comment": "weight units (unitless)",
        })
        results.write(
            weight,
            extname="WEIGHT",
        )
        results["WEIGHT"].write_comment("Weights in wavelength bins")
        results["WEIGHT"].write_checksum()

        # CONT HDU
        continuum = np.full((len(forests), len(Forest.log_lambda_grid)),
                            np.nan)
        for i, forest in enumerate(forests):
            continuum[i][forest.log_lambda_index] = forest.continuum

        hdr = fitsio.FITSHDR()
        hdr.add_record({
            "name": "BUNIT",
            "value": Forest.flux_units,
            "comment": "flux units",
        })
        results.write(
            continuum,
            extname="CONT",
        )
        results["CONT"].write_comment("Quasar continuum in wavelength bins")
        results["CONT"].write_checksum()

        # Weight components HDUs
        self._write_weight_components_image(results, forests)

        return forests

    def _write_weight_components_image(self, results, forests):
        """Write weight component HDUs (VAR_LSS, ETA, FUDGE, VAR_PIPE)
        when available on the forest objects.

        Arguments
        ---------
        results: fitsio.FITS
        Open FITS file to write to.

        forests: List of Forest
        List of forests.
        """
        components = [
            ("VAR_LSS", "var_lss", "Variance of the LSS"),
            ("ETA", "eta", "Noise correction factor eta"),
            ("FUDGE", "fudge", "Fudge contribution to variance"),
            ("VAR_PIPE", "var_pipe", "Pipeline variance"),
        ]

        for extname, attr, comment in components:
            if getattr(forests[0], attr, None) is not None:
                data = np.full(
                    (len(forests), len(Forest.log_lambda_grid)), np.nan)
                for i, forest in enumerate(forests):
                    values = getattr(forest, attr, None)
                    if values is not None:
                        data[i][forest.log_lambda_index] = values

                hdr = fitsio.FITSHDR()
                hdr.add_record({
                    "name": "BUNIT",
                    "value": "",
                    "comment": f"{extname} units (unitless)",
                })
                results.write(
                    data,
                    header=hdr,
                    extname=extname,
                )
                results[extname].write_comment(
                    f"{comment} in wavelength bins")
                results[extname].write_checksum()
