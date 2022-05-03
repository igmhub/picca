"""This module defines the abstract class QuasarCatalogue from which all
classes loading quasar catalogues must inherit
"""
import healpy
import numpy as np

from picca.delta_extraction.errors import QuasarCatalogueError

accepted_options = ["lambda max", "lambda max rest frame", "lambda min",
                    "lambda min rest frame", "max num spec", "z max", "z min"]

class QuasarCatalogue:
    """Abstract class to contain a general quasar catalogue

    Methods
    -------
    __init__
    trim_catalogue

    Attributes
    ----------
    catalogue: astropy.table.Table
    The quasar catalogue

    max_num_spec: int or None
    Maximum number of spectra to read. None for no maximum

    z_max: float
    Maximum redshift. Quasars with redshifts higher than or equal to
    z_max will be discarded

    z_min: float
    Minimum redshift. Quasars with redshifts lower than z_min will be
    discarded
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        # load variables from config
        self.max_num_spec = config.getint("max num spec")

        self.z_min = config.getfloat("z min")
        if self.z_min is None:
            if (config.getfloat("lambda min") is None or
                    config.getfloat("lambda max rest frame") is None):
                raise QuasarCatalogueError("Missing argument 'z min' "
                                           "required by QuasarCatalogue")
            self.z_min = max(0., (config.getfloat("lambda min") /
                                  config.getfloat("lambda max rest frame") -
                                  1.))

        self.z_max = config.getfloat("z max")
        if self.z_max is None:
            if (config.getfloat("lambda max") is None or
                    config.getfloat("lambda min rest frame") is None):
                raise QuasarCatalogueError("Missing argument 'z max' "
                                           "required by QuasarCatalogue")
            self.z_max = max(0., (config.getfloat("lambda max") /
                                  config.getfloat("lambda min rest frame") -
                                  1.))
        self.catalogue = None

    def trim_catalogue(self):
        """Trim the current catalogue.

        Keep only max_num_spec objects that are close by. This is achieved by
        first ordering the catalogue through healpix

        If self.catalogue or self.max_num_spec are None, then do nothing
        """
        if self.max_num_spec is not None and self.catalogue is not None:
            # sort forests by healpix
            healpix = [healpy.ang2pix(16, np.pi / 2 - row["DEC"], row["RA"])
                       for row in self.catalogue]
            self.catalogue["healpix"] = healpix
            self.catalogue.sort("healpix")
            self.catalogue = self.catalogue[:self.max_num_spec]
