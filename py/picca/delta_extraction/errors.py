"""This module define the different Error and Warning types related to the
package picca.delta_extraction
"""
class AstronomicalObjectError(Exception):
    """
        Exceptions occurred in class AstronomicalObject
    """

class ConfigError(Exception):
    """
        Exceptions occurred in class Config
    """

class ConfigWarning(UserWarning):
    """
        Warnings occurred in class Config
    """

class CorrectionError(Exception):
    """
        Exceptions occurred in class Correction
    """

class DataError(Exception):
    """
        Exceptions occurred in class Data
    """

class DataWarning(UserWarning):
    """
        Warnings occurred in class Data
    """

class DeltaExtractionError(Exception):
    """
        Exceptions occurred in class Survey
    """

class ExpectedFluxError(Exception):
    """
        Exceptions occurred in class ExpectedFlux
    """

class MaskError(Exception):
    """
        Exceptions occurred in class Mask
    """

class QuasarCatalogueError(Exception):
    """
        Exceptions occurred in class Mask
    """

class QuasarCatalogueWarning(UserWarning):
    """
        Warnings occurred in class Data
    """


if __name__ == '__main__':
    raise Exception()
