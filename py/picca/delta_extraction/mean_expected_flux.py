"""This module defines the abstract class MeanExpectedFlux from which all
classes computing the mean expected flux must inherit. The mean expected flux
is the product of the unabsorbed quasar continuum and the mean transmission
"""
defaults = {
    "minimum number pixels in forest": 50,
}

class MeanExpectedFlux:
    """Abstract class from which all classes computing the mean expected flux
    must inherit. Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    __init__
    extract_deltas

    Attributes
    ----------
    los_ids: dict
    A dictionary containing the mean expected flux fraction, the weights, and
    the inverse variance for each line of sight. Keys are the identifier for the
    line of sight and values are dictionaries with the keys "mean expected flux",
    "weights" and "inverse variance" pointing to the respective arrays. Arrays
    must have the same size as the flux array for the corresponding line of
    sight forest instance.
    """
    def __init__(self):
        """Initialize class instance"""
        self.los_ids = {}

    def extract_deltas(self, forest):
        """Applies the continuum to compute the delta field

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the continuum is applied
        """
        if self.los_ids.get(forest.los_id) is not None:
            forest.continuum = self.los_ids.get(forest.los_id).get("mean expected flux")
            forest.deltas = forest.flux/forest.continuum - 1
            forest.weights = self.los_ids.get(forest.los_id).get("weights")
            forest.ivar = self.los_ids.get(forest.los_id).get("inverse variance")
