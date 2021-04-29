"""This module defines the abstract class ExpectedFlux from which all
classes computing the mean expected flux must inherit. The mean expected flux
is the product of the unabsorbed quasar continuum and the mean transmission
"""
from picca.delta_extraction.errors import ExpectedFluxError

defaults = {
    "minimum number pixels in forest": 50,
}

class ExpectedFlux:
    """Abstract class from which all classes computing the expected flux
    must inherit. Classes that inherit from this should be initialized using
    a configparser.SectionProxy instance.

    Methods
    -------
    __init__
    compute_expected_flux
    extract_deltas

    Attributes
    ----------
    los_ids: dict
    A dictionary containing the mean expected flux fraction, the weights, and
    the inverse variance for each line of sight. Keys are the identifier for the
    line of sight and values are dictionaries with the keys "mean expected flux",
    and "weights" pointing to the respective arrays. Arrays must have the same
    size as the flux array for the corresponding line of sight forest instance.
    """
    def __init__(self):
        """Initialize class instance"""
        self.los_ids = {}

    # pylint: disable=no-self-use
    # this method should use self in child classes
    def compute_expected_flux(self, forests, out_dir):
        """Compute the mean expected flux of the forests.
        This includes the quasar continua and the mean transimission. It is
        computed iteratively following as explained in du Mas des Bourboux et
        al. (2020)

        Arguments
        ---------
        forests: List of Forest
        A list of Forest from which to compute the deltas.

        out_dir: str
        Directory where expected flux information will be saved

        Raise
        -----
        MeanExpectedFluxError if function was not overloaded by child class
        """
        raise ExpectedFluxError("Function 'compute_expected_flux' was not "
                                "overloaded by child class")

    def extract_deltas(self, forest):
        """Apply the continuum to compute the delta field

        Arguments
        ---------
        forest: Forest
        A Forest instance to which the continuum is applied
        """
        if self.los_ids.get(forest.los_id) is not None:
            continuum = self.los_ids.get(forest.los_id).get("mean expected flux")
            forest.deltas = forest.flux/continuum - 1
            forest.weights = self.los_ids.get(forest.los_id).get("weights")
            forest.ivar = self.los_ids.get(forest.los_id).get("ivar")
