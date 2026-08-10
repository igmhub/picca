"""MPI-parallel expected flux: a mixin plus a factory that generates the MPI
version of any ExpectedFlux subclass on demand.

Why a mixin works for every estimator
-------------------------------------
The iterative continuum fit needs three sample-wide quantities: the mean
continuum, the variance functions (eta/var_lss/fudge) and the delta stack. Each
is a *sum* of per-forest accumulators followed by a normalisation, and the base
classes route every such accumulator through the ``ExpectedFlux.reduce_sum`` hook
(identity in the serial case). Overriding that single hook with an MPI
Allreduce-sum gives every estimator the correct sample-wide statistics with no
further change; the per-bin minimisation then runs on identical inputs on every
rank. The per-forest fit parameters (FIT_METADATA) are instead gathered on
rank 0, which is the only rank that writes the ``delta_attributes`` log.

Why a factory
-------------
Since ``(MpiExpectedFluxMixin, BaseClass)`` is pure boilerplate, the classes are
built here dynamically rather than one file per estimator: accessing
``<BaseName>Mpi`` on this module returns the generated class. A new estimator
therefore gets its MPI version for free -- if ``FooExpectedFlux`` exists in
``foo_expected_flux.py``, ``type = FooExpectedFluxMpi`` just works. The config
layer routes any ``type`` ending in ``Mpi`` here (see
Config.__format_expected_flux_section), and class_from_string reads the options
from the ``default_options``/``accepted_options`` class attributes set below.

Note: this only covers the expected-flux side, where the MPI version is pure
boilerplate. The readers (DesiHealpixMpi, ...) are not generated here because
their MPI version carries real logic (healpix decomposition, global find_nside,
redistribution before writing) that differs from reader to reader.
"""
import importlib
import re

from picca.delta_extraction.expected_flux import ExpectedFlux
from picca.delta_extraction.utils_mpi import allreduce_sum, get_comm

MPI_SUFFIX = "Mpi"

# cache of already-built classes, keyed by class name
_MPI_CLASSES = {}


class MpiExpectedFluxMixin:
    """Mixin adding MPI reductions to an ExpectedFlux subclass. Must come first
    in the base list so its methods take precedence.

    Attributes
    ----------
    comm: mpi4py.MPI.Comm
    The world communicator

    mpi_rank: int
    Rank of the current process

    mpi_size: int
    Number of MPI processes
    """

    def __init__(self, config):
        """Set up the communicator, then delegate to the wrapped estimator."""
        self.comm = get_comm()
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()
        super().__init__(config)

    def reduce_sum(self, array):
        """Combine a per-rank accumulator into the global sample sum (available
        on every rank)."""
        return allreduce_sum(self.comm, array)

    def _gather_continuum_fit_parameters(self):
        """Gather the per-line-of-sight continuum fit parameters on rank 0.

        Unlike the stacks, the FIT_METADATA HDU is a per-forest collection (one
        row per line of sight), so it is gathered and merged rather than summed.
        Estimators without ``continuum_fit_parameters`` are left untouched.
        """
        parameters = getattr(self, "continuum_fit_parameters", None)
        if parameters is None:
            return
        gathered = self.comm.gather(parameters, root=0)
        if self.mpi_rank == 0:
            merged = {}
            for rank_parameters in gathered:
                merged.update(rank_parameters)
            self.continuum_fit_parameters = merged

    def save_iteration_step(self, iteration):
        """Gather the per-forest fit parameters, then let rank 0 write the log.

        Arguments
        ---------
        iteration: int
        Iteration number. -1 for the final iteration.
        """
        self._gather_continuum_fit_parameters()
        if self.mpi_rank == 0:
            super().save_iteration_step(iteration)
        self.comm.Barrier()


def _base_module_name(base_class_name):
    """Return the module holding the base estimator, using the same class-name
    -> module-name convention as the configuration layer.
    """
    snake = re.sub('(?<!^)(?=[A-Z])', '_', base_class_name).lower()
    return f"picca.delta_extraction.expected_fluxes.{snake}"


def build_mpi_expected_flux(name):
    """Build (and cache) the MPI version of an ExpectedFlux subclass.

    Arguments
    ---------
    name: str
    Name of the MPI class, e.g. "Dr16ExpectedFluxMpi". Must end in "Mpi".

    Return
    ------
    The generated class ``(MpiExpectedFluxMixin, base)``.

    Raise
    -----
    AttributeError if the name does not end in "Mpi"
    ImportError if the base estimator module cannot be imported
    TypeError if the resolved base class is not an ExpectedFlux subclass
    """
    if name in _MPI_CLASSES:
        return _MPI_CLASSES[name]
    if not name.endswith(MPI_SUFFIX):
        raise AttributeError(
            f"module {__name__!r} only provides MPI classes (names ending in "
            f"{MPI_SUFFIX!r}); got {name!r}")

    base_name = name[:-len(MPI_SUFFIX)]
    base_module = importlib.import_module(_base_module_name(base_name))
    base_class = getattr(base_module, base_name)

    if not issubclass(base_class, ExpectedFlux):
        raise TypeError(
            f"Cannot build {name}: {base_name} is not an ExpectedFlux subclass")

    mpi_class = type(
        name,
        (MpiExpectedFluxMixin, base_class),
        {
            "__doc__": (f"MPI-parallel version of {base_name}, generated by "
                        f"{__name__}. See MpiExpectedFluxMixin."),
            "__module__": __name__,
            # picked up by class_from_string as a fallback for the module-level
            # "defaults"/"accepted_options" that this factory module cannot hold
            # per class
            "default_options": getattr(base_module, "defaults", {}),
            "accepted_options": getattr(base_module, "accepted_options", []),
        },
    )

    _MPI_CLASSES[name] = mpi_class
    return mpi_class


def __getattr__(name):
    """PEP 562 module-level hook: build ``<BaseName>Mpi`` on first access."""
    return build_mpi_expected_flux(name)
