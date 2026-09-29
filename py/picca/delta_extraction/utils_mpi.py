"""This module gathers the MPI helpers used by the MPI-enabled delta
extraction classes (DesiHealpixMpi, Dr16ExpectedFluxMpi, ...).

The design principle is domain decomposition by healpix: each MPI rank reads and
processes a fixed subset of the input healpix files, and the ranks only
communicate at the few points where a global (sample-wide) quantity is needed.
Those points are:

- the automatic determination of nside (needs the global object count);
- the three global reductions of the iterative continuum fit (mean continuum,
  variance functions eta/var_lss/fudge, and the delta stack), all of which are
  sums of per-forest accumulators and are therefore combined with an
  Allreduce-sum;
- the redistribution of the fitted forests by output healpix so that each rank
  writes a disjoint set of delta files.

mpi4py is imported lazily so that non-MPI installations are unaffected.
"""
import numpy as np

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover
    MPI = None


def get_comm():
    """Return the world communicator.

    Raise
    -----
    ImportError if mpi4py is not installed
    """
    if MPI is None:
        raise ImportError(
            "mpi4py is required by the MPI delta extraction classes but could "
            "not be imported")
    return MPI.COMM_WORLD


def allreduce_sum(comm, array):
    """Sum a numpy array element-wise across all ranks and return the result on
    every rank.

    Arguments
    ---------
    comm: mpi4py.MPI.Comm
    The communicator

    array: numpy.ndarray
    The local accumulator. It is copied to a contiguous float64 buffer before
    the reduction.

    Return
    ------
    total: numpy.ndarray
    The element-wise sum over all ranks, same shape as the input.
    """
    sendbuf = np.ascontiguousarray(array, dtype=np.float64)
    recvbuf = np.empty_like(sendbuf)
    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    return recvbuf.reshape(array.shape)


def redistribute_by_key(comm, items, key_of):
    """Redistribute a list of picklable items across ranks by a routing key.

    Each item is sent to the rank ``key_of(item) % size``; every rank returns
    the concatenation of the items routed to it. This is used to gather all the
    forests of a given output healpix onto a single rank so that the delta files
    are written without collisions.

    Arguments
    ---------
    comm: mpi4py.MPI.Comm
    The communicator

    items: list
    The local items to redistribute

    key_of: callable
    Maps an item to an integer routing key

    Return
    ------
    received: list
    The items routed to this rank.
    """
    size = comm.Get_size()
    outgoing = [[] for _ in range(size)]
    for item in items:
        outgoing[int(key_of(item)) % size].append(item)

    incoming = comm.alltoall(outgoing)

    received = []
    for chunk in incoming:
        received.extend(chunk)
    return received
