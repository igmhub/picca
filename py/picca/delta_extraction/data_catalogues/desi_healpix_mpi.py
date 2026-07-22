"""This module defines the class DesiHealpixMpi, the MPI-parallel reader for
DESI healpix data.

The workspace is decomposed by healpix: rank ``r`` owns every reading healpix
``h`` with ``h % size == r`` and reads, corrects, masks, filters, continuum-fits
and writes only the forests of the pixels it owns. Because all the surveys of a
given healpix are owned by the same rank, forests sharing a targetid are coadded
locally exactly as in the serial reader, so no communication is needed while
reading.

The ranks only communicate at the points where a genuinely global quantity is
required:

- ``find_nside``:   the automatic nside needs the global object count, so the
                    coordinates are gathered on rank 0, the nside is decided
                    there and broadcast back;
- the continuum fit: the three sample-wide stacks (mean continuum, variance
                    functions, delta stack) are Allreduce-summed. This is handled
                    entirely by Dr16ExpectedFluxMpi through the ``reduce_sum``
                    hook, not here;
- ``save_deltas``:  the output nside is coarser than the reading nside, so a
                    coarse output pixel can gather forests from several reading
                    pixels living on different ranks. The forests are therefore
                    redistributed by output healpix so that each rank writes a
                    disjoint set of delta files.
"""
import os

import healpy
import numpy as np

from picca.delta_extraction.data_catalogues.desi_data import (
    merge_new_forest,
    verify_exposures_shape,
)
from picca.delta_extraction.data_catalogues.desi_healpix import (  # pylint: disable=unused-import
    DesiHealpix,
    DesiHealpixFileHandler,
    accepted_options,
    defaults,
)
from picca.delta_extraction.errors import DataError
from picca.delta_extraction.utils_mpi import get_comm, redistribute_by_key


class DesiHealpixMpi(DesiHealpix):
    """Reads the spectra from DESI using healpix mode, decomposing the healpix
    pixels over MPI ranks.

    Only the parallelisation strategy differs from DesiHealpix: get_filename and
    the PK 1D setup are inherited, read_data reads the pixels owned by this rank,
    find_nside and save_deltas add the required communication.

    Methods
    -------
    (see DesiHealpix in py/picca/delta_extraction/data_catalogues/desi_healpix.py)
    __init__
    read_data
    find_nside
    save_deltas

    Attributes
    ----------
    (see DesiHealpix in py/picca/delta_extraction/data_catalogues/desi_healpix.py)

    comm: mpi4py.MPI.Comm
    The world communicator

    mpi_rank: int
    Rank of the current process

    mpi_size: int
    Number of MPI processes
    """

    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        self.comm = get_comm()
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        # DesiHealpix.__init__ reads the catalogue and calls read_data (below)
        super().__init__(config)

        # write a per-rank rejection log so that the ranks do not write to the
        # same file concurrently
        if self.rejection_log is not None:
            directory = os.path.dirname(self.rejection_log.file)
            basename = os.path.basename(self.rejection_log.file)
            name = basename.split(".")[0]
            extension = basename[len(name):]
            self.rejection_log.file = os.path.join(
                directory, f"{name}_rank{self.mpi_rank}{extension}")

    def read_data(self):
        """Read the data owned by this rank.

        Every rank builds the same list of files, then reads only the files
        whose healpix it owns (``healpix % size == rank``). Forests are not
        gathered: each rank keeps and processes its own subset.

        Return
        ------
        is_mock: bool
        False for DESI data and True for mocks

        Raise
        -----
        DataError if no quasars were found on any rank
        """
        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        is_mock = False

        arguments = []
        healpix_of_file = []
        for group in grouped_catalogue.groups:
            healpix, survey = group["HEALPIX", "SURVEY"][0]

            filename, is_mock_aux = self.get_filename(survey, healpix)
            if is_mock_aux:
                is_mock = True

            arguments.append((filename, group))
            healpix_of_file.append(int(healpix))

        num_owned = sum(healpix % self.mpi_size == self.mpi_rank
                        for healpix in healpix_of_file)
        if self.mpi_rank == 0:
            self.logger.info(
                f"reading data from {len(arguments)} files "
                f"decomposed by healpix over {self.mpi_size} MPI ranks")
        self.logger.progress(
            f"rank {self.mpi_rank} owns {num_owned} files")

        reader = DesiHealpixFileHandler(self.analysis_type,
                                        self.use_non_coadded_spectra,
                                        self.uniquify_night_targetid,
                                        self.keep_single_exposures,
                                        self.logger)

        # Read only the files of the healpix pixels owned by this rank. The
        # global file index is kept so that the keep_single_exposures keys stay
        # globally unique.
        forests_by_targetid = {}
        for index, (this_arg, healpix) in enumerate(
                zip(arguments, healpix_of_file)):
            if healpix % self.mpi_size != self.mpi_rank:
                continue
            forests_by_targetid_aux, _ = reader(this_arg)
            if self.use_non_coadded_spectra & self.keep_single_exposures:
                forests_by_targetid_aux = {
                    f"{index}_{key}": items
                    for key, items in forests_by_targetid_aux.items()
                }
            merge_new_forest(forests_by_targetid, forests_by_targetid_aux)

        if self.use_non_coadded_spectra & self.keep_single_exposures:
            forests_by_targetid = verify_exposures_shape(forests_by_targetid)

        self.forests = list(forests_by_targetid.values())

        # A given rank may legitimately own no forest; only raise if the whole
        # run found nothing.
        total_forests = self.comm.allreduce(len(self.forests))
        if total_forests == 0:
            raise DataError("No quasars found, stopping here")
        if self.mpi_rank == 0:
            self.logger.info(f"read {total_forests} forests over all ranks")

        return is_mock

    def find_nside(self):
        """Determine nside such that there are 500 objs per pixel on average.

        The count is global, so the coordinates are gathered on rank 0, nside is
        decided there with the same algorithm as the serial reader, and then
        broadcast. Every rank finally assigns the output healpix of its own
        forests.
        """
        ra = np.array([forest.ra for forest in self.forests])
        dec = np.array([forest.dec for forest in self.forests])

        all_ra = self.comm.gather(ra, root=0)
        all_dec = self.comm.gather(dec, root=0)

        nside = None
        if self.mpi_rank == 0:
            self.logger.progress("determining nside")
            ra_all = np.concatenate(all_ra)
            dec_all = np.concatenate(all_dec)

            nside = 256
            target_mean_num_obj = 500
            nside_min = 8
            healpixs = healpy.ang2pix(nside, np.pi / 2 - dec_all, ra_all)
            mean_num_obj = len(healpixs) / len(np.unique(healpixs))
            while mean_num_obj < target_mean_num_obj and nside >= nside_min:
                nside //= 2
                healpixs = healpy.ang2pix(nside, np.pi / 2 - dec_all, ra_all)
                mean_num_obj = len(healpixs) / len(np.unique(healpixs))
            self.logger.progress(f"nside = {nside} -- mean #obj per pixel = "
                                 f"{mean_num_obj}")

        nside = self.comm.bcast(nside, root=0)

        healpixs = healpy.ang2pix(nside, np.pi / 2 - dec, ra)
        for forest, healpix in zip(self.forests, healpixs):
            forest.healpix = healpix

    def save_deltas(self):
        """Save the deltas.

        The output nside is coarser than the reading nside, so the forests are
        first redistributed by output healpix (owner rank = ``healpix % size``)
        so that every output pixel is written by a single rank. Each rank then
        writes its own delta files and its own (rank-suffixed) rejection log.
        """
        self.forests = redistribute_by_key(self.comm, self.forests,
                                            lambda forest: forest.healpix)

        # A rank owning no output pixel still has to write its rejection log
        # (it may hold entries of forests rejected while it was reading).
        if len(self.forests) == 0:
            self.rejection_log.save_rejection_log()
            return

        super().save_deltas()
