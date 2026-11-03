"""This module defines the class DesiHealpixFastMpi, the MPI-parallel version of
the DesiHealpixFast reader. The healpix decomposition and the communication
points are those of DesiHealpixMpi (see MpiDataMixin in desi_healpix_mpi.py).
"""
from picca.delta_extraction.data_catalogues.desi_healpix_fast import (  # pylint: disable=unused-import
    DesiHealpixFast,
    DesiHealpixFileHandler,
    accepted_options,
    combine_results,
    defaults,
)
from picca.delta_extraction.data_catalogues.desi_healpix_mpi import MpiDataMixin


class DesiHealpixFastMpi(MpiDataMixin, DesiHealpixFast):
    """Reads the spectra from DESI using the fast healpix mode, decomposing the
    healpix pixels over MPI ranks.

    Each rank reads its own files serially, so set "num processors = 1".

    Methods
    -------
    (see DesiHealpixFast in py/picca/delta_extraction/data_catalogues/desi_healpix_fast.py)
    (see MpiDataMixin in py/picca/delta_extraction/data_catalogues/desi_healpix_mpi.py)
    read_data

    Attributes
    ----------
    (see DesiHealpixFast in py/picca/delta_extraction/data_catalogues/desi_healpix_fast.py)
    (see MpiDataMixin in py/picca/delta_extraction/data_catalogues/desi_healpix_mpi.py)
    """

    def read_data(self, is_mock=False):
        """Read the data owned by this rank.

        Every rank reads only the files whose healpix it owns
        (``healpix % size == rank``). Forests are not gathered: each rank keeps
        and processes its own subset.

        Return
        ------
        is_mock: bool
        False for DESI data and True for mocks

        Raise
        -----
        DataError if no quasars were found on any rank
        """
        coadd_name = "spectra" if self.use_non_coadded_spectra else "coadd"

        grouped_catalogue = self.catalogue.group_by(["HEALPIX", "SURVEY"])

        arguments = [
            (self.get_filename(group["SURVEY"][0], group["HEALPIX"][0], coadd_name), group)
            for group in grouped_catalogue.groups
            if self.owns_healpix(group["HEALPIX"][0])
        ]

        if self.mpi_rank == 0:
            self.logger.info(
                f"reading data from {len(grouped_catalogue.groups)} files "
                f"decomposed by healpix over {self.mpi_size} MPI ranks")
        self.logger.debug(f"rank {self.mpi_rank} owns {len(arguments)} files")

        reader = DesiHealpixFileHandler(self.logger)
        self.forests = combine_results(reader(this_arg) for this_arg in arguments)

        self.check_total_forests()

        return is_mock
