"""Unit tests for absorber-specific propagation of DESI blinding strategies."""

from types import SimpleNamespace
from unittest.mock import Mock

import fitsio
import numpy as np
import pytest

from picca import io
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data_catalogues.desi_data import DesiData


def write_delta_header(tmp_path, blinding, delta_format):
    """Write a small synthetic delta file without extracting spectra.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    blinding : str or None
        Stored strategy, or None to omit the BLINDING keyword.
    delta_format : {'image', 'table'}
        Delta FITS representation.

    Returns
    -------
    filename : pathlib.Path
        Path to the synthetic FITS file.
    """
    filename = tmp_path / "delta.fits"
    header = {} if blinding is None else {"BLINDING": blinding}
    delta_name = "DELTA" if blinding in (None, "none") else "DELTA_BLIND"
    delta_array = np.array([0.1, -0.2])

    with fitsio.FITS(str(filename), "rw", clobber=True) as hdul:
        if delta_format == "image":
            hdul.write(np.array([4000.0, 4001.0]), extname="LAMBDA")
            hdul.write([np.array([1])], names=["TARGETID"],
                       extname="METADATA", header=header)
            hdul.write(delta_array[None, :], extname=delta_name)
        else:
            hdul.write([delta_array], names=[delta_name], header=header)

    return filename


@pytest.mark.parametrize("delta_format", ["image", "table"])
@pytest.mark.parametrize("lambda_abs, lambda_abs2, expected", [
    (None, None, "desi_dr3"),
    ("CIV(eff)", None, "desi_dr3_civ"),
    ("CIV(1548)", None, "desi_dr3_civ"),
    ("CIV(1551)", None, "desi_dr3_civ"),
    (None, "CIV(eff)", "desi_dr3_civ"),
    ("CIV(eff)", "CIV(eff)", "desi_dr3_civ"),
    ("CIV(eff)", "CIV(1548)", "desi_dr3_civ"),
    ("CIV(eff)", "CIV(1551)", "desi_dr3_civ"),
    ("CIV(1548)", "CIV(eff)", "desi_dr3_civ"),
    ("CIV(1548)", "CIV(1548)", "desi_dr3_civ"),
    ("CIV(1548)", "CIV(1551)", "desi_dr3_civ"),
    ("CIV(1551)", "CIV(eff)", "desi_dr3_civ"),
    ("CIV(1551)", "CIV(1548)", "desi_dr3_civ"),
    ("CIV(1551)", "CIV(1551)", "desi_dr3_civ"),
    ("LYA", None, "desi_dr3"),
    ("LYB", None, "desi_dr3"),
    ("LYA", "LYB", "desi_dr3"),
    ("CIV(eff)", "LYA", "desi_dr3"),
    ("LYB", "CIV(1548)", "desi_dr3"),
    ("CIV(eff)", "MgII(2796)", "desi_dr3"),
    ("MgII(2796)", "CIV(1551)", "desi_dr3"),
    ("LYA", "MgII(2796)", "desi_dr3"),
    ("MgII(2796)", "LYB", "desi_dr3"),
    ("MgII(2796)", None, "none"),
    (None, "MgII(2804)", "none"),
    ("MgII(2796)", "MgII(2804)", "none"),
])
def test_read_blinding_selects_absorber_strategy(tmp_path, delta_format,
                                                lambda_abs, lambda_abs2,
                                                expected):
    """Select the measurement strategy without modifying input delta data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.
    lambda_abs, lambda_abs2 : str or None
        Primary measurement absorber identifiers.
    expected : str
        Expected propagated strategy.

    Returns
    -------
    None
        Assert strategy selection and byte-for-byte preservation of the file.
    """
    filename = write_delta_header(tmp_path, "desi_dr3", delta_format)
    original_delta = filename.read_bytes()

    assert io.read_blinding(str(tmp_path), lambda_abs=lambda_abs,
                           lambda_abs2=lambda_abs2) == expected
    assert filename.read_bytes() == original_delta


@pytest.mark.parametrize("delta_format", ["image", "table"])
@pytest.mark.parametrize("stored_strategy", [
    "none", "desi_m2", "desi_y1", "desi_y3", "desi_dr3_civ"
])
def test_read_blinding_preserves_other_strategies(tmp_path, delta_format,
                                                 stored_strategy):
    """Keep existing strategies for standard/CIV absorbers and omitted inputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.
    stored_strategy : str
        Existing strategy to preserve, except for other-absorber measurements.

    Returns
    -------
    None
        Assert retained strategies and the other-absorber exemption.
    """
    filename = write_delta_header(tmp_path, stored_strategy, delta_format)

    for lambda_abs, lambda_abs2 in (
            (None, None), ("CIV(eff)", None),
            ("CIV(1548)", "CIV(1551)"), ("LYA", None),
            ("LYB", None), ("LYA", "CIV(eff)")):
        assert io.read_blinding(str(filename), lambda_abs=lambda_abs,
                               lambda_abs2=lambda_abs2) == stored_strategy

    assert io.read_blinding(str(filename), lambda_abs="MgII(2796)",
                           lambda_abs2="MgII(2804)") == "none"


def test_read_blinding_table_without_flag(tmp_path):
    """Preserve the legacy unblinded default for table deltas without a flag.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    None
        Assert that neither omitted nor CIV absorbers introduce blinding.
    """
    filename = write_delta_header(tmp_path, None, "table")

    assert io.read_blinding(str(filename)) == "none"
    assert io.read_blinding(str(filename), lambda_abs="CIV(eff)") == "none"


@pytest.mark.parametrize("is_mock, last_nights, expected", [
    (True, [20250101], "none"),
    (False, [20210513], "none"),
    (False, [20210514], "desi_m2"),
    (False, [20210731], "desi_m2"),
    (False, [20210801], "desi_y1"),
    (False, [20220731], "desi_y1"),
    (False, [20220801], "desi_y3"),
    (False, [20240409], "desi_y3"),
    (False, [20240410], "desi_dr3"),
    (False, [20210101, 20250101], "desi_dr3"),
])
def test_set_blinding_uses_mock_status_and_dates(monkeypatch, is_mock,
                                                last_nights, expected):
    """Select delta strategies without reading spectra or a wavelength grid.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restore shared Forest attributes after the test.
    is_mock : bool
        Whether the synthetic catalogue represents mocks.
    last_nights : list of int
        Catalogue observing dates in YYYYMMDD format.
    expected : str
        Strategy expected from the latest observing date and mock status.

    Returns
    -------
    None
        Assert the data and shared Forest strategies.
    """
    monkeypatch.setattr(Forest, "blinding", "none")
    monkeypatch.setattr(Forest, "log_lambda_rest_frame_grid", None)
    data = SimpleNamespace(
        catalogue={"LASTNIGHT": np.asarray(last_nights)},
        logger=Mock())

    DesiData.set_blinding(data, is_mock)

    assert data.blinding == expected
    assert Forest.blinding == expected
