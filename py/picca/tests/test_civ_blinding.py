"""Unit tests for absorber-specific propagation of DESI blinding strategies."""

from types import SimpleNamespace
from unittest.mock import Mock

import fitsio
import numpy as np
import pytest

from picca import io
from picca.delta_extraction.astronomical_objects.forest import Forest
from picca.delta_extraction.data_catalogues.desi_data import DesiData


def write_delta_header(tmp_path, blinding, delta_format, z_qso=1.5,
                       lambda_obs=(4000.0, 4001.0), weights=(1.0, 1.0)):
    """Write a small synthetic delta file without extracting spectra.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    blinding : str or None
        Stored strategy, or None to omit the BLINDING keyword.
    delta_format : {'image', 'table'}
        Delta FITS representation.
    z_qso : float, optional
        Quasar redshift. The default observed wavelengths, 4000-4001 A, lie
        redward of LYA in the rest frame for the default, 1.5, and blueward
        for z_qso > 2.29.
    lambda_obs : tuple of float, optional
        Observed wavelengths of the two pixels, in Angstrom.
    weights : tuple of float, optional
        Pixel weights. Image-format readers treat non-positive or NaN
        weights as pixels outside the forest.

    Returns
    -------
    filename : pathlib.Path
        Path to the synthetic FITS file.
    """
    filename = tmp_path / "delta-0.fits"
    header = {} if blinding is None else {"BLINDING": blinding}
    delta_name = "DELTA" if blinding in (None, "none") else "DELTA_BLIND"
    lambda_obs = np.array(lambda_obs)
    delta_array = np.array([0.1, -0.2])
    weights = np.array(weights)
    continuum = np.ones_like(delta_array)

    with fitsio.FITS(str(filename), "rw", clobber=True) as hdul:
        if delta_format == "image":
            hdul.write(lambda_obs, extname="LAMBDA")
            hdul.write([np.array([1]), np.array([0.1]), np.array([0.2]),
                        np.array([z_qso])],
                       names=["LOS_ID", "RA", "DEC", "Z"],
                       extname="METADATA", header=header)
            hdul.write(delta_array[None, :], extname=delta_name)
            hdul.write(weights[None, :], extname="WEIGHT")
            hdul.write(continuum[None, :], extname="CONT")
        else:
            header.update({"LOS_ID": 1, "RA": 0.1, "DEC": 0.2, "Z": z_qso})
            hdul.write([lambda_obs, delta_array, weights, continuum],
                       names=["LAMBDA", delta_name, "WEIGHT", "CONT"],
                       header=header)

    return filename


def region_z_qso(lambda_abs):
    """Return a quasar redshift whose rest-frame window matches the absorber.

    Parameters
    ----------
    lambda_abs : str or None
        Primary absorber identifier.

    Returns
    -------
    z_qso : float
        2.5 (Lya region, 1143 A for the default fixture wavelengths) for
        LYA, and 1.5 (CIV region, 1600 A) otherwise.
    """
    return 2.5 if lambda_abs == "LYA" else 1.5


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
    filename = write_delta_header(tmp_path, "desi_dr3", delta_format,
                                  z_qso=region_z_qso(lambda_abs))
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
    for lambda_abs, lambda_abs2 in (
            (None, None), ("CIV(eff)", None),
            ("CIV(1548)", "CIV(1551)"), ("LYA", None),
            ("LYB", None), ("LYA", "CIV(eff)")):
        filename = write_delta_header(tmp_path, stored_strategy,
                                      delta_format,
                                      z_qso=region_z_qso(lambda_abs))
        assert io.read_blinding(str(filename), lambda_abs=lambda_abs,
                               lambda_abs2=lambda_abs2) == stored_strategy

    filename = write_delta_header(tmp_path, stored_strategy, delta_format)
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


@pytest.mark.parametrize("delta_format", ["image", "table"])
def test_read_blinding_requires_lya_for_lya_region(tmp_path, delta_format):
    """Reject non-LYA primary absorbers for deltas blueward of LYA.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.

    Returns
    -------
    None
        Assert that Lya-region deltas (rest frame 1143 A at z_qso = 2.5)
        accept only LYA or an omitted absorber.
    """
    filename = write_delta_header(tmp_path, "desi_dr3", delta_format,
                                  z_qso=2.5)

    assert io.read_blinding(str(filename)) == "desi_dr3"
    assert io.read_blinding(str(filename), lambda_abs="LYA") == "desi_dr3"
    assert io.read_blinding(str(filename), lambda_abs="LYA",
                            lambda_abs2="SiII(1260)") == "desi_dr3"

    for lambda_abs in ("CIV(eff)", "LYB", "SiII(1260)"):
        with pytest.raises(ValueError, match="blueward of the LYA line"):
            io.read_blinding(str(filename), lambda_abs=lambda_abs)


@pytest.mark.parametrize("delta_format", ["image", "table"])
def test_read_blinding_requires_lya_for_window_straddling_lya(
        tmp_path, delta_format):
    """Reject non-LYA absorbers when the forest window straddles LYA.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.

    Returns
    -------
    None
        Assert that a rest-frame window of 1143-1286 A (z_qso = 2.5) requires
        LYA because its blue edge contains Lya-forest pixels.
    """
    filename = write_delta_header(tmp_path, "desi_dr3", delta_format,
                                  z_qso=2.5, lambda_obs=(4000.0, 4500.0))

    with pytest.raises(ValueError, match="blueward of the LYA line"):
        io.read_blinding(str(filename), lambda_abs="CIV(eff)")


@pytest.mark.parametrize("delta_format", ["image", "table"])
@pytest.mark.parametrize("lambda_abs, lambda_abs2, raises", [
    ("CIV(eff)", None, True),
    ("CIV(eff)", "CIV(1548)", True),
    ("SiII(1260)", "SiIII(1207)", True),
    ("CIV(eff)", "LYA", False),
])
def test_read_blinding_requires_lya_for_lya_region_in_dir2(
        tmp_path, delta_format, lambda_abs, lambda_abs2, raises):
    """Check the second field against its own absorber.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.
    lambda_abs, lambda_abs2 : str or None
        Absorbers of the first and second fields; the second field uses
        ``lambda_abs`` when ``lambda_abs2`` is None.
    raises : bool
        Whether the second-field absorber is rejected.

    Returns
    -------
    None
        Assert that CIV-region deltas (1600 A, z_qso = 1.5) in ``in_dir``
        combined with Lya-region deltas (1143 A, z_qso = 2.5) in ``in_dir2``
        require LYA for the second field only.
    """
    in_dir = tmp_path / "civ_region"
    in_dir2 = tmp_path / "lya_region"
    in_dir.mkdir()
    in_dir2.mkdir()
    write_delta_header(in_dir, "desi_dr3", delta_format)
    write_delta_header(in_dir2, "desi_dr3", delta_format, z_qso=2.5)

    if raises:
        with pytest.raises(ValueError, match="lya_region"):
            io.read_blinding(str(in_dir), lambda_abs=lambda_abs,
                             lambda_abs2=lambda_abs2, in_dir2=str(in_dir2))
    else:
        io.read_blinding(str(in_dir), lambda_abs=lambda_abs,
                         lambda_abs2=lambda_abs2, in_dir2=str(in_dir2))

    # Without in_dir2 the second field is read from in_dir: no check applies
    io.read_blinding(str(in_dir), lambda_abs=lambda_abs,
                     lambda_abs2=lambda_abs2)


@pytest.mark.parametrize("delta_format", ["image", "table"])
def test_read_blinding_rejects_lya_for_redward_region(tmp_path, delta_format):
    """Reject LYA for deltas entirely redward of the LYA line.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.

    Returns
    -------
    None
        Assert that CIV-region deltas (1600 A, z_qso = 1.5) reject LYA in
        either field, including through the ``lambda_abs`` default of the
        second field, while Lya-region deltas (1143 A, z_qso = 2.5) accept it.
    """
    civ_dir = tmp_path / "civ_region"
    lya_dir = tmp_path / "lya_region"
    civ_dir.mkdir()
    lya_dir.mkdir()
    write_delta_header(civ_dir, "desi_dr3", delta_format)
    write_delta_header(lya_dir, "desi_dr3", delta_format, z_qso=2.5)

    with pytest.raises(ValueError, match="redward of the LYA line"):
        io.read_blinding(str(civ_dir), lambda_abs="LYA")
    with pytest.raises(ValueError, match="redward of the LYA line"):
        io.read_blinding(str(lya_dir), lambda_abs="LYA",
                         in_dir2=str(civ_dir))

    assert io.read_blinding(str(lya_dir), lambda_abs="LYA",
                            lambda_abs2="CIV(eff)",
                            in_dir2=str(civ_dir)) == "desi_dr3"
    assert io.read_blinding(str(civ_dir), lambda_abs="CIV(eff)",
                            lambda_abs2="LYA",
                            in_dir2=str(lya_dir)) == "desi_dr3"


@pytest.mark.parametrize("delta_format", ["image", "table"])
@pytest.mark.parametrize("stored_strategy", [
    "none", "desi_m2", "desi_y1", "desi_y3", "desi_dr3_civ"
])
def test_read_blinding_checks_region_only_for_current_blinding(
        tmp_path, delta_format, stored_strategy):
    """Skip the absorber-region check for strategies other than DR3.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    delta_format : str
        Synthetic delta representation.
    stored_strategy : str
        Stored strategy differing from ``io.CURRENT_DESI_BLINDING``.

    Returns
    -------
    None
        Assert that absorbers inconsistent with the rest-frame region, which
        raise for the current DESI strategy, are accepted otherwise.
    """
    civ_dir = tmp_path / "civ_region"
    lya_dir = tmp_path / "lya_region"
    civ_dir.mkdir()
    lya_dir.mkdir()
    write_delta_header(civ_dir, stored_strategy, delta_format)
    write_delta_header(lya_dir, stored_strategy, delta_format, z_qso=2.5)

    assert stored_strategy != io.CURRENT_DESI_BLINDING
    io.read_blinding(str(civ_dir), lambda_abs="LYA")
    io.read_blinding(str(lya_dir), lambda_abs="SiII(1260)")
    io.read_blinding(str(civ_dir), lambda_abs="CIV(eff)",
                     in_dir2=str(lya_dir))


def test_read_blinding_ignores_image_pixels_outside_forest(tmp_path):
    """Exclude zero- or NaN-weight image pixels from the rest-frame window.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    None
        Assert that a NaN-weight pixel at 1200 A rest frame (z_qso = 1.5)
        does not trigger the Lya-region requirement for CIV-region deltas.
    """
    filename = write_delta_header(tmp_path, "desi_dr3", "image",
                                  lambda_obs=(3000.0, 4000.0),
                                  weights=(np.nan, 1.0))

    assert io.read_blinding(str(filename),
                            lambda_abs="CIV(eff)") == "desi_dr3_civ"


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
