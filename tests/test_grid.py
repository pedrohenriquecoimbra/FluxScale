"""Tests for the fixed calendar grid.

The central claim of ``fluxscale.core.grid`` is that a period gets the same
result however the run is cut into blocks and wherever the run starts, down to
the last bit. The window here, 3600 samples, is a multiple of 2**4 only, so that
at the levels used it falls off the grid as it does at 20 Hz.
"""
import datetime
import glob
import os
import shutil
import sys

import numpy as np
import pytest
import xarray as xr

from fluxscale import main as wmain
from fluxscale.core import cascade, grid, handlers

pytestmark = pytest.mark.skipif(cascade.zarr is None, reason="zarr not installed")

FS = 2
MINUTES = 30
WINDOW = FS * MINUTES * 60
LEVEL = 7
WINDOWS = 16
START = datetime.datetime(2022, 5, 13)
RANGE = "20220513T0000-20220513T0800"
COVARIANCE = ["w*co2|w*h2o"]


@pytest.fixture(scope="module")
def raw_files(tmp_path_factory):
    """Eight hours of half-hourly files, named after the end of their window."""
    folder = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((WINDOWS * WINDOW, 6))
    samples = noise + np.cumsum(noise, axis=0) * .02
    samples[:, 4] += samples[:, 2]
    samples[:, 5] -= samples[:, 2]
    samples[9000:9400, 5] = samples[9000, 5]    # a sensor stuck on one value
    header = "\n".join(["legend"] * 9 + ["  ".join(["u", "v", "w", "ts", "co2", "h2o"])])
    for i in range(WINDOWS):
        end = START + datetime.timedelta(minutes=MINUTES * (i + 1))
        np.savetxt(folder / f"{end:%Y%m%d-%H%M}_raw_dataset.txt",
                   samples[i * WINDOW:(i + 1) * WINDOW], header=header, comments="")
    return folder


def run(input_path, out, block="2h", period=RANGE, **kwargs):
    return handlers.process(
        period, MINUTES, str(input_path), FS, covariance=COVARIANCE,
        processing_time_duration=block, output_folderpath=str(out),
        wt_kwargs={"level": LEVEL}, **kwargs)


def data_only(ds):
    """Without the attributes, which name the settings of the run."""
    out = ds.copy(deep=False)
    out.attrs = {}
    return out


@pytest.fixture(scope="module")
def reference(raw_files, tmp_path_factory):
    return run(raw_files, tmp_path_factory.mktemp("reference"))


def test_sample_number_counts_from_the_unix_epoch():
    assert grid.sample_number("1970-01-01T00:00:01", 20) == 20
    assert grid.sample_number("2022-05-14T10:00", 20) == 33_050_448_000


@pytest.mark.parametrize("fs", [20, 10.0, 12.5])
def test_sample_time_undoes_sample_number(fs):
    number = grid.sample_number("2022-05-14T10:00:00.4", fs)
    assert grid.sample_number(grid.sample_time(number, fs), fs) == number


def test_deepest_level_is_the_one_universal_wt_uses():
    assert grid.deepest_level(20) == 18


def test_grid_bounds_hold_the_period_and_its_neighbours():
    first, last = grid.grid_bounds("2022-05-14T10:00", "2022-05-14T10:30", 20, 15)
    margin = cascade.cascade_margin(15)
    assert first % 2 ** 15 == 0 and last % 2 ** 15 == 0
    assert first <= grid.sample_number("2022-05-14T10:00", 20) - margin
    assert last >= grid.sample_number("2022-05-14T10:30", 20) + margin


def test_reconstruct_approximation_completes_the_levels(tmp_path):
    x = np.random.default_rng(1).standard_normal(2 ** 12).cumsum()
    store = cascade.CascadeStore.create(
        str(tmp_path / "s.zarr"), "x", x.size, 20, split_level=4, chunk_size=2 ** 10,
        t0="2020-01-01", details_dtype="float64")
    cascade.cascade_run(store, {"x": x})
    levels = cascade.reconstruct(store, "x", start=100, stop=900).sum("natural_frequency")
    low = cascade.reconstruct_approximation(store, "x", 100, 900)
    np.testing.assert_allclose(levels.values + low, x[100:900], atol=1e-9)


def test_reconstruct_signal_is_the_filled_record_whatever_the_range(tmp_path):
    x = np.random.default_rng(1).standard_normal(2 ** 12).cumsum()
    x[1000:1100] = np.nan
    store = cascade.CascadeStore.create(
        str(tmp_path / "s.zarr"), "x", x.size, 20, split_level=4, chunk_size=2 ** 10,
        t0="2020-01-01", details_dtype="float64")
    cascade.cascade_run(store, {"x": x})
    filled = np.interp(np.arange(x.size), np.flatnonzero(np.isfinite(x)), x[np.isfinite(x)])
    whole = cascade.reconstruct_signal(store, "x", 100, 3000)
    np.testing.assert_allclose(whole, filled[100:3000], atol=1e-9)
    np.testing.assert_array_equal(cascade.reconstruct_signal(store, "x", 1037, 2011),
                                  whole[937:1911])


@pytest.mark.parametrize("block", ["1h", "3h", "8h"])
def test_block_size_does_not_change_the_result(raw_files, reference, tmp_path, block):
    xr.testing.assert_identical(data_only(run(raw_files, tmp_path, block)),
                                data_only(reference))


def test_a_run_of_part_of_the_data_gets_the_same_result(raw_files, reference, tmp_path):
    part = run(raw_files, tmp_path, period="20220513T0300-20220513T0500")
    assert part.sizes["TIMESTAMP"] == 4
    xr.testing.assert_identical(part, reference.sel(TIMESTAMP=part.TIMESTAMP))


def test_parallel_workers_get_the_same_result(raw_files, reference, tmp_path):
    xr.testing.assert_identical(run(raw_files, tmp_path, n_jobs=2), reference)


def test_it_gives_the_variables_of_the_regular_run(raw_files, reference, tmp_path):
    regular = handlers.process(
        RANGE, MINUTES, str(raw_files), FS, covariance=COVARIANCE,
        processing_time_duration="8h", output_folderpath=str(tmp_path),
        wt_kwargs={"level": LEVEL}, fixed_grid=False)
    assert "grid_origin" not in regular.attrs
    assert set(reference.data_vars) == set(regular.data_vars)
    assert reference.attrs["grid_origin"] == grid.ORIGIN
    assert reference.attrs["grid_level"] == LEVEL
    np.testing.assert_array_equal(reference.natural_frequency, regular.natural_frequency)


def test_a_missing_file_stays_a_gap_on_the_grid(raw_files, reference, tmp_path):
    folder = tmp_path / "raw"
    shutil.copytree(raw_files, folder)
    os.remove(glob.glob(str(folder / "20220513-0400*"))[0])
    out = run(folder, tmp_path / "out")
    missing = np.datetime64("2022-05-13T04:00")
    assert out.sizes["TIMESTAMP"] == WINDOWS - 1
    assert missing not in out.TIMESTAMP.values
    # far from the gap nothing changes, and what changes is flagged
    kept = reference.sel(TIMESTAMP=out.TIMESTAMP)
    distance = np.abs(out.TIMESTAMP.values - missing) / np.timedelta64(1, "h")
    far = distance > 1
    xr.testing.assert_identical(data_only(out.isel(TIMESTAMP=far)),
                                data_only(kept.isel(TIMESTAMP=far)))
    changed = np.abs(out["csp_wco2~"] - kept["csp_wco2~"]) > 1e-12
    flagged = out["csp_w_qc~"] > kept["csp_w_qc~"]
    assert changed.any() and not (changed & ~flagged).any()


def test_the_ends_of_the_data_are_flagged(reference):
    deepest = reference["csp_w_qc~"].isel(natural_frequency=-1)
    assert deepest.isel(TIMESTAMP=0) > 0 and deepest.isel(TIMESTAMP=-1) > 0
    assert deepest.isel(TIMESTAMP=WINDOWS // 2) == 0


def test_a_second_run_uses_the_decomposition_unless_overwriting(
        raw_files, reference, tmp_path, monkeypatch):
    run(raw_files, tmp_path)
    runs = []
    real = cascade.cascade_run
    monkeypatch.setattr(cascade, "cascade_run",
                        lambda *a, **k: runs.append(1) or real(*a, **k))
    run(raw_files, tmp_path / "other")
    assert len(runs) == 1
    runs.clear()
    again = run(raw_files, tmp_path, overwrite=False)
    assert not runs and not again.sizes.get("TIMESTAMP")   # periods already done
    run(raw_files, tmp_path, overwrite=True)
    assert len(runs) == 1


def test_the_output_feeds_the_partition(raw_files, tmp_path):
    run(raw_files, tmp_path)
    data = handlers.open_files_in_folder(
        str(tmp_path / "wavelet_full_cospectra")).sortby("TIMESTAMP")
    data = handlers.data_integrate_in_frequency(handlers.data_average_dims(data), 1 / 60)
    out = handlers.data_partition(data)
    assert np.isfinite(out["NEE"]).all() and np.isfinite(out["GPP"]).all()


def test_it_works_without_an_output_folder(raw_files, reference):
    out = handlers.process(
        RANGE, MINUTES, str(raw_files), FS, covariance=COVARIANCE,
        processing_time_duration="2h", wt_kwargs={"level": LEVEL})
    xr.testing.assert_identical(data_only(out), data_only(reference))


def test_other_methods_use_regular_blocks():
    assert handlers._build_grid_store(
        [], "folder", "id", "input", None, MINUTES, {}, {"method": "cwt"}, False, 1) is None


def test_no_file_gives_the_empty_result_of_regular_blocks(tmp_path):
    (tmp_path / "raw").mkdir()
    assert not run(tmp_path / "raw", tmp_path / "out").data_vars


def test_files_of_another_length_use_regular_blocks(raw_files, tmp_path, caplog):
    out = handlers.process(
        RANGE, MINUTES, str(raw_files), FS * 2, covariance=COVARIANCE,
        processing_time_duration="8h", output_folderpath=str(tmp_path),
        wt_kwargs={"level": LEVEL})
    assert "grid_origin" not in out.attrs
    assert f"files hold {WINDOW} samples, not the {WINDOW * 2}" in caplog.text


def test_a_reader_without_windows_is_not_for_the_grid(raw_files):
    def read(path):
        return xr.Dataset({"w": ("TIMESTAMP", [1.0])}, coords={"TIMESTAMP": [START]})

    reason = grid.why_not_on_grid(str(raw_files), read, START,
                                  START + datetime.timedelta(hours=8), FS, MINUTES)
    assert "one window per file" in reason


def test_a_file_that_cannot_be_read_does_not_stop_the_run(raw_files, tmp_path, monkeypatch):
    reader = handlers.READERS.get("ep_raw_lvl")

    def fussy(path):
        if "20220513-0030" in path or "20220513-0430" in path:
            raise OSError("bad file")
        return reader(path)

    monkeypatch.setattr(handlers.READERS, "get", lambda name: fussy)
    out = run(raw_files, tmp_path)
    assert out.attrs["grid_origin"] == grid.ORIGIN
    assert out.sizes["TIMESTAMP"] == WINDOWS - 2


def test_each_file_is_read_once_after_a_look_at_three(raw_files, tmp_path, monkeypatch):
    calls = []
    reader = handlers.READERS.get("ep_raw_lvl")

    def counting(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(handlers.READERS, "get", lambda name: counting)
    run(raw_files, tmp_path)
    assert len(calls) == WINDOWS + 3


def test_the_grid_is_the_default(reference):
    assert reference.attrs["grid_origin"] == grid.ORIGIN


def test_data_run_on_a_store_needs_averaging(raw_files):
    with pytest.raises(ValueError, match="average=True"):
        handlers.data_run(xr.Dataset(), COVARIANCE, store="somewhere")


def test_a_failed_save_leaves_no_file_for_the_next_run_to_skip(
        raw_files, tmp_path, monkeypatch):
    def fail(self, path, *args, **kwargs):
        open(path, "w").close()
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail)
    handlers.process(
        RANGE, MINUTES, str(raw_files), FS, covariance=COVARIANCE,
        processing_time_duration="8h", output_folderpath=str(tmp_path),
        wt_kwargs={"level": LEVEL}, fixed_grid=False)
    assert not glob.glob(str(tmp_path / "wavelet_full_cospectra" / "*.nc"))


@pytest.fixture(scope="module")
def modwt_reference(raw_files, tmp_path_factory):
    return run(raw_files, tmp_path_factory.mktemp("modwt"), method="modwt")


def same_to_rounding(a, b):
    xr.testing.assert_allclose(data_only(a), data_only(b), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("block", ["1h", "3h", "8h"])
def test_modwt_block_size_does_not_change_the_result(
        raw_files, modwt_reference, tmp_path, block):
    same_to_rounding(run(raw_files, tmp_path, block, method="modwt"), modwt_reference)


def test_modwt_run_of_part_of_the_data_gets_the_same_result(
        raw_files, modwt_reference, tmp_path):
    part = run(raw_files, tmp_path, period="20220513T0300-20220513T0500", method="modwt")
    assert part.sizes["TIMESTAMP"] == 4
    same_to_rounding(part, modwt_reference.sel(TIMESTAMP=part.TIMESTAMP))


def test_modwt_parallel_workers_get_the_same_result(raw_files, modwt_reference, tmp_path):
    same_to_rounding(run(raw_files, tmp_path, method="modwt", n_jobs=2), modwt_reference)


def test_modwt_with_neighbours_is_the_transform_of_the_whole_record(
        raw_files, modwt_reference, tmp_path):
    whole = handlers.process(
        RANGE, MINUTES, str(raw_files), FS, covariance=COVARIANCE, method="modwt",
        processing_time_duration="8h", output_folderpath=str(tmp_path),
        wt_kwargs={"level": LEVEL}, fixed_grid=False)
    assert "grid_origin" not in modwt_reference.attrs
    # the regular run wraps around the ends of the record, and has no windows to
    # size rounding against, so a class can differ next to a flat stretch
    keep = slice(2, -2)
    names = [v for v in whole.data_vars
             if v.startswith("csp_") and v.endswith("~") and "_qc" not in v]
    same_to_rounding(whole[names].isel(TIMESTAMP=keep),
                     modwt_reference[names].isel(TIMESTAMP=keep))


def test_modwt_gives_the_variables_of_the_discrete_transform(reference, modwt_reference):
    # less those of the cross-level flux, which modwt coefficients do not give
    flux = {v for v in reference.data_vars if v.endswith("_filled")} | {"w*h2o_lf"}
    assert modwt_reference.attrs["flux_rule"] == "same_scale"
    assert set(modwt_reference.data_vars) == set(reference.data_vars) - flux
    np.testing.assert_array_equal(modwt_reference.natural_frequency, reference.natural_frequency)


def test_modwt_keeps_samples_and_reuses_them(raw_files, tmp_path, monkeypatch):
    run(raw_files, tmp_path, method="modwt")
    stores = glob.glob(str(tmp_path / "decomposition" / "*.samples"))
    assert len(stores) == 1
    monkeypatch.setattr(grid, "_stage", lambda *a, **k: pytest.fail("read the files again"))
    run(raw_files, tmp_path, method="modwt", identifier="00000", overwrite=False)


def test_modwt_missing_file_stays_a_gap(raw_files, modwt_reference, tmp_path):
    folder = tmp_path / "raw"
    shutil.copytree(raw_files, folder)
    os.remove(glob.glob(str(folder / "20220513-0400*"))[0])
    out = run(folder, tmp_path / "out", method="modwt")
    assert out.sizes["TIMESTAMP"] == WINDOWS - 1
    kept = modwt_reference.sel(TIMESTAMP=out.TIMESTAMP)
    changed = np.abs(out["csp_wco2~"] - kept["csp_wco2~"]) > 1e-9
    flagged = out["csp_w_qc~"] > kept["csp_w_qc~"]
    assert changed.any() and not (changed & ~flagged).any()


def test_modwt_ends_of_the_data_are_flagged(modwt_reference):
    deepest = modwt_reference["csp_w_qc~"].isel(natural_frequency=-1)
    assert deepest.isel(TIMESTAMP=0) > 0 and deepest.isel(TIMESTAMP=-1) > 0
    assert deepest.isel(TIMESTAMP=WINDOWS // 2) == 0


def test_modwt_output_feeds_the_partition(raw_files, tmp_path):
    run(raw_files, tmp_path, method="modwt")
    data = handlers.open_files_in_folder(
        str(tmp_path / "wavelet_full_cospectra")).sortby("TIMESTAMP")
    data = handlers.data_integrate_in_frequency(handlers.data_average_dims(data), 1 / 60)
    out = handlers.data_partition(data)
    assert np.isfinite(out["NEE"]).all() and np.isfinite(out["GPP"]).all()


@pytest.mark.parametrize("entry_point", ["run", "main"])
def test_cli_passes_the_method_on(monkeypatch, entry_point):
    class Stop(Exception):
        pass

    def stop(**kwargs):
        assert kwargs["method"] == "modwt"
        raise Stop

    monkeypatch.setattr(wmain, "_init_logging", lambda *args: None)
    monkeypatch.setattr(wmain.handlers, "process", stop)
    monkeypatch.setattr(sys, "argv", ["fluxscale", "--method", "modwt"])
    with pytest.raises(Stop):
        getattr(wmain, entry_point)()


@pytest.mark.parametrize("flags, expected", [([], True), (["--no_fixed_grid"], False)])
@pytest.mark.parametrize("entry_point, target", [
    ("run", "process"), ("main", "process"), ("eddypro_run", "run_from_eddypro")])
def test_cli_passes_fixed_grid_on(monkeypatch, entry_point, target, flags, expected):
    class Stop(Exception):
        pass

    def stop(**kwargs):
        assert kwargs["fixed_grid"] is expected
        raise Stop

    monkeypatch.setattr(wmain, "_init_logging", lambda *args: None)
    monkeypatch.setattr(wmain.handlers, target, stop)
    monkeypatch.setattr(sys, "argv", ["fluxscale", *flags])
    with pytest.raises(Stop):
        getattr(wmain, entry_point)()


def closes(ds, xy="wco2", low="w*co2_lf"):
    np.testing.assert_allclose(ds[f"csp_{xy}"].sum("natural_frequency") + ds[low],
                               ds[f"cov_{xy}_filled"], rtol=1e-11, atol=1e-13)


def test_the_flux_closes_on_the_grid(reference):
    assert reference.attrs["flux_rule"] == "cross_level"
    closes(reference)
    closes(reference, "wh2o", "w*h2o_lf")
    # the files have no gap, so the filled signal is the measured one
    n = WINDOW
    np.testing.assert_allclose(reference["cov_wco2_filled"],
                               reference["cov_wco2"] * (n - 1) / n, rtol=1e-10)
    classes = [f"csp_wco2{a}wh2o{b}" for a in "+-" for b in "+-"]
    np.testing.assert_allclose(sum(reference[c] for c in classes), reference["csp_wco2"],
                               atol=1e-12)


@pytest.fixture(scope="module")
def moda_reference(raw_files, tmp_path_factory):
    return run(raw_files, tmp_path_factory.mktemp("moda"), method="moda")


def test_moda_closes_and_does_not_depend_on_the_block(raw_files, moda_reference, tmp_path):
    assert moda_reference.attrs["flux_rule"] == "cross_level"
    assert moda_reference.attrs["method"].startswith("moda")
    closes(moda_reference)
    same_to_rounding(run(raw_files, tmp_path / "3h", "3h", method="moda"), moda_reference)
    part = run(raw_files, tmp_path / "part", period="20220513T0300-20220513T0500",
               method="moda")
    same_to_rounding(part, moda_reference.sel(TIMESTAMP=part.TIMESTAMP))


def test_moda_on_the_grid_is_the_transform_of_the_whole_record(
        raw_files, moda_reference, tmp_path):
    whole = handlers.process(
        RANGE, MINUTES, str(raw_files), FS, covariance=COVARIANCE, method="moda",
        processing_time_duration="8h", output_folderpath=str(tmp_path),
        wt_kwargs={"level": LEVEL}, fixed_grid=False)
    keep = slice(2, -2)
    names = ["csp_wco2", "csp_wh2o", "w*co2_lf", "cov_wco2_filled"]
    same_to_rounding(whole[names].isel(TIMESTAMP=keep),
                     moda_reference[names].isel(TIMESTAMP=keep))
