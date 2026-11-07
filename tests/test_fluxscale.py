"""Regression tests for the fixes applied to fluxscale.

These cover the pure/cheap logic that previously crashed or silently
misbehaved: formula parsing, CLI kwarg parsing, combination filtering, reader
de-duplication, conditional sampling, the DWT forward/inverse round-trip, and
the conditional-sampling partition wiring (distinct output suffixes, list/None
inputs). Continuous-wavelet paths are skipped when ``pycwt`` is unavailable.
"""
import os
import sys

import dask
from joblib.externals.loky.process_executor import TerminatedWorkerError
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxscale.core import wavelet_functions as wf
from fluxscale.core import commons
from fluxscale.core import handlers
from fluxscale.extra.partitioning import coimbra_et_al_2025 as co
from fluxscale.io import readers
from fluxscale import main as wmain


# --------------------------------------------------------------------------- #
# formula_to_vars
# --------------------------------------------------------------------------- #
def test_formula_to_vars_parses_components():
    v = wf.formula_to_vars("w*co2|w*h2o")
    assert v.xy == ["w", "co2"]
    assert v.condsamp_pair == [["w", "h2o"]]
    assert v.condsamp_flat == ["w", "h2o"]
    assert v.uniquevars == ["w", "co2", "h2o"]
    assert v.combinations == ["w*co2", "w*h2o"]


def test_formula_to_vars_is_deterministic():
    # set()-based dedup used to make ordering non-reproducible across runs
    f = "w*co2|w*h2o|w*ch4|w*co2"
    first = wf.formula_to_vars(f)
    for _ in range(5):
        again = wf.formula_to_vars(f)
        assert again.uniquevars == first.uniquevars
        assert again.combinations == first.combinations
    # duplicate "w*co2" collapsed, order preserved
    assert first.combinations == ["w*co2", "w*h2o", "w*ch4"]
    assert first.uniquevars == ["w", "co2", "h2o", "ch4"]


# --------------------------------------------------------------------------- #
# __custom_params__
# --------------------------------------------------------------------------- #
def test_custom_params_scalar_flag_and_list():
    assert wmain.__custom_params__(["--alpha", "5"]) == {"alpha": 5}
    assert wmain.__custom_params__(["--beta", "1.5"]) == {"beta": 1.5}
    assert wmain.__custom_params__(["--verbose"]) == {"verbose": True}
    assert wmain.__custom_params__(["--xs", "1", "2", "3"]) == {"xs": [1, 2, 3]}
    assert wmain.__custom_params__(["--name", "abc"]) == {"name": "abc"}


def test_custom_params_value_without_key_raises():
    with pytest.raises(ValueError):
        wmain.__custom_params__(["7"])


# --------------------------------------------------------------------------- #
# available_combinations
# --------------------------------------------------------------------------- #
def test_available_combinations_filters_by_present_variables():
    out = commons.available_combinations(
        ["w*co2", "w*ch4"], variables_available=["u", "v", "w", "ts", "co2", "h2o"])
    assert out == ["w*co2"]


# --------------------------------------------------------------------------- #
# readers de-duplication
# --------------------------------------------------------------------------- #
def test_read_raw_and_read_fluxnet_are_equivalent(tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "TIMESTAMP_START,TIMESTAMP_END,co2\n"
        "202205130000,202205130030,1.0\n"
        "202205130030,202205130100,2.0\n")
    a = readers.read_raw(str(csv))
    b = readers.read_fluxnet(str(csv))
    xr.testing.assert_identical(a, b)
    assert "TIMESTAMP" in a.coords or "TIMESTAMP" in a.dims


# --------------------------------------------------------------------------- #
# _as_series
# --------------------------------------------------------------------------- #
def test_as_series_resolves_none_str_and_list():
    d = xr.Dataset({"a": ("t", [1.0, 2.0]), "b": ("t", [3.0, 4.0])})
    assert co._as_series(d, None) == 0
    np.testing.assert_array_equal(co._as_series(d, "a").values, [1.0, 2.0])
    np.testing.assert_array_equal(co._as_series(d, ["a", "b"]).values, [4.0, 6.0])


# --------------------------------------------------------------------------- #
# conditional_sampling
# --------------------------------------------------------------------------- #
def test_conditional_sampling_keys_and_masking():
    w = xr.DataArray(np.array([1.0, -1.0, 1.0, -1.0]), dims=["t"])
    c = xr.DataArray(np.array([1.0, 1.0, -1.0, -1.0]), dims=["t"])
    Y = w * c
    ds = co.conditional_sampling(Y, c, names=["wc"], label={1: "+", -1: "-"})
    assert set(ds.data_vars) == {"wc+", "wc-"}
    # the two sign-masked components partition Y exactly
    recombined = ds["wc+"] + ds["wc-"]
    np.testing.assert_allclose(recombined.values, Y.values)


# --------------------------------------------------------------------------- #
# conditional_sampling_mean: the means of the classes, without building them
# --------------------------------------------------------------------------- #
def _signed_products():
    """Products on (scale, period, sample), each with NaNs and exact zeros."""
    rng = np.random.default_rng(3)
    shape = (3, 2, 200)
    data = {}
    for name in ("wco2", "wh2o", "wts"):
        x = rng.standard_normal(shape)
        x[rng.random(shape) < 0.05] = 0.0
        x[rng.random(shape) < 0.05] = np.nan
        data[name] = (("natural_frequency", "TIMESTAMP", "ns"), x)
    coords = {"natural_frequency": [1.0, 0.5, 0.25],
              "TIMESTAMP": pd.date_range("2022-05-13", periods=2, freq="30min"),
              "ns": np.arange(shape[-1])}
    return xr.Dataset(data, coords=coords)


def _assert_mean_matches_classes(data, formulas, dim):
    expected = xr.merge(
        [handlers.data_conditional_sampling(data, f) for f in formulas]).mean(dim)
    result = handlers.data_conditional_sampling_mean(data, formulas, dim)
    xr.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
    return result


@pytest.mark.parametrize("formulas", [
    ["w*co2"],
    ["w*co2|w*h2o"],
    ["w*co2|w*h2o|w*ts"],
    ["w*co2|w*h2o", "w*h2o|w*co2", "w*co2|w*ts", "w*co2|w*h2o|w*ts"],
])
def test_conditional_sampling_mean_matches_averaged_classes(formulas):
    _assert_mean_matches_classes(_signed_products(), formulas, "ns")


@pytest.mark.parametrize("dim", [
    ["ns"], ["TIMESTAMP", "ns"], ["natural_frequency", "TIMESTAMP", "ns"]])
def test_conditional_sampling_mean_over_several_dims(dim):
    _assert_mean_matches_classes(_signed_products(), ["w*co2|w*h2o"], dim)


def test_conditional_sampling_mean_keeps_complex_values():
    # the CWT reconstruction is complex
    result = _assert_mean_matches_classes(
        _signed_products() * (1 + 0.5j), ["w*co2|w*h2o"], "ns")
    assert result["wco2+wh2o+"].dtype == complex


def test_conditional_sampling_mean_with_an_all_nan_variable():
    data = _signed_products()
    data["wh2o"] = data["wh2o"] * np.nan
    result = _assert_mean_matches_classes(
        data, ["w*co2|w*h2o", "w*h2o|w*co2"], "ns")
    # wco2 has samples, none in a class; wh2o has no sample to average
    assert (result["wco2+wh2o+"] == 0).all()
    assert result["wh2o+wco2+"].isnull().all()


def _raw_dataset():
    rng = np.random.default_rng(5)
    shape = (2, 1024)
    variables = {v: (("TIMESTAMP", "ns"), rng.standard_normal(shape).cumsum(axis=1))
                 for v in ("w", "co2", "h2o", "ts")}
    ds = xr.Dataset(variables, coords={
        "TIMESTAMP": pd.date_range("2022-05-13", periods=2, freq="30min"),
        "ns": np.arange(shape[-1])})
    ds["co2"].values[0, 100:120] = np.nan
    return ds


@pytest.mark.filterwarnings("ignore:Level value of")
def test_data_run_average_matches_averaging_the_full_result():
    formulas = ["w*co2|w*h2o", "w*co2|w*h2o|w*ts", "w*ts"]
    full = handlers.data_run(_raw_dataset(), formulas, fs=20)
    averaged = handlers.data_run(_raw_dataset(), formulas, fs=20, average=True)
    assert set(full.dims) == {"natural_frequency", "TIMESTAMP", "ns"}
    xr.testing.assert_allclose(averaged, handlers.data_average_dims(full),
                               rtol=1e-12, atol=1e-14)


def test_data_run_average_without_formulas():
    assert "w" in handlers.data_run(_raw_dataset(), [], fs=20, average=True)


def _write_raw_files(folder, *stamps):
    rng = np.random.default_rng(11)
    header = "\n".join(["legend"] * 9 + ["  ".join(["u", "v", "w", "ts", "co2", "h2o"])])
    for stamp in stamps:
        np.savetxt(folder / f"{stamp}_raw_dataset.txt",
                   rng.standard_normal((512, 6)).cumsum(axis=0),
                   header=header, comments="")


@pytest.mark.filterwarnings("ignore:Level value of")
def test_process_saves_the_averaged_classes(tmp_path):
    _write_raw_files(tmp_path, "20220513-0030", "20220513-0100")

    ds = handlers.process("20220513T0000-20220513T0100", 30, str(tmp_path), 20,
                          covariance=["w*co2|w*h2o"], processing_time_duration="1h",
                          output_folderpath=str(tmp_path / "out"))

    assert ds.sizes["TIMESTAMP"] == 2
    assert list((tmp_path / "out" / "wavelet_full_cospectra").glob("*.nc"))
    assert ds.attrs["flux_rule"] == "cross_level"
    # the classes split the flux between them
    classes = [f"csp_wco2{a}wh2o{b}" for a in "+-" for b in "+-"]
    np.testing.assert_allclose(sum(ds[c] for c in classes), ds["csp_wco2"], atol=1e-12)
    # and the levels with the low frequencies add up to the covariance
    np.testing.assert_allclose(
        ds["csp_wco2"].sum("natural_frequency") + ds["w*co2_lf"], ds["cov_wco2_filled"],
        rtol=1e-12, atol=1e-15)


@pytest.mark.filterwarnings("ignore:Level value of")
def test_process_keeps_the_same_scale_products_on_request(tmp_path):
    _write_raw_files(tmp_path, "20220513-0030", "20220513-0100")

    ds = handlers.process("20220513T0000-20220513T0100", 30, str(tmp_path), 20,
                          covariance=["w*co2|w*h2o"], processing_time_duration="1h",
                          output_folderpath=str(tmp_path / "out"), cross_level=False)

    assert ds.attrs["flux_rule"] == "same_scale"
    assert not [v for v in ds.data_vars if v.endswith("_filled")]
    classes = [f"csp_wco2{a}wh2o{b}" for a in "+-" for b in "+-"]
    np.testing.assert_allclose(sum(ds[c] for c in classes), ds["csp_wco2"], atol=1e-12)


TWO_PERIODS = ("20220513T0000-20220513T0200", 30)
TWO_PERIODS_KWARGS = dict(covariance=["w*co2|w*h2o"], processing_time_duration="1h")


def _two_periods(tmp_path):
    _write_raw_files(tmp_path, "20220513-0030", "20220513-0100",
                     "20220513-0130", "20220513-0200")
    return (*TWO_PERIODS, str(tmp_path), 20)


@pytest.mark.filterwarnings("ignore:Level value of", "ignore:index 0 is out of bounds")
def test_process_in_parallel_matches_the_serial_run(tmp_path):
    args = _two_periods(tmp_path)
    folders = {name: tmp_path / name for name in ("serial", "parallel")}

    serial = handlers.process(*args, output_folderpath=str(folders["serial"]),
                              **TWO_PERIODS_KWARGS)
    parallel = handlers.process(*args, output_folderpath=str(folders["parallel"]),
                                n_jobs=2, **TWO_PERIODS_KWARGS)

    assert parallel.sizes["TIMESTAMP"] == 4
    xr.testing.assert_identical(parallel, serial)
    saved = {name: sorted(f.name.rsplit("_", 1)[0] for f in
                          (folder / "wavelet_full_cospectra").iterdir())
             for name, folder in folders.items()}
    assert len(saved["parallel"]) == 2
    assert saved["parallel"] == saved["serial"]


@pytest.mark.filterwarnings("ignore:Level value of", "ignore:index 0 is out of bounds")
def test_process_in_parallel_logs_what_the_workers_log(tmp_path):
    out = tmp_path / "out"
    handlers.process(*_two_periods(tmp_path), output_folderpath=str(out),
                     n_jobs=2, **TWO_PERIODS_KWARGS)
    log = next((out / "log").glob("current_*.log")).read_text()
    assert "20220513003000 reading" in log
    assert "20220513013000 reading" in log


@pytest.mark.filterwarnings("ignore:Level value of", "ignore:index 0 is out of bounds")
def test_process_in_parallel_uses_readers_registered_while_running(tmp_path, monkeypatch):
    reader = readers.READERS.get("ep_raw_lvl")
    monkeypatch.setitem(readers.READERS.available, "local", lambda path: reader(path))
    ds = handlers.process(*_two_periods(tmp_path), reader_method="local",
                          output_folderpath=str(tmp_path / "out"),
                          n_jobs=2, **TWO_PERIODS_KWARGS)
    assert ds.sizes["TIMESTAMP"] == 4


def test_process_leaves_no_marker_when_a_worker_dies(tmp_path, monkeypatch):
    monkeypatch.setitem(readers.READERS.available, "dying", lambda path: os._exit(1))
    out = tmp_path / "out"
    with pytest.raises(TerminatedWorkerError):
        handlers.process(*_two_periods(tmp_path), reader_method="dying",
                         output_folderpath=str(out), n_jobs=2, fixed_grid=False,
                         **TWO_PERIODS_KWARGS)
    assert not list(out.rglob("*.inprogress"))


def test_clear_unfinished_only_touches_the_unfinished_files_of_the_run(tmp_path):
    folder = tmp_path / "wavelet_full_cospectra"
    folder.mkdir()
    kept = ["a_1.nc", "b_2.nc", "b_2.nc.inprogress"]
    for name in kept + ["c_1.nc", "c_1.nc.inprogress", "d_1.nc.inprogress"]:
        (folder / name).touch()
    handlers._clear_unfinished(str(tmp_path), "1")
    assert sorted(f.name for f in folder.iterdir()) == kept


@pytest.mark.filterwarnings("ignore:Level value of", "ignore:index 0 is out of bounds")
def test_process_without_output_folder_returns_the_periods(tmp_path):
    ds = handlers.process(*_two_periods(tmp_path), **TWO_PERIODS_KWARGS)
    assert ds.sizes["TIMESTAMP"] == 4


def test_process_rejects_an_unknown_reader_at_once(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        handlers.process(*_two_periods(tmp_path), reader_method="nope",
                         **TWO_PERIODS_KWARGS)


@pytest.mark.parametrize("entry_point, target", [
    ("run", "process"), ("main", "process"), ("eddypro_run", "run_from_eddypro")])
def test_cli_passes_n_jobs_on(monkeypatch, entry_point, target):
    class Stop(Exception):
        pass

    def stop(**kwargs):
        assert kwargs["n_jobs"] == 3
        raise Stop

    monkeypatch.setattr(wmain, "_init_logging", lambda *args: None)
    monkeypatch.setattr(wmain.handlers, target, stop)
    monkeypatch.setattr(sys, "argv", ["fluxscale", "-j", "3"])
    with pytest.raises(Stop):
        getattr(wmain, entry_point)()


# --------------------------------------------------------------------------- #
# universal_wt
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore:Level value of")
def test_universal_wt_dwt_roundtrip_reconstructs_signal():
    rng = np.random.default_rng(42)
    sig = xr.DataArray(rng.standard_normal(256), dims=["ns"])
    res = wf.universal_wt(sig, method="dwt", wt=True, iwt=True, fs=10, fn=8)
    assert {"signal", "wave", "approximation"}.issubset(res.data_vars)
    recon = res["wave"].sum("natural_frequency") + res["approximation"]
    np.testing.assert_allclose(recon.values, sig.values, atol=1e-9)


def test_universal_wt_rejects_unsupported_flag_combinations():
    sig = xr.DataArray(np.zeros(64), dims=["ns"])
    with pytest.raises(NotImplementedError):
        wf.universal_wt(sig, method="dwt", wt=True, iwt=False)
    with pytest.raises(NotImplementedError):
        wf.universal_wt(sig, method="dwt", wt=False, iwt=True)


@pytest.mark.skipif(wf.pycwt is None, reason="pycwt not installed")
def test_universal_wt_cwt_roundtrip_runs():
    rng = np.random.default_rng(1)
    sig = xr.DataArray(rng.standard_normal(128), dims=["ns"])
    res = wf.universal_wt(sig, method="cwt", wt=True, iwt=True, fs=10, fn=8)
    assert "approximation" in res.data_vars


# --------------------------------------------------------------------------- #
# data_partition: distinct suffixes, list/None inputs don't crash
# --------------------------------------------------------------------------- #
def _covariance_dataset():
    t = pd.date_range("2022-05-13", periods=4, freq="30min")
    rng = np.random.default_rng(7)
    cols = [
        "wco2",
        "wco2-wh2o+", "wco2-wh2o-",
        "wco2+wco+", "wco2+wco-", "wco2-wco+", "wco2-wco-",
        "wco2+wch4+", "wco2+wch4-", "wco2-wch4+", "wco2-wch4-",
    ]
    data = {c: ("TIMESTAMP", rng.standard_normal(4)) for c in cols}
    return xr.Dataset(data, coords={"TIMESTAMP": t})


def test_data_partition_produces_distinct_non_colliding_suffixes():
    ds = _covariance_dataset()
    out = handlers.data_partition(
        ds, variables_available=["w", "co2", "h2o", "co", "ch4"])
    # primary H2O channel stays unsuffixed
    for base in ("NEE", "GPP", "Reco"):
        assert base in out.data_vars
    # other channels are suffixed and therefore do not clobber each other
    for name in ("NEE_pH2O_CO", "NEE_pCO", "NEE_pCH4"):
        assert name in out.data_vars
    # the CO/CH4 branches use a list CO2neg_H2Opos and CO2neg_H2Oneg=None;
    # their presence proves _as_series handled both without raising.
    assert "ffCO2_pCO" in out.data_vars
    assert "ffCO2_pCH4" in out.data_vars


def test_data_partition_skips_channels_with_missing_variables():
    ds = _covariance_dataset()
    out = handlers.data_partition(ds, variables_available=["w", "co2", "h2o"])
    # only the H2O channel should run
    assert "NEE" in out.data_vars
    assert "NEE_pCO" not in out.data_vars
    assert "NEE_pCH4" not in out.data_vars


# --------------------------------------------------------------------------- #
# partition_DWCS_CO night masking / None handling
# --------------------------------------------------------------------------- #
def test_partition_dwcs_co_handles_list_and_none_inputs():
    ds = _covariance_dataset()
    out = co.partition_DWCS_CO(
        ds, CO2="wco2",
        CO2neg_H2Opos=["wco2-wco+", "wco2-wco-"], CO2neg_H2Oneg=None,
        CO2pos_COpos="wco2+wco+", CO2pos_COneg="wco2+wco-", NIGHT=None)
    for base in ("NEE", "GPP", "Reco", "ffCO2"):
        assert base in out.data_vars
    # with NIGHT=None, islight is 1 everywhere -> GPP is not forced to zero
    assert np.isfinite(out["GPP"].values).all()


# --------------------------------------------------------------------------- #
# open_files_in_folder: same result as the former lazy open_mfdataset reader
# --------------------------------------------------------------------------- #
def _write_cospectra_folder(folder, heterogeneous=False):
    """Small per-period files; sorted filename order differs from time order."""
    rng = np.random.default_rng(3)
    starts = {"b_202001010000.nc": "2020-01-01 00:30",   # earliest times, sorts 2nd
              "a_202001010300.nc": "2020-01-01 03:30",
              "c_202001010600.nc": "2020-01-01 06:30"}
    for i, (name, start) in enumerate(starts.items()):
        t = pd.date_range(start, periods=6, freq="30min")
        f = [10.0, 5.0, 2.5] + ([1.25] if heterogeneous and i == 1 else [])
        ds = xr.Dataset(
            {"wco2~": (("natural_frequency", "TIMESTAMP"),
                       rng.standard_normal((len(f), 6))),
             "wco2_lf": ("TIMESTAMP", rng.standard_normal(6))},
            coords={"natural_frequency": f, "TIMESTAMP": t},
            attrs={"method": "dwt", "dt": np.nan, "run": i})
        if heterogeneous:
            ds["weight"] = ("natural_frequency", np.arange(len(f), dtype=float))
            ds = ds.assign_coords(site="ES-LMa", block=i)
            if i == 2:
                ds = ds.drop_vars("wco2_lf")
        ds.to_netcdf(folder / name)


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("load", [None, True, False])
def test_open_files_in_folder_matches_former_reader(tmp_path, load, heterogeneous):
    _write_cospectra_folder(tmp_path, heterogeneous)
    with xr.open_mfdataset(  # former reader, with xarray's then-default merge rules
            str(tmp_path / "*.nc"), combine="nested", concat_dim="TIMESTAMP",
            data_vars="all", coords="different", compat="no_conflicts",
            join="outer") as ref:
        ref = ref.load()
    out = handlers.open_files_in_folder(str(tmp_path), load=load)
    try:
        xr.testing.assert_identical(out, ref)
        assert list(out.variables) == list(ref.variables)
        assert out.TIMESTAMP.encoding["units"] == ref.TIMESTAMP.encoding["units"]
        # concatenated in sorted filename order (a, b, c), not in time order
        assert out.TIMESTAMP.values[0] == np.datetime64("2020-01-01T03:30")
        chunks = out["wco2~"].chunks
        if load is False:
            assert len(chunks[out["wco2~"].dims.index("TIMESTAMP")]) == 1
        else:
            assert chunks is None
    finally:
        out.close()


def test_open_files_in_folder_chunks_loaded_result(tmp_path):
    _write_cospectra_folder(tmp_path)
    out = handlers.open_files_in_folder(str(tmp_path), chunks={"TIMESTAMP": 5})
    assert out["wco2_lf"].chunks == ((5, 5, 5, 3),)


def test_open_files_in_folder_without_files_raises(tmp_path):
    with pytest.raises(OSError, match="no files to open"):
        handlers.open_files_in_folder(str(tmp_path))


def test_open_files_in_folder_opens_large_folders_lazily(
        tmp_path, monkeypatch, caplog):
    _write_cospectra_folder(tmp_path)
    monkeypatch.setattr(handlers, "LOAD_LIMIT", 1)
    with caplog.at_level("WARNING", logger=handlers.logger.name):
        out = handlers.open_files_in_folder(str(tmp_path))
    try:
        assert out["wco2~"].chunks is not None
        assert "opening them lazily" in caplog.text
    finally:
        out.close()


def test_open_files_in_folder_load_overrides_the_limit(tmp_path, monkeypatch):
    _write_cospectra_folder(tmp_path)
    monkeypatch.setattr(handlers, "LOAD_LIMIT", 1)
    out = handlers.open_files_in_folder(str(tmp_path), load=True)
    assert out["wco2~"].chunks is None


def test_open_files_in_folder_lazy_chunks_keep_to_the_dask_limit(tmp_path):
    _write_cospectra_folder(tmp_path)
    with dask.config.set({"array.chunk-size": "128B"}):
        out = handlers.open_files_in_folder(str(tmp_path), load=False)
    try:
        data = out["wco2~"].data
        assert data.npartitions > 1
        assert data.blocks[0, 0].nbytes <= 128
    finally:
        out.close()


def test_open_files_in_folder_lazy_open_keeps_text_variables(tmp_path):
    for i, site in enumerate(["ES-LMa", "FR-Gri"]):
        t = pd.date_range(f"2020-01-0{i + 1}", periods=4, freq="30min")
        xr.Dataset({"x": ("TIMESTAMP", np.arange(4.0)),
                    "site": ("TIMESTAMP", [site] * 4)},
                   coords={"TIMESTAMP": t}).to_netcdf(tmp_path / f"{i}.nc")
    out = handlers.open_files_in_folder(str(tmp_path), load=False)
    try:
        assert out["site"].values[[0, -1]].tolist() == ["ES-LMa", "FR-Gri"]
    finally:
        out.close()


# --------------------------------------------------------------------------- #
# cross-level flux
# --------------------------------------------------------------------------- #
def _gappy_dataset():
    """Windows with an interior gap in co2 and NaNs at the end of each file."""
    rng = np.random.default_rng(7)
    shape = (3, 1024)
    noise = rng.standard_normal((4, *shape))
    variables = {v: (("TIMESTAMP", "ns"), noise[i] + .05 * noise[i].cumsum(axis=1))
                 for i, v in enumerate(("w", "co2", "h2o", "ts"))}
    ds = xr.Dataset(variables, coords={
        "TIMESTAMP": pd.date_range("2022-05-13", periods=shape[0], freq="30min"),
        "ns": np.arange(shape[-1])})
    ds["co2"] = ds["co2"] + .5 * ds["w"]
    ds["co2"].values[1, 300:360] = np.nan
    for v in ds.data_vars:
        ds[v].values[:, -3:] = np.nan
    return ds


@pytest.mark.filterwarnings("ignore:Level value of")
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("formula", ["w*co2", "w*co2*h2o", "w*w"])
@pytest.mark.parametrize("method", ["dwt", "moda"])
def test_the_flux_closes_in_every_window(method, formula, average):
    data = _gappy_dataset()
    out = handlers.data_run(data, [formula], fs=20, method=method, average=average, level=8)
    assert out.attrs["flux_rule"] == "cross_level"
    if not average:
        out = handlers.data_average_dims(out)
    xy = formula.replace("*", "")
    total = out[xy].sum("natural_frequency") + out[f"{formula}_lf"]
    np.testing.assert_allclose(total, out[f"cov_{xy}_filled"], rtol=1e-11, atol=1e-12)
    if formula == "w*co2":
        assert not np.allclose(out["cov_wco2_filled"], out["cov_wco2"], equal_nan=True)


@pytest.mark.filterwarnings("ignore:Level value of")
@pytest.mark.parametrize("method", ["dwt", "moda"])
def test_the_flux_density_adds_up_to_the_product_sample_by_sample(method):
    data = _gappy_dataset()
    decomposed = handlers.decompose_variables(
        data.stack(z=data.dims), ["w", "co2", "h2o"], keep_signal=True, method=method,
        fs=20, level=8).unstack()
    density = handlers.data_compute_flux(decomposed, "w*co2*h2o")
    anomalies = [decomposed[f"{v}_signal"] - decomposed[f"{v}_signal"].mean("ns")
                 for v in ("w", "co2", "h2o")]
    total = density["wco2h2o"].sum("natural_frequency") + density["w*co2*h2o_lf"]
    xr.testing.assert_allclose(total, (anomalies[0] * anomalies[1] * anomalies[2])
                               .transpose(*total.dims), rtol=1e-10, atol=1e-10)
    averaged = handlers.data_compute_flux(decomposed, "w*co2*h2o", average=True)
    xr.testing.assert_allclose(handlers.data_average_dims(density), averaged,
                               rtol=1e-12, atol=1e-13)


def test_the_flux_conjugates_the_factors_after_the_first():
    rng = np.random.default_rng(3)

    def complex_noise(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    dims = ("natural_frequency", "TIMESTAMP", "ns")
    data = xr.Dataset({f"{v}{suffix}": (dims[k:], complex_noise(*(4, 2, 64)[k:]))
                       for v in ("x", "y") for suffix, k in (("", 0), ("_lf", 1))})
    for v in ("x", "y"):
        data[f"{v}_signal"] = data[v].sum("natural_frequency") + data[f"{v}_lf"]
    out = handlers.data_compute_flux(data, "x*y", average=True)
    x, y = (data[f"{v}_signal"] - data[f"{v}_signal"].mean("ns") for v in ("x", "y"))
    expected = (x * y.conj()).mean("ns")
    np.testing.assert_allclose(out["xy"].sum("natural_frequency") + out["x*y_lf"], expected)
    np.testing.assert_allclose(out["cov_xy_filled"], expected)


def test_a_single_variable_has_no_flux():
    assert not handlers.data_compute_flux(xr.Dataset(), "co2").data_vars


@pytest.mark.filterwarnings("ignore:Level value of")
def test_the_classes_take_their_sign_from_the_same_scale_products():
    data = _gappy_dataset()
    formulas = ["w*co2|w*h2o"]
    new = handlers.data_run(data, formulas, fs=20, level=8)
    old = handlers.data_run(data, formulas, fs=20, level=8, cross_level=False)
    classes = [f"wco2{a}wh2o{b}" for a in "+-" for b in "+-"]
    for name in classes:
        # a sample is in the same class, with the flux density as its value
        xr.testing.assert_equal(new[name] != 0, old[name] != 0)
        xr.testing.assert_equal(new[name], new["wco2"].where(old[name] != 0, 0))
    changed = {"wco2", "wco2~", "wh2o", "wh2o~", "w*co2_lf", *classes}
    added = {"w*h2o_lf", "cov_wco2_filled", "cov_wh2o_filled"}
    assert set(new.data_vars) == set(old.data_vars) | added
    for name in set(old.data_vars) - changed:
        xr.testing.assert_identical(new[name], old[name])


@pytest.mark.filterwarnings("ignore:Level value of")
def test_the_classes_of_the_mean_are_the_mean_of_the_classes():
    data = _gappy_dataset()
    formulas = ["w*co2|w*h2o"]
    full = handlers.data_average_dims(handlers.data_run(data, formulas, fs=20, level=8))
    averaged = handlers.data_run(data, formulas, fs=20, level=8, average=True)
    xr.testing.assert_allclose(averaged, full, rtol=1e-12, atol=1e-14)


@pytest.mark.filterwarnings("ignore:Level value of")
def test_without_windows_the_products_stay_same_scale(caplog):
    rng = np.random.default_rng(2)
    data = xr.Dataset(
        {v: ("TIMESTAMP", rng.standard_normal(1024).cumsum()) for v in ("w", "co2")},
        coords={"TIMESTAMP": pd.date_range("2022-05-13", periods=1024, freq="50ms")})
    with caplog.at_level("WARNING"):
        new = handlers.data_run(data, ["w*co2"], fs=20, level=6)
    old = handlers.data_run(data, ["w*co2"], fs=20, level=6, cross_level=False)
    assert "No window dim" in caplog.text
    assert new.attrs["flux_rule"] == "same_scale"
    xr.testing.assert_identical(new, old)


@pytest.mark.filterwarnings("ignore:Level value of")
@pytest.mark.parametrize("method, level", [("modwt", "INFO"), ("cwt", "WARNING")])
def test_other_methods_keep_the_same_scale_products(caplog, method, level):
    data = _gappy_dataset()
    with caplog.at_level(level, logger="fluxscale.core.handlers"):
        new = handlers.data_run(data, ["w*co2|w*h2o"], fs=20, level=8, method=method)
    assert "same-scale" in caplog.text
    assert new.attrs["flux_rule"] == "same_scale"
    old = handlers.data_run(data, ["w*co2|w*h2o"], fs=20, level=8, method=method,
                            cross_level=False)
    xr.testing.assert_identical(new, old)


@pytest.mark.filterwarnings("ignore:Level value of")
def test_the_partition_of_the_flux():
    data = _gappy_dataset()
    out = handlers.data_run(data, ["w*co2|w*h2o"], fs=20, level=8, average=True)
    out = handlers.data_partition(handlers.data_integrate_in_frequency(out, 1 / 60))
    np.testing.assert_array_equal(out["NEE"], out["wco2"])
    np.testing.assert_allclose(out["GPP"] + out["Reco"], out["NEE"])


@pytest.mark.parametrize("load", [True, False])
def test_open_files_in_folder_refuses_files_of_two_flux_rules(tmp_path, load):
    _write_cospectra_folder(tmp_path)
    for path in sorted(tmp_path.glob("*.nc"))[1:]:
        ds = xr.load_dataset(path)
        ds.attrs["flux_rule"] = "cross_level"
        ds.to_netcdf(path)
    with pytest.raises(ValueError, match="1 same_scale"):
        handlers.open_files_in_folder(str(tmp_path), load=load)


@pytest.mark.parametrize("cross_level", [True, False])
def test_process_warns_about_a_folder_of_two_flux_rules(tmp_path, caplog, cross_level):
    folder = tmp_path / "out" / "wavelet_full_cospectra"
    folder.mkdir(parents=True)
    _write_cospectra_folder(folder)
    _write_raw_files(tmp_path, "20220513-0030")
    with caplog.at_level("WARNING"):
        handlers.process("20220513T0000-20220513T0030", 30, str(tmp_path), 20,
                         covariance=["w*co2"], output_folderpath=str(tmp_path / "out"),
                         processing_time_duration="1h", fixed_grid=False,
                         cross_level=cross_level)
    # the old files are same_scale: only a cross-level run mixes the rules
    assert ("cross_level and same_scale flux rules" in caplog.text) == cross_level


@pytest.mark.parametrize("flags, expected", [([], True), (["--no_cross_level"], False)])
@pytest.mark.parametrize("entry_point, target", [
    ("run", "process"), ("main", "process"), ("eddypro_run", "run_from_eddypro")])
def test_cli_passes_cross_level_on(monkeypatch, entry_point, target, flags, expected):
    class Stop(Exception):
        pass

    def stop(**kwargs):
        assert kwargs["cross_level"] is expected
        raise Stop

    monkeypatch.setattr(wmain, "_init_logging", lambda *args: None)
    monkeypatch.setattr(wmain.handlers, target, stop)
    monkeypatch.setattr(sys, "argv", ["fluxscale", *flags])
    with pytest.raises(Stop):
        getattr(wmain, entry_point)()
