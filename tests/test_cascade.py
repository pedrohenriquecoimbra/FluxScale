"""Tests for the deconstructed (chunked) DWT cascade.

The central claim of ``fluxscale.core.cascade`` is exactness: decomposing a
long record chunk by chunk (with overlap margins and periodized boundary
handling), then continuing from the concatenated approximation, must
reproduce -- to machine precision -- the coefficients ``pywt.wavedec`` would
return on the whole record at once. These tests assert that claim for the
pure chunk mathematics and for the full store round-trip, plus the
supporting behaviour: resumability, gap QC, coefficient-space covariance,
and on-demand reconstruction. Store-backed tests are skipped when ``zarr``
is unavailable.
"""
import numpy as np
import pandas as pd
import pytest
import pywt
import xarray as xr

from fluxscale.core import cascade

WAVELET = 'db6'
MODE = 'periodization'

needs_zarr = pytest.mark.skipif(cascade.zarr is None, reason="zarr not installed")


def _signal(n, seed=42):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    return x - x.mean()


def _full_wavedec(x, level):
    return pywt.wavedec(x, WAVELET, level=level, mode=MODE)


# --------------------------------------------------------------------------- #
# chunk_wavedec (pure math, no storage)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nchunks", [2, 4])
def test_chunk_wavedec_matches_full_wavedec(nchunks):
    n, split, level = 2 ** 13, 4, 8
    x = _signal(n)
    full = _full_wavedec(x, level)

    chunk = n // nchunks
    margin = cascade.cascade_margin(split, WAVELET)
    det_parts = {j: [] for j in range(1, split + 1)}
    carry_parts = []
    for s in range(0, n, chunk):
        padded = x[np.arange(s - margin, s + chunk + margin) % n]
        details, carry = cascade.chunk_wavedec(padded, margin, split, WAVELET)
        carry_parts.append(carry)
        for j in range(1, split + 1):
            det_parts[j].append(details[j])

    # stage-1 details equal the full transform's shallow levels
    for j in range(1, split + 1):
        got = np.concatenate(det_parts[j])
        np.testing.assert_allclose(got, full[level - j + 1], atol=1e-10)

    # continuing from the assembled carry equals the deep levels
    deep = pywt.wavedec(np.concatenate(carry_parts), WAVELET,
                        level=level - split, mode=MODE)
    for i, ref in enumerate(full[: level - split + 1]):
        np.testing.assert_allclose(deep[i], ref, atol=1e-10)


def test_chunk_wavedec_rejects_misaligned_input():
    split = 4
    margin = cascade.cascade_margin(split, WAVELET)
    with pytest.raises(ValueError):  # core not a multiple of 2**split
        cascade.chunk_wavedec(np.zeros(2 * margin + 100), margin, split, WAVELET)
    with pytest.raises(ValueError):  # margin not a multiple of 2**split
        cascade.chunk_wavedec(np.zeros(2 * 7 + 64), 7, split, WAVELET)


# --------------------------------------------------------------------------- #
# store round-trip: run + finish == pywt.wavedec of the whole record
# --------------------------------------------------------------------------- #
@needs_zarr
def test_cascade_store_end_to_end_is_exact(tmp_path):
    n, split, level, fs = 2 ** 13, 4, 8, 20.0
    w, co2 = _signal(n, 1), _signal(n, 2)

    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w', 'co2'], n_samples=n, fs=fs,
        wavelet=WAVELET, split_level=split, chunk_size=2 ** 11,
        t0='2022-05-13T00:00:00', details_dtype='float64')
    cascade.cascade_run(store, {'w': w, 'co2': co2})
    cascade.cascade_finish(store, level=level)

    for var, x in [('w', w), ('co2', co2)]:
        full = _full_wavedec(x, level)
        np.testing.assert_allclose(
            store[var]['cA'][:], full[0], atol=1e-10)
        for j in range(1, level + 1):
            np.testing.assert_allclose(
                store[var][f'cD{j:02d}'][:], full[level - j + 1], atol=1e-10,
                err_msg=f"{var} cD{j}")

    # open_level: sizes, frequency metadata, timestamps
    da = cascade.open_level(store, 'w', 3)
    assert da.sizes['TIMESTAMP'] == n >> 3
    assert da.attrs['natural_frequency'] == pytest.approx(fs * 2.0 ** -3)
    first = np.datetime64('2022-05-13T00:00:00', 'ns') \
        + np.timedelta64(int(0.5 * 2 ** 3 / fs * 1e9), 'ns')
    assert da.TIMESTAMP.values[0] == first
    # approximation
    assert cascade.open_level(store, 'w').sizes['TIMESTAMP'] == n >> level


@needs_zarr
def test_cascade_run_is_resumable_and_idempotent(tmp_path):
    n, split = 2 ** 12, 3
    w, co2 = _signal(n, 3), _signal(n, 4)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w', 'co2'], n_samples=n, fs=10.0,
        split_level=split, chunk_size=2 ** 10, details_dtype='float64')

    # partial run (one variable), then complete run, then a redundant rerun
    cascade.cascade_run(store, {'w': w})
    cascade.cascade_run(store, {'w': w, 'co2': co2})
    cascade.cascade_run(store, {'w': w, 'co2': co2})

    for var, x in [('w', w), ('co2', co2)]:
        assert store[var].attrs['n_done'] == n
        full = _full_wavedec(x, split)
        np.testing.assert_allclose(store[var]['carry'][:], full[0], atol=1e-10)
        for j in range(1, split + 1):
            assert store[var][f'cD{j:02d}'].shape == (n >> j,)

    with pytest.raises(ValueError):
        cascade.cascade_run(store, {'not_a_var': w})


@needs_zarr
def test_gap_handling_flags_qc(tmp_path):
    n, split = 2 ** 12, 3
    reach = pywt.Wavelet(WAVELET).dec_len - 1     # qc dilation radius
    x = _signal(n, 5)
    x[512:1024] = np.nan  # a gap spanning dyadic blocks
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=10.0,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {'w': x})
    cascade.cascade_finish(store, level=6)

    assert np.isfinite(store['w']['cD01'][:]).all()
    # level 1: gap covers blocks [256, 512); qc is dilated by the filter
    # reach so gap-influenced neighbours are flagged too, none missed
    qc1 = store['w']['qc01'][:]
    assert qc1[256:512].min() == 1.0                  # fully inside the gap
    assert qc1[256 - reach: 256].min() > 0.0          # influenced neighbours
    assert qc1[: 256 - reach].max() == 0.0            # out of reach: pristine
    assert qc1[512 + reach:].max() == 0.0
    # deep-level QC carries the same fraction: level 6 blocks of 64 samples
    qc6 = store['w']['qc06'][:]
    np.testing.assert_allclose(qc6[8:16], 1.0)
    assert qc6[30:58].max() == 0.0            # out of reach on both sides
    assert qc6[-1] > 0.0                      # the reach wraps round the end

    # windowed products expose the same information: qc row of level 1 for
    # the window containing the gap is high, later windows are clean
    ds = cascade.scale_covariance(store, 'w', 'w', window='102.4s')  # 1024 smp
    row = ds['ww_qc'].sel(natural_frequency=10.0 * 2.0 ** -1).values
    assert row[0] > 0.4                       # window holding the gap itself
    assert 0.0 < row[1] < 0.1                 # touched only through dilation
    assert row[2:].max() == 0.0
    assert np.isfinite(ds['ww'].values).all()


# --------------------------------------------------------------------------- #
# scale_covariance
# --------------------------------------------------------------------------- #
def _brute_force_window_mean(p, level_step, w, nwin):
    """Reference: spread each coefficient uniformly over its samples, then
    average per window."""
    density = np.repeat(np.asarray(p, dtype='float64'), level_step) / level_step
    density = density[: nwin * w]
    return density.reshape(nwin, w).sum(axis=1) / w


@needs_zarr
def test_scale_covariance_partitions_total_covariance(tmp_path):
    n, split, level, fs = 2 ** 12, 4, 8, 4.0
    w_, c_ = _signal(n, 6), _signal(n, 7)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w', 'co2'], n_samples=n, fs=fs,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {'w': w_, 'co2': c_})
    cascade.cascade_finish(store, level=level)

    window = 64 / fs                     # 64 samples -> all-reshape path
    ds = cascade.scale_covariance(store, 'w', 'co2', window=window)
    total = (ds['wco2'].sum('natural_frequency') + ds['wco2_lf']).values
    # global Parseval/partition: window-summed covariance == mean(x*y)
    np.testing.assert_allclose(total.mean(), np.mean(w_ * c_), rtol=1e-9)

    # a window that 2**j does not divide for deep j -> fractional path;
    # both paths must agree with the per-sample density reference
    window = 48 / fs
    ds = cascade.scale_covariance(store, 'w', 'co2', window=window)
    nwin = n // 48
    for j in range(1, level + 1):
        p = store['w'][f'cD{j:02d}'][:] * store['co2'][f'cD{j:02d}'][:]
        ref = _brute_force_window_mean(p, 2 ** j, 48, nwin)
        np.testing.assert_allclose(
            ds['wco2'].sel(natural_frequency=fs * 2.0 ** -j).values, ref,
            atol=1e-12, err_msg=f"level {j}")


# --------------------------------------------------------------------------- #
# reconstruct
# --------------------------------------------------------------------------- #
@needs_zarr
def test_reconstruct_matches_full_record_reconstruction(tmp_path):
    n, split = 2 ** 13, 4
    x = _signal(n, 8)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=20.0,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {'w': x})

    start, stop = 1000, 3000
    got = cascade.reconstruct(store, 'w', levels=[1, 2, 4],
                              start=start, stop=stop)
    assert got.shape == (3, stop - start)

    full = _full_wavedec(x, split)
    for row, j in enumerate([1, 2, 4]):
        coeffs = [np.zeros_like(c) for c in full]
        coeffs[split - j + 1] = full[split - j + 1]
        ref = pywt.waverec(coeffs, WAVELET, mode=MODE)[start:stop]
        np.testing.assert_allclose(got.values[row], ref, atol=1e-10,
                                   err_msg=f"level {j}")

    with pytest.raises(ValueError):  # deeper than split_level
        cascade.reconstruct(store, 'w', levels=[split + 1], start=0, stop=64)


# --------------------------------------------------------------------------- #
# store validation
# --------------------------------------------------------------------------- #
@needs_zarr
def test_create_validates_alignment(tmp_path):
    with pytest.raises(ValueError):  # n_samples not a multiple of 2**split
        cascade.CascadeStore.create(
            tmp_path / "a.zarr", ['w'], n_samples=1000, fs=10.0, split_level=4)
    with pytest.raises(ValueError):  # chunk not a multiple of 2**split
        cascade.CascadeStore.create(
            tmp_path / "b.zarr", ['w'], n_samples=1024, fs=10.0,
            split_level=4, chunk_size=100)
    # finishing before stage 1 is complete
    with pytest.raises(ValueError, match="run cascade_run"):
        store = cascade.CascadeStore.create(
            tmp_path / "c.zarr", ['w'], n_samples=1024, fs=10.0, split_level=4)
        cascade.cascade_finish(store)
    with pytest.raises(ValueError):  # margin below the exactness bound
        cascade.CascadeStore.create(
            tmp_path / "d.zarr", ['w'], n_samples=1024, fs=10.0,
            split_level=4, margin=16)


# --------------------------------------------------------------------------- #
# crash consistency: a kill mid-chunk must not corrupt the resumed store
# --------------------------------------------------------------------------- #
@needs_zarr
def test_crash_mid_chunk_then_resume_stays_exact(tmp_path, monkeypatch):
    n, split, level = 2 ** 12, 3, 6
    x = _signal(n, 9)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=10.0,
        split_level=split, chunk_size=2 ** 10, details_dtype='float64')

    # die partway through the second chunk's appends: cD01 of that chunk is
    # already extended, its qc/carry and the n_done commit are not
    calls = {'k': 0}
    orig = cascade._block_fraction

    def dying(mask, width, nblocks=None):
        calls['k'] += 1
        if calls['k'] == (split + 1) + 1:      # first qc call of chunk 2
            raise KeyboardInterrupt
        return orig(mask, width, nblocks)

    monkeypatch.setattr(cascade, '_block_fraction', dying)
    with pytest.raises(KeyboardInterrupt):
        cascade.cascade_run(store, {'w': x})
    monkeypatch.setattr(cascade, '_block_fraction', orig)
    # the partial chunk is on disk beyond n_done: exactly the corrupt state
    assert store['w']['cD01'].shape[0] > int(store['w'].attrs['n_done']) >> 1

    cascade.cascade_run(store, {'w': x})       # re-launch, as documented
    cascade.cascade_finish(store, level=level)

    full = _full_wavedec(x, level)
    np.testing.assert_allclose(store['w']['cA'][:], full[0], atol=1e-10)
    for j in range(1, level + 1):
        got = store['w'][f'cD{j:02d}'][:]
        assert got.shape == full[level - j + 1].shape
        np.testing.assert_allclose(got, full[level - j + 1], atol=1e-10,
                                   err_msg=f"cD{j}")


# --------------------------------------------------------------------------- #
# shipped defaults: float32 details, default chunk/margin/level
# --------------------------------------------------------------------------- #
@needs_zarr
def test_default_parameters_end_to_end(tmp_path):
    n, fs = 2 ** 14, 20.0        # smaller than the default chunk_size (2**21)
    w = _signal(n, 10)
    store = cascade.CascadeStore.create(
        tmp_path / "defaults.zarr", ['w'], n_samples=n, fs=fs,
        t0='2022-05-13')                        # all other params default
    cascade.cascade_run(store, {'w': w})
    cascade.cascade_finish(store)               # default level

    split = store.attrs['split_level']
    assert store['w']['cD01'][:].dtype == np.float32     # default archive
    assert store['w']['carry'][:].dtype == np.float64    # carry always f64
    assert store['w']['cA'][:].dtype == np.float64

    level = store.attrs['level']
    full = _full_wavedec(w, level)
    for j in (1, split, level):
        got = store['w'][f'cD{j:02d}'][:]
        ref = full[level - j + 1]
        assert got.shape == ref.shape
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(got - ref)) / scale < 1e-6  # float32 rounding


# --------------------------------------------------------------------------- #
# dask branches must agree with the eager paths
# --------------------------------------------------------------------------- #
@needs_zarr
def test_dask_paths_match_eager(tmp_path, monkeypatch):
    n, split, level, fs = 2 ** 12, 4, 8, 4.0
    w_, c_ = _signal(n, 11), _signal(n, 12)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w', 'co2'], n_samples=n, fs=fs,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {'w': w_, 'co2': c_})
    cascade.cascade_finish(store, level=level)

    ref64 = cascade.scale_covariance(store, 'w', 'co2', window=64 / fs)
    ref48 = cascade.scale_covariance(store, 'w', 'co2', window=48 / fs)
    ref_lvl = cascade.open_level(store, 'w', 3).values

    monkeypatch.setattr(cascade, '_TIME_COORD_MAX', 0)   # force dask branches
    d64 = cascade.scale_covariance(store, 'w', 'co2', window=64 / fs)
    d48 = cascade.scale_covariance(store, 'w', 'co2', window=48 / fs)
    np.testing.assert_allclose(d64['wco2'].values, ref64['wco2'].values,
                               atol=1e-12)
    np.testing.assert_allclose(d64['wco2_qc'].values, ref64['wco2_qc'].values,
                               atol=1e-12)
    np.testing.assert_allclose(d48['wco2'].values, ref48['wco2'].values,
                               atol=1e-12)
    da = cascade.open_level(store, 'w', 3)
    assert hasattr(da.data, 'dask')                      # lazy
    assert 'TIMESTAMP' not in da.coords                  # too large for coords
    np.testing.assert_allclose(np.asarray(da.values), ref_lvl, atol=0)


# --------------------------------------------------------------------------- #
# callable and xarray sources
# --------------------------------------------------------------------------- #
@needs_zarr
def test_callable_and_xarray_sources_match_array(tmp_path):
    n, split = 2 ** 12, 3
    x = _signal(n, 13)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['a', 'b', 'c'], n_samples=n, fs=10.0,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {
        'a': x,
        'b': lambda i0, i1: x[i0:i1],
        'c': xr.DataArray(x, dims=['t']),
    })
    for var in ('b', 'c'):
        for j in range(1, split + 1):
            np.testing.assert_allclose(store[var][f'cD{j:02d}'][:],
                                       store['a'][f'cD{j:02d}'][:], atol=0)
        np.testing.assert_allclose(store[var]['carry'][:],
                                   store['a']['carry'][:], atol=0)


# --------------------------------------------------------------------------- #
# open_level metadata contract
# --------------------------------------------------------------------------- #
@needs_zarr
def test_open_level_validation_naming_and_frequency(tmp_path):
    n, split, level, fs = 2 ** 12, 4, 6, 20.0
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=fs,
        split_level=split, chunk_size=2 ** 11, details_dtype='float64')
    cascade.cascade_run(store, {'w': _signal(n, 14)})

    # before finish: approximation falls back to the carry, labelled _lf
    lf = cascade.open_level(store, 'w')
    assert lf.name == 'w_lf' and lf.attrs['array'] == 'carry'
    assert lf.attrs['natural_frequency'] == pytest.approx(fs * 2.0 ** -(split + 1))
    with pytest.raises(ValueError):          # stage-2 level not available yet
        cascade.open_level(store, 'w', split + 1)

    cascade.cascade_finish(store, level=level)
    lf = cascade.open_level(store, 'w')
    assert lf.name == 'w_lf' and lf.attrs['array'] == 'cA'
    # the approximation occupies the fs*2**-(level+1) slot, one octave below
    # the deepest detail -- no collision on natural_frequency
    assert lf.attrs['natural_frequency'] == pytest.approx(fs * 2.0 ** -(level + 1))
    deep = cascade.open_level(store, 'w', level)
    assert deep.attrs['natural_frequency'] == pytest.approx(fs * 2.0 ** -level)
    with pytest.raises(ValueError):          # beyond the finish level
        cascade.open_level(store, 'w', level + 1)

    # reconstruct: fail-fast guard against accidental full-record requests
    with pytest.raises(ValueError):
        old = cascade._TIME_COORD_MAX
        try:
            cascade._TIME_COORD_MAX = 100
            cascade.reconstruct(store, 'w', levels=[1, 2], start=0, stop=n)
        finally:
            cascade._TIME_COORD_MAX = old


# --------------------------------------------------------------------------- #
# gaps: the fill belongs to the record, not to the chunk
# --------------------------------------------------------------------------- #
def _store(path, data, split=3, chunk=2 ** 10, fs=10.0, **kwargs):
    kwargs.setdefault('details_dtype', 'float64')
    kwargs.setdefault('t0', '2022-05-13T00:00:00')
    store = cascade.CascadeStore.create(
        path, list(data), n_samples=len(next(iter(data.values()))), fs=fs,
        wavelet=WAVELET, split_level=split, chunk_size=chunk, **kwargs)
    cascade.cascade_run(store, data)
    return store


def _periodic_fill(x):
    """Reference: linear interpolation across gaps on the periodic record."""
    n = x.size
    idx = np.arange(n)
    finite = idx[np.isfinite(x)]
    return np.interp(idx, np.r_[finite - n, finite, finite + n],
                     np.tile(x[finite], 3))


@needs_zarr
@pytest.mark.parametrize("gaps", [
    [(1500, 1520)],                     # short
    [(1000, 1050)],                     # across a seam
    [(750, 3500)],                      # longer than any padded chunk
    [(0, 2500)],                        # from the record start
    [(1500, 2 ** 12)],                  # to the record end
    [(0, 100), (2 ** 12 - 50, 2 ** 12)],    # one gap through the record ends
    [(500, 1500), (1550, 3000)],        # two long gaps, a few samples apart
    [(936, 2136)],                      # exactly a padded chunk (margin 88)
    [(2 ** 12 - 88, 2 ** 12)],          # the first window starts on the gap
    [(1900, 2136)],                     # the gap ends with a window
])
@pytest.mark.parametrize("chunk", [2 ** 8, 2 ** 10, 2 ** 12])
def test_gaps_are_interpolated_whatever_their_length_and_the_chunks(
        tmp_path, gaps, chunk):
    n, level = 2 ** 12, 8
    x = 415.0 + 0.01 * _signal(n, 15)       # raw channel: nowhere near zero
    for a, b in gaps:
        x[a:b] = np.nan
    store = _store(tmp_path / "store.zarr", {'w': x}, chunk=chunk)
    cascade.cascade_finish(store, level=level)

    full = _full_wavedec(_periodic_fill(x), level)
    np.testing.assert_allclose(store['w']['cA'][:], full[0], atol=1e-9)
    for j in range(1, level + 1):
        np.testing.assert_allclose(store['w'][f'cD{j:02d}'][:],
                                   full[level - j + 1], atol=1e-9,
                                   err_msg=f"cD{j}")


def _counting(x):
    reads = []

    def source(i0, i1):
        reads.append(i1 - i0)
        return x[i0:i1]
    return source, reads


@needs_zarr
def test_long_outage_is_scanned_once_not_once_per_chunk(tmp_path):
    n, chunk = 2 ** 13, 2 ** 7
    x = _signal(n, 16)
    x[1000:7000] = np.nan
    source, reads = _counting(x)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=10.0, split_level=3,
        chunk_size=chunk, t0='2022-05-13')
    cascade.cascade_run(store, {'w': source})
    windows = (n // chunk) * (chunk + 2 * store.attrs['margin'])
    assert sum(reads) < windows + 3 * 6000      # the windows, plus the gap


@needs_zarr
def test_record_without_any_finite_sample(tmp_path):
    n, chunk = 2 ** 12, 2 ** 7
    source, reads = _counting(np.full(n, np.nan))
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=10.0, split_level=3,
        chunk_size=chunk, t0='2022-05-13')
    cascade.cascade_run(store, {'w': source})
    cascade.cascade_finish(store, level=6)
    assert not store['w']['cA'][:].any() and not store['w']['cD01'][:].any()
    assert store['w']['qc01'][:].min() == 1.0
    assert store['w']['qc_cA'][:].min() == 1.0
    windows = (n // chunk) * (chunk + 2 * store.attrs['margin'])
    assert sum(reads) < windows + 3 * n         # one look at the record


# --------------------------------------------------------------------------- #
# qc: nothing the gap touched may report 0, wherever the gap is
# --------------------------------------------------------------------------- #
@needs_zarr
@pytest.mark.parametrize("gap", [
    (3000, 3100), (300, 500), (600, 700), (800, 1000), (1124, 2148),
    (0, 100), (2 ** 13 - 100, 2 ** 13),
    (2 ** 13 - 700, 2 ** 13 - 600), (2 ** 13 - 1000, 2 ** 13 - 800),
])
def test_every_gap_influenced_coefficient_has_qc(tmp_path, gap):
    n, level = 2 ** 13, 9
    x = 0.05 * _signal(n, 17)
    gappy = x.copy()
    gappy[slice(*gap)] = np.nan
    clean = _store(tmp_path / "clean.zarr", {'w': x})
    store = _store(tmp_path / "gappy.zarr", {'w': gappy})
    cascade.cascade_finish(clean, level=level)
    cascade.cascade_finish(store, level=level)
    pairs = [('cA', 'qc_cA'), ('carry', 'qc_carry')] + [
        (f'cD{j:02d}', f'qc{j:02d}') for j in range(1, level + 1)]
    for name, qc in pairs:
        changed = np.abs(clean['w'][name][:] - store['w'][name][:]) > 1e-12
        assert changed.any() or name.startswith('cD')
        assert (store['w'][qc][:][changed] > 0).all(), name


@needs_zarr
def test_qc_arrays_line_up(tmp_path):
    n, split, level = 2 ** 14, 3, 9
    x = _signal(n, 18)
    a, b = 6144, 8192
    x[a:b] = np.nan
    store = _store(tmp_path / "store.zarr", {'w': x}, split=split,
                   chunk=2 ** 11)
    cascade.cascade_finish(store, level=level)
    np.testing.assert_array_equal(store['w']['qc_carry'][:],
                                  store['w'][f'qc{split:02d}'][:])
    np.testing.assert_array_equal(store['w']['qc_cA'][:],
                                  store['w'][f'qc{level:02d}'][:])
    for j in range(1, level + 1):
        assert (store['w'][f'qc{j:02d}'][:][a >> j:b >> j] == 1).all(), j


@needs_zarr
@pytest.mark.parametrize("gappy", ['x', 'y'])
def test_pair_qc_sees_a_gap_in_either_variable(tmp_path, gappy):
    n, fs, level = 2 ** 13, 4.0, 9
    data = {'x': _signal(n, 19), 'y': _signal(n, 20)}
    data[gappy][3000:3400] = np.nan
    store = _store(tmp_path / "store.zarr", data, fs=fs)
    cascade.cascade_finish(store, level=level)
    ds = cascade.scale_covariance(store, 'x', 'y', window=256 / fs)
    at_gap = ds.isel(TIMESTAMP=3200 // 256)
    assert (at_gap['xy_qc'].values > 0).all()
    assert float(at_gap['xy_lf_qc']) > 0


# --------------------------------------------------------------------------- #
# sources: a read is the samples asked for, by position
# --------------------------------------------------------------------------- #
@needs_zarr
@pytest.mark.parametrize("read", [
    lambda x, a, b: x[a:min(b, x.size - 8)],        # the file ends early
    lambda x, a, b: (np.delete(x[a:b], np.s_[:16])  # samples lost mid-record
                     if a <= 1500 < b else x[a:b]),
    lambda x, a, b: np.r_[x[a:b], x[:8]],           # too many samples
], ids=['early_end', 'lost_samples', 'too_many'])
def test_callable_reading_the_wrong_length_is_refused(tmp_path, read):
    n = 2 ** 12
    x = _signal(n, 21)
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=10.0, split_level=3,
        chunk_size=2 ** 10, t0='2022-05-13')
    with pytest.raises(ValueError, match="expected"):
        cascade.cascade_run(store, {'w': lambda a, b: read(x, a, b)})
    assert store['w'].attrs['n_done'] < n


@needs_zarr
@pytest.mark.parametrize("index", [
    None,
    lambda n: np.arange(n) / 10.0,
    lambda n: np.arange(n) + 5000,
    lambda n: pd.date_range('2022-05-13', periods=n, freq='100ms'),
], ids=['default', 'float', 'offset_int', 'datetime'])
def test_pandas_sources_are_read_by_position(tmp_path, index):
    n = 2 ** 12
    x = _signal(n, 22)
    series = pd.Series(x, index=None if index is None else index(n))
    ref = _store(tmp_path / "ref.zarr", {'w': x})
    got = _store(tmp_path / "got.zarr", {'w': series})
    np.testing.assert_array_equal(got['w']['carry'][:], ref['w']['carry'][:])


@needs_zarr
def test_inconsistent_store_is_refused(tmp_path):
    n = 2 ** 12
    x = _signal(n, 23)
    store = _store(tmp_path / "store.zarr", {'w': x})
    store['w']['cD01'].resize(((n >> 1) - 16,))
    with pytest.raises(ValueError, match="inconsistent"):
        cascade.cascade_finish(store, level=6)
    with pytest.raises(ValueError, match="inconsistent"):
        cascade.cascade_run(store, {'w': x})


# --------------------------------------------------------------------------- #
# handles and re-finishing
# --------------------------------------------------------------------------- #
@needs_zarr
def test_handles_see_what_other_handles_did(tmp_path):
    n, fs = 2 ** 13, 20.0
    path = tmp_path / "store.zarr"
    early = cascade.CascadeStore.create(
        path, ['w'], n_samples=n, fs=fs, split_level=3, chunk_size=2 ** 10,
        details_dtype='float64', t0='2022-05-13')
    cascade.cascade_run(path, {'w': _signal(n, 24)})
    cascade.cascade_finish(path, level=6)
    held = cascade.CascadeStore.open(path, mode='r')
    assert early.attrs['level'] == held.attrs['level'] == 6

    cascade.cascade_finish(path, level=8)
    window = 300 / fs                     # deep levels: fractional path
    ref = cascade.scale_covariance(path, 'w', 'w', window=window)
    for handle in (early, held):
        lf = cascade.open_level(handle, 'w')
        assert lf.attrs['level'] == 8 and lf.sizes['TIMESTAMP'] == n >> 8
        step = (lf.TIMESTAMP.values[1] - lf.TIMESTAMP.values[0])
        assert step / np.timedelta64(1, 's') == pytest.approx(2 ** 8 / fs)
        ds = cascade.scale_covariance(handle, 'w', 'w', window=window)
        assert ds.attrs['lf_level'] == 8
        np.testing.assert_array_equal(ds['ww_lf'].values, ref['ww_lf'].values)


@needs_zarr
def test_refused_finish_leaves_the_store_finished(tmp_path):
    n, split, level = 2 ** 12, 3, 7
    store = _store(tmp_path / "store.zarr", {'w': _signal(n, 25)}, split=split)
    cascade.cascade_finish(store, level=level)
    with pytest.raises(ValueError):
        cascade.cascade_finish(store, level=split)
    assert store.attrs['level'] == level
    deepest = cascade.open_level(store, 'w', level)
    assert deepest.sizes['TIMESTAMP'] == n >> level


@needs_zarr
def test_finish_can_go_deeper_and_back(tmp_path):
    n = 2 ** 13
    x = _signal(n, 26)
    store = _store(tmp_path / "store.zarr", {'w': x})
    for level in (9, 5, 10, 4):
        cascade.cascade_finish(store, level=level)
        full = _full_wavedec(x, level)
        np.testing.assert_allclose(store['w']['cA'][:], full[0], atol=1e-9)
        for j in range(1, level + 1):
            np.testing.assert_allclose(store['w'][f'cD{j:02d}'][:],
                                       full[level - j + 1], atol=1e-9)
        names = set(store['w'].keys())
        assert f'cD{level + 1:02d}' not in names
        assert f'qc{level + 1:02d}' not in names


@needs_zarr
def test_finish_invalidates_the_level_before_rewriting(tmp_path, monkeypatch):
    store = _store(tmp_path / "store.zarr", {'w': _signal(2 ** 12, 27)})
    cascade.cascade_finish(store, level=7)
    seen = []
    orig = cascade._create_array

    def spy(*args, **kwargs):
        seen.append(store.attrs['level'])
        return orig(*args, **kwargs)

    monkeypatch.setattr(cascade, '_create_array', spy)
    cascade.cascade_finish(store, level=6)
    assert seen and all(level is None for level in seen)
    assert store.attrs['level'] == 6


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
@needs_zarr
@pytest.mark.parametrize("bad, message", [
    ({'n_samples': 0}, "n_samples"), ({'n_samples': -16}, "n_samples"),
    ({'chunk_size': 0}, "chunk_size"), ({'chunk_size': -16}, "chunk_size"),
    ({'split_level': -1}, "negative"), ({'fs': 0}, "fs"), ({'fs': -1.0}, "fs"),
    ({'fs': float('nan')}, "fs"), ({'variables': []}, "variables"),
    ({'variables': ['w', 'w']}, "variables"),
], ids=str)
def test_create_refuses_nonsense_before_writing(tmp_path, bad, message):
    args = {'variables': ['w'], 'n_samples': 2 ** 12, 'fs': 10.0,
            'split_level': 3, 'chunk_size': 2 ** 10}
    args.update(bad)
    with pytest.raises(ValueError, match=message):
        cascade.CascadeStore.create(tmp_path / "store.zarr", **args)
    assert not (tmp_path / "store.zarr").exists()


@needs_zarr
def test_a_single_variable_name_is_not_split_into_letters(tmp_path):
    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", 'co2', n_samples=2 ** 12, fs=10.0,
        split_level=3, chunk_size=2 ** 10, t0='2022-05-13')
    assert store.variables == ['co2']


@needs_zarr
@pytest.mark.parametrize("kwargs", [
    {'window': 0.0}, {'window': -5.0}, {'window': 1e4},
    {'window': 64.0, 'levels': [9]}, {'window': 64.0, 'levels': [0]},
    {'window': 64.0, 'levels': []},
], ids=str)
def test_scale_covariance_refuses_unusable_requests(tmp_path, kwargs):
    store = _store(tmp_path / "store.zarr", {'w': _signal(2 ** 12, 28)},
                   fs=4.0)
    cascade.cascade_finish(store, level=8)
    with pytest.raises(ValueError):
        cascade.scale_covariance(store, 'w', 'w', **kwargs)


@needs_zarr
def test_a_half_written_variable_is_refused(tmp_path):
    n = 2 ** 13
    x = _signal(n, 29)
    reads = []

    def dying(i0, i1):
        reads.append(i0)
        if len(reads) > 4:
            raise RuntimeError("the disk went away")
        return x[i0:i1]

    store = cascade.CascadeStore.create(
        tmp_path / "store.zarr", ['w'], n_samples=n, fs=4.0, split_level=3,
        chunk_size=2 ** 10, details_dtype='float64', t0='2022-05-13')
    with pytest.raises(RuntimeError):
        cascade.cascade_run(store, {'w': dying})
    assert 0 < store['w'].attrs['n_done'] < n

    # 101 samples: odd, so every level takes the fractional path, which used
    # to answer zeros for what was not written yet
    with pytest.raises(ValueError, match="not fully decomposed"):
        cascade.scale_covariance(store, 'w', 'w', window=101 / 4.0)
    with pytest.raises(ValueError, match="not fully decomposed"):
        cascade.reconstruct(store, 'w', levels=[1], start=100, stop=110)


# --------------------------------------------------------------------------- #
# time axes and the odd halving chain
# --------------------------------------------------------------------------- #
@needs_zarr
def test_time_axes(tmp_path):
    n, fs = 2 ** 13, 4.0
    t0 = np.datetime64('2022-05-12T23:30:00.05', 'ns')
    store = _store(tmp_path / "store.zarr", {'w': _signal(n, 30)}, fs=fs,
                   t0='2022-05-12T23:30:00.05')
    ds = cascade.scale_covariance(store, 'w', 'w', window=64.0)
    assert ds.TIMESTAMP.values[0] == t0                  # window starts
    assert ds.TIMESTAMP.values[1] - t0 == np.timedelta64(64, 's')
    rec = cascade.reconstruct(store, 'w', levels=[1], start=8, stop=12)
    assert rec.TIMESTAMP.values[0] == t0 + np.timedelta64(2, 's')   # samples


def _density_window_mean(p, step, w, nwin, n):
    """Reference for records n does not tile: the last coefficient spreads
    over what remains of the record."""
    density = np.zeros(n)
    for k in range(max(1, min(p.size, -(-n // step)))):
        a, b = k * step, min((k + 1) * step, n)
        density[a:b] = p[k] / (b - a)
    return density[: nwin * w].reshape(nwin, w).sum(axis=1) / w


@needs_zarr
@pytest.mark.parametrize("window", [64, 100, 1001, 2048])
def test_window_means_on_an_odd_halving_chain(tmp_path, window):
    n, fs, level = 2 ** 3 * 1001, 4.0, 11
    data = {'x': _signal(n, 31), 'y': _signal(n, 32)}
    store = _store(tmp_path / "store.zarr", data, fs=fs)
    cascade.cascade_finish(store, level=level)
    ds = cascade.scale_covariance(store, 'x', 'y', window=window / fs)
    for j in range(1, level + 1):
        p = store['x'][f'cD{j:02d}'][:] * store['y'][f'cD{j:02d}'][:]
        np.testing.assert_allclose(
            ds['xy'].sel(natural_frequency=fs * 2.0 ** -j).values,
            _density_window_mean(p, 2 ** j, window, n // window, n),
            atol=1e-8, rtol=1e-8, err_msg=f"level {j}")
