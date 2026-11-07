"""
Fixed calendar grid for the discrete wavelet transform.

A discrete transform keeps one coefficient every ``2**level`` samples, counted
from the first sample it is given. A period decomposed on its own therefore
sits somewhere else on that grid than the same period inside a longer run, and
the energy of each half-hour moves between scales.

Here every sample has a permanent number, counted from the Unix epoch, and
every run starts on a multiple of ``2**level``. A run reads the neighbours of
its period on each side, decomposes once with the cascade, and each period
reads its levels back: a period's result depends neither on how the run is cut
into periods, nor, away from the ends of the data, on where the run starts and
stops.

The MODWT has no grid to keep, but it needs the rest: the samples are kept
instead of coefficients, and each period filters its stretch of them with the
neighbours each level needs (see modwt). Its result does not depend on the
block either, down to rounding. The same samples serve the MODWT details
(method 'moda').

Samples are placed by their timestamp, never by counting rows: what is missing
stays missing on the grid, and is filled and flagged by the cascade. So are the
neighbours that do not exist at the ends of the data.

The main functions are:
- sample_number
    function: the permanent number of a sample
    call: sample_number('2022-05-14T10:00', fs=20)
- grid_bounds
    function: first and last sample a run reads
    call: grid_bounds(start, stop, fs=20, level=15)
- why_not_on_grid
    function: check that files fit the grid
    call: why_not_on_grid(input_path, read, start, stop, fs, fileduration)
- build_store
    function: read the files onto the grid and decompose them
    call: build_store(folder, identifier, input_path, read, summarise, ...)
- read_windows
    function: what a period needs of its files, kept by build_store
    call: read_windows(store, labels)
- open_block
    function: read the windows of a period back from a store
    call: open_block(path, timestamps, window, method).decomposed(variables, level, low)
"""

# built-in modules
import json
import logging
import os
import shutil
from fractions import Fraction

# 3rd party modules
import joblib
import numpy as np
import pandas as pd
from scipy import fft
import xarray as xr

# project modules
from . import cascade, modwt


logger = logging.getLogger(__name__)

ORIGIN = '1970-01-01T00:00:00'
DEFAULT_F0 = 1 / (3 * 60 * 60)
SAMPLES = '.samples'


def _rate(fs):
    return Fraction(fs).limit_denominator(10 ** 6)


def sample_number(time, fs):
    """
    function: number of the sample taken at *time*, counted from the Unix
        epoch in whole numbers, so that no rounding can move a sample
    call: sample_number()
    Input:
        time: anything np.datetime64 accepts
        fs: sampling rate (Hz)
    Return:
        number: python int
    """
    nanoseconds = int(np.datetime64(time, 'ns').astype('int64'))
    rate = _rate(fs)
    return nanoseconds * rate.numerator // (rate.denominator * 10 ** 9)


def sample_time(number, fs):
    """Timestamp of sample *number* (c.f. sample_number)."""
    rate = _rate(fs)
    return np.datetime64(number * rate.denominator * 10 ** 9 // rate.numerator, 'ns')


def deepest_level(fs, f0=DEFAULT_F0):
    """Level reaching frequency *f0*, as universal_wt works it out."""
    return int(np.ceil(np.log2(fs / f0)))


SAMPLE_METHODS = ('modwt', 'moda')


def _margin(level, wavelet, method):
    if method in SAMPLE_METHODS:
        # and a step more, for flags that follow the grid (see SampleBlock)
        return modwt.reach(level, wavelet) + 2 ** int(level)
    return cascade.cascade_margin(level, wavelet)


def grid_bounds(start, stop, fs, level, wavelet=cascade.DEFAULT_WAVELET, method='dwt'):
    """
    function: first and last sample a run of [start, stop) reads: the period
        with its neighbours on each side, rounded out to the grid of *level*
    call: grid_bounds()
    Input:
        start, stop: the period, as anything np.datetime64 accepts
        fs: sampling rate (Hz)
        level: deepest level of the run
        wavelet: mother wavelet (comprehensible to pywt)
        method: 'dwt', or 'modwt' or 'moda', which need different neighbours
    Return:
        first, last: sample numbers, multiples of ``2**level``
    """
    step = 2 ** int(level)
    margin = _margin(level, wavelet, method)
    first = (sample_number(start, fs) - margin) // step * step
    last = -(-(sample_number(stop, fs) + margin) // step) * step
    return first, last


def _paths_by_label(input_path):
    """Files of *input_path* by the ``%Y%m%d-%H%M`` their name starts with."""
    paths = {}
    for name in sorted(os.listdir(input_path)):
        paths.setdefault(name[:13], []).append(os.path.join(input_path, name))
    return paths


def _labels(start, stop, fileduration):
    """End of each window of [start, stop), the time files are named after."""
    duration = pd.Timedelta(minutes=fileduration)
    return pd.date_range(pd.Timestamp(start) + duration, pd.Timestamp(stop), freq=duration)


def why_not_on_grid(input_path, read, start, stop, fs, fileduration):
    """
    function: check a few files of [start, stop) against what the grid needs,
        one window of the expected length per file
    call: why_not_on_grid()
    Input:
        input_path: folder of the files, named after the end of their window
        read: function reading the path of a file into a Dataset
        start, stop: the period of the run
        fs: sampling rate (Hz)
        fileduration: minutes covered by a file
    Return:
        reason: text saying what is wrong, or None if the grid can be used
    """
    paths = _paths_by_label(input_path)
    found = [paths[label] for label in (
        t.strftime('%Y%m%d-%H%M') for t in _labels(start, stop, fileduration))
        if label in paths]
    if not found:
        return f"no file to read in {input_path!r}"
    window = int(round(fileduration * 60 * fs))
    sizes = []
    for files in (found[0], found[len(found) // 2], found[-1]):
        try:
            data = read(files[0])
        except Exception as e:
            logger.error(f"Error when reading {files[0]}: {e}")
            continue
        if 'ns' not in data.dims:
            return "the reader does not return one window per file (no 'ns' dimension)"
        sizes.append(data.sizes['ns'])
    if not sizes:
        return "the files checked could not be read"
    if abs(np.median(sizes) - window) > .1 * window:
        return (f"files hold {int(np.median(sizes))} samples, not the {window} that "
                f"{fs} Hz and {fileduration} minutes give")
    return None


def _stage_file(label, files, read, summarise, staged, first, n, fs, fileduration):
    """Put the samples of the window ending at *label* in their place on the grid."""
    window = int(round(fileduration * 60 * fs))
    offset = sample_number(label - pd.Timedelta(minutes=fileduration), fs) - first
    summary, errors = None, []
    for path in files:
        try:
            data = read(path)
        except Exception as e:
            errors.append(f"Error when reading {path}: {e}")
            continue
        for var, target in staged.items():
            if var not in data:
                continue
            values = np.asarray(data[var].values, dtype='float64').ravel()[:window]
            lo, hi = max(offset, 0), min(offset + values.size, n)
            if lo < hi:
                with open(target, 'r+b') as file:
                    file.seek(8 * lo)
                    values[lo - offset: hi - offset].tofile(file)
        means = summarise(data)
        summary = {var: float(means[var].values.ravel()[0]) for var in means.data_vars
                   if means[var].dtype.kind in 'fiub'}
        summary['TIMESTAMP'] = label
    return summary, errors


def _stage(folder, input_path, read, summarise, variables, start, first, last, fs,
           fileduration, n_jobs):
    """
    Read each file once: write the samples of each variable to a flat file, at
    their place on the grid and NaN where nothing was measured, so that the
    cascade can read any stretch of a variable, and return what periods will
    need of the rest of the file, as one row per window.
    """
    n = last - first
    os.makedirs(folder, exist_ok=True)
    staged = {var: os.path.join(folder, f"{i}.f8") for i, var in enumerate(variables)}
    blank = np.full(min(n, 2 ** 20), np.nan)
    for path in staged.values():
        with open(path, 'wb') as file:
            for s in range(0, n, blank.size):
                blank[:n - s].tofile(file)

    duration = pd.Timedelta(minutes=fileduration)
    window = int(round(fileduration * 60 * fs))
    begin = sample_number(start, fs)
    before = -(-(begin - first) // window)
    after = -(-(last - begin) // window)
    labels = pd.date_range(pd.Timestamp(start) + (1 - before) * duration,
                           periods=before + after + 1, freq=duration)
    paths = _paths_by_label(input_path)
    tasks = [(label, paths[name]) for label, name in
             ((t, t.strftime('%Y%m%d-%H%M')) for t in labels) if name in paths]
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_stage_file)(label, files, read, summarise, staged, first, n,
                                    fs, fileduration)
        for label, files in tasks)
    for _, errors in results:
        for error in errors:
            logger.error(error)
    rows = [summary for summary, _ in results if summary is not None]
    if not rows:
        raise ValueError(f"No file to read in {input_path!r}.")
    logger.info("Read %d files onto the grid.", len(rows))
    table = pd.DataFrame(rows).set_index('TIMESTAMP').sort_index()
    return staged, table[~table.index.duplicated(keep='last')]


def _staged_source(path):
    def read(i0, i1):
        return np.fromfile(path, dtype='float64', count=i1 - i0, offset=8 * i0)
    return read


def _windows_path(store):
    return f"{store}.windows.nc"


def read_windows(store, labels):
    """
    function: what a period needs of its files, saved by build_store: the means
        of each window, and whatever else *summarise* gave
    call: read_windows()
    Input:
        store: path of the store
        labels: end of each window of the period
    Return:
        ds: xarray.Dataset along TIMESTAMP, of the windows that have a file
    """
    windows = xr.load_dataset(_windows_path(store))
    return windows.sel(TIMESTAMP=np.intersect1d(
        windows.TIMESTAMP.values, np.asarray(labels, dtype='datetime64[ns]')))


class SampleStore:
    """
    Samples of a run at their place on the grid, NaN where nothing was measured:
    what the MODWT keeps instead of the coefficients of the cascade.
    """

    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'attrs.json'), encoding='utf-8') as file:
            self.attrs = json.load(file)

    def source(self, var):
        """Function reading the samples [i0, i1) of *var* (c.f. cascade_run)."""
        return _staged_source(os.path.join(self.path, self.attrs['files'][var]))

    def is_decomposed(self, var):
        return var in self.attrs['files']


def _open_store(path):
    if path.endswith(SAMPLES):
        return SampleStore(path)
    return cascade.CascadeStore.open(path, mode='r')


def _is_complete(path, variables, first, last, fs, level, wavelet):
    if not os.path.exists(path):
        return False
    try:
        store = _open_store(path)
    except Exception:
        return False
    a = store.attrs
    if not (a.get('first_sample') == first and a['n_samples'] == last - first
            and a['fs'] == float(fs) and a['split_level'] == level
            and a['wavelet'] == wavelet and set(variables) <= set(a['variables'])
            and os.path.exists(_windows_path(path))):
        return False
    if isinstance(store, SampleStore):
        return True
    return all(int(store[v].attrs['n_done']) == a['n_samples'] for v in variables)


def build_store(folder, identifier, input_path, read, summarise, variables, start,
                stop, fs, fileduration, level, wavelet=cascade.DEFAULT_WAVELET,
                overwrite=False, n_jobs=1, method='dwt'):
    """
    function: read the files of a run onto the grid, once each, and decompose
        them. A finished store of the same run is used as it is
    call: build_store()
    Input:
        folder: where to keep the store
        identifier: start of the name of the store
        input_path: folder of the files, named after the end of their window
        read: function reading the path of a file into a Dataset
        summarise: function reducing such a Dataset to what periods need of the
            file (c.f. read_windows), along TIMESTAMP alone
        variables: list of variables to decompose
        start, stop: the period of the run
        fs: sampling rate (Hz)
        fileduration: minutes covered by a file
        level: deepest level of the run
        wavelet: mother wavelet (comprehensible to pywt)
        overwrite: decompose again even if a finished store exists
        n_jobs: files to read at once (c.f. joblib.Parallel)
        method: 'dwt' to decompose with the cascade, 'modwt' or 'moda' to keep
            the samples
    Return:
        path: path of the store
    """
    first, last = grid_bounds(start, stop, fs, level, wavelet, method)
    name = f"{identifier}_{first}_{last}_L{level}"
    path = os.path.join(folder, name + (SAMPLES if method in SAMPLE_METHODS else '.zarr'))
    if not overwrite and _is_complete(path, variables, first, last, fs, level, wavelet):
        logger.info("Using the decomposition in %s.", path)
        return path

    os.makedirs(folder, exist_ok=True)
    if method in SAMPLE_METHODS:
        return _keep_samples(path, input_path, read, summarise, variables, start, stop,
                             first, last, fs, fileduration, level, wavelet, n_jobs)

    store = cascade.CascadeStore.create(
        path, variables, last - first, fs, wavelet=wavelet, split_level=level,
        t0=sample_time(first, fs), details_dtype='float64', overwrite=True)
    store.root.attrs['first_sample'] = first
    staging = f"{path}.staging"
    try:
        staged, windows = _stage(staging, input_path, read, summarise, variables,
                                 start, first, last, fs, fileduration, n_jobs)
        cascade.cascade_run(
            store, {var: _staged_source(file) for var, file in staged.items()})
        windows.to_xarray().to_netcdf(_windows_path(path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


def _keep_samples(path, input_path, read, summarise, variables, start, stop, first,
                  last, fs, fileduration, level, wavelet, n_jobs):
    """build_store of the MODWT: the staged samples are the store."""
    shutil.rmtree(path, ignore_errors=True)
    staging = f"{path}.staging"
    try:
        staged, windows = _stage(staging, input_path, read, summarise, variables, start,
                                 first, last, fs, fileduration, n_jobs)
        windows.to_xarray().to_netcdf(_windows_path(path))
        with open(os.path.join(staging, 'attrs.json'), 'w', encoding='utf-8') as file:
            json.dump({'first_sample': first, 'n_samples': last - first, 'fs': float(fs),
                       'split_level': level, 'wavelet': wavelet,
                       'variables': list(variables),
                       'files': {var: os.path.basename(p) for var, p in staged.items()}},
                      file)
        os.replace(staging, path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


class _Block:
    """
    The windows of a period, read back from a store as decompose_variables
    returns them, one level at a time so that memory follows the period and
    not the number of levels.

    Input:
        store: what build_store made
        timestamps: end of each window of the period
        window: samples per window
    """

    def __init__(self, store, timestamps, window):
        a = store.attrs
        self.store = store
        self.fs = a['fs']
        self.window = int(window)
        self.levels = range(1, a['split_level'] + 1)
        self.timestamps = timestamps
        self.ns = np.arange(self.window)
        self.attrs = {'fs': self.fs, 'wavelet': a['wavelet']}
        starts = np.array([sample_number(t, self.fs) - self.window - a['first_sample']
                           for t in self.timestamps])
        self.start = int(starts.min())
        self.stop = int(starts.max()) + self.window
        if self.start < 0 or self.stop > a['n_samples']:
            raise ValueError(
                f"Windows {self.timestamps[0]} to {self.timestamps[-1]} are "
                "outside the store.")
        if np.any((starts - self.start) % self.window):
            raise ValueError("Windows do not follow each other regularly.")
        self.rows = (starts - self.start) // self.window

    def _windows(self, series):
        """Cut a series of the whole block into (TIMESTAMP, ns)."""
        return series.reshape(-1, self.window)[self.rows]

    def approximation(self, var):
        """The part of *var* below the deepest level, as (TIMESTAMP, ns)."""
        return self._windows(self._approximation(var))

    def signal(self, var):
        """The gap-filled signal of *var*, as (TIMESTAMP, ns)."""
        return self._windows(self._signal(var))

    def decomposed(self, variables, level, low, signal=None):
        """
        function: one level of *variables*, for data_run
        call: decomposed()
        Input:
            variables: list of variables
            level: level to read
            low: dict of variable to its approximation
            signal: dict of variable to its gap-filled signal, for the
                cross-level flux (c.f. handlers.data_compute_flux), or None
        Return:
            ds: xarray.Dataset with <var>, <var>_qc and <var>_lf, and
                <var>_signal when *signal* is given
        """
        dims = ('natural_frequency', 'TIMESTAMP', 'ns')
        out = xr.Dataset(coords={'natural_frequency': [self.fs * 2.0 ** -level],
                                 'TIMESTAMP': self.timestamps, 'ns': self.ns})
        for var in variables:
            out[var] = (dims, self._level_windows(var, level)[np.newaxis])
            out[f'{var}_qc'] = (dims, self._windows(self._gaps(var, level))[np.newaxis])
            out[f'{var}_lf'] = (dims[1:], low[var])
            if signal is not None:
                out[f'{var}_signal'] = (dims[1:], signal[var])
        out.attrs.update(self.attrs)
        return out


    def _level_windows(self, var, level):
        return self._windows(self._level(var, level))


class Block(_Block):
    """A period read back from the cascade of a discrete transform."""

    def __init__(self, store, timestamps, window):
        super().__init__(store, timestamps, window)
        a = store.attrs
        # not 'mode': scipy keeps the mode of the file under that attribute
        self.attrs.update({'method': 'dwt', 'wavelet_mode': a['mode'],
                           'grid_origin': ORIGIN, 'grid_level': a['split_level']})

    def _level(self, var, level):
        return cascade.reconstruct(self.store, var, levels=[level], start=self.start,
                                   stop=self.stop).values[0]

    def _gaps(self, var, level):
        """Share of filled samples within reach of each sample of *level*."""
        width = 2 ** level
        first, last = self.start // width, -(-self.stop // width)
        qc = np.asarray(self.store[var][f'qc{level:02d}'][first:last], dtype='float32')
        return np.repeat(qc, width)[self.start - first * width: self.stop - first * width]

    def _approximation(self, var):
        return cascade.reconstruct_approximation(self.store, var, self.start, self.stop)

    def _signal(self, var):
        return cascade.reconstruct_signal(self.store, var, self.start, self.stop)


class SampleBlock(_Block):
    """
    A period filtered from the samples of a run, for the MODWT: its
    coefficients with method 'modwt', its details with 'moda', which filter
    with the square of the gain.
    """

    def __init__(self, store, timestamps, window, method='modwt'):
        super().__init__(store, timestamps, window)
        if method not in SAMPLE_METHODS:
            raise ValueError(f"No such method for samples: {method!r}.")
        self.attrs['method'] = method
        self.power = 2 if method == 'moda' else 1
        self.wavelet = store.attrs['wavelet']
        self.deepest = store.attrs['split_level']
        self.margin = _margin(self.deepest, self.wavelet, 'modwt')
        self.size = self.stop - self.start
        self._samples = {}

    def _read(self, var):
        """The samples around the period, gaps filled, and which were filled."""
        if var not in self._samples:
            n = self.store.attrs['n_samples']
            source = self.store.source(var)
            begin = self.start - self.margin
            raw = cascade._read_wrapped(source, begin, self.stop + self.margin, n)
            self._samples[var] = cascade._fill_window(raw, begin, source, n, 2 ** 20, [])
        return self._samples[var]

    def _filtered(self, var, margin, gain):
        """
        The period, filtered with *margin* neighbours on each side, and the mean
        taken out first, which a filter of a level drops anyway.
        """
        stretch = self._read(var)[0][self.margin - margin:
                                     self.margin + self.size + margin]
        mean = stretch.mean()
        n = fft.next_fast_len(stretch.size, real=True)
        padded = np.pad(stretch - mean, (0, n - stretch.size), mode='edge')
        return fft.irfft(fft.rfft(padded) * gain(n), n)[margin: margin + self.size], mean

    def _level_windows(self, var, level):
        """
        A level without rounding noise, against the size of the signal in each
        window: that belongs to the measurements, and to no run in particular.
        """
        series, _ = self._filtered(var, modwt.reach(level, self.wavelet),
                                   lambda n: self._gain(modwt.detail_gain(n, level, self.wavelet)))
        scale = np.abs(self._windows(self._signal(var))).max(axis=1, keepdims=True)
        return modwt.clean(self._windows(series), scale)

    def _gain(self, gain):
        return gain if self.power == 1 else gain ** self.power

    def _approximation(self, var):
        series, mean = self._filtered(
            var, modwt.reach(self.deepest, self.wavelet),
            lambda n: self._gain(modwt.approximation_gain(n, self.deepest, self.wavelet)))
        return series + mean

    def _signal(self, var):
        return self._read(var)[0][self.margin: self.margin + self.size]

    def _gaps(self, var, level):
        """
        Share of filled samples within reach of each sample of *level*, counted
        in blocks that follow the grid, so that it does not depend on the period.
        """
        width = 2 ** level
        margin = modwt.reach(level, self.wavelet)
        first = (self.start - margin) // width * width
        last = -(-(self.stop + margin) // width) * width
        offset = self.margin - self.start
        stretch = self._read(var)[1][first + offset: last + offset]
        shares = cascade._block_fraction(stretch, width, (last - first) // width)
        qc = cascade._dilate_max(shares, margin // width).astype('float32')
        return np.repeat(qc, width)[self.start - first: self.start - first + self.size]


def open_block(path, timestamps, window, method=None):
    """
    function: the windows of a period, from the store of a run
    call: open_block()
    Input:
        path: path of the store
        timestamps: end of each window of the period
        window: samples per window
        method: 'modwt' or 'moda' for a store of samples (default 'modwt');
            a cascade store is always 'dwt'
    Return:
        block: Block or SampleBlock
    """
    store = _open_store(path)
    if isinstance(store, SampleStore):
        return SampleBlock(store, timestamps, window,
                           method if method in SAMPLE_METHODS else 'modwt')
    return Block(store, timestamps, window)
