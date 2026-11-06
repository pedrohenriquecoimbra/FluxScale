"""
Deconstructed (chunked) discrete wavelet cascade.

Exact DWT of arbitrarily long records (years of 20 Hz data) with bounded
memory, in two stages:

- Stage 1 (`cascade_run`): the record is walked chunk by chunk. Each chunk is
  read together with a margin of *real* neighbouring samples, decomposed with
  ``pywt.wavedec(..., mode='periodization')`` down to ``split_level``, the
  boundary-contaminated edge coefficients are trimmed away, and the remaining
  coefficients are appended to a zarr store (one array per level, per
  variable). The level-``split_level`` approximation (the "carry") is stored
  as well.
- Stage 2 (`cascade_finish`): the concatenated carry -- a factor
  ``2**split_level`` shorter than the record -- is decomposed in memory down
  to the requested final level and the deep coefficients are written to the
  store.

Because ``wavedec`` is recursive (levels ``split_level+1..level`` of the full
transform are exactly the transform of the level-``split_level``
approximation), the two-stage result equals, to machine precision, the
coefficients ``pywt.wavedec`` would return on the whole record at once.
With an orthonormal periodized DWT the total number of stored coefficients
equals the number of samples, regardless of depth, when ``n_samples`` is
divisible by ``2**level``.

The main functions are:
- cascade_run
    function: stage 1, stream the record through chunked wavedec into a store
    call: cascade_run(store, {'w': w_array, 'co2': co2_array})
- cascade_finish
    function: stage 2, decompose the carried approximation to the final level
    call: cascade_finish(store, level=None)
- open_level
    function: read one level of coefficients back as an xarray.DataArray
    call: open_level(store, 'w', 12)
- scale_covariance
    function: per-scale covariance time series computed in coefficient space
    call: scale_covariance(store, 'w', 'co2', window=1800)
- reconstruct
    function: per-level time series for a sample window (stage-1 levels)
    call: reconstruct(store, 'co2', levels=[1, 2, 3], start=a, stop=b)
"""

# built-in modules
import logging

# 3rd party modules
import numpy as np
import pywt
import xarray as xr
try:
    import zarr
except ImportError:
    zarr = None

# project modules

logger = logging.getLogger(__name__)

# The only pywt extension mode whose per-level coefficient counts are exactly
# n/2 (no redundant boundary coefficients), which is what keeps chunked and
# full transforms aligned. Do not change on an existing store.
MODE = 'periodization'
DEFAULT_WAVELET = 'db6'
# zarr on-disk chunk length (elements) for coefficient arrays
_ZARR_CHUNK = 2 ** 20
# open_level only materialises a TIMESTAMP coordinate up to this many
# coefficients (shallow levels of multi-year records get too large)
_TIME_COORD_MAX = 2 ** 24


def _require_zarr():
    if zarr is None:
        raise ImportError(
            "The cascade coefficient store requires `zarr`. "
            "Install it with `pip install zarr` "
            "(or `pip install waveletec[cascade]`).")


# --------------------------------------------------------------------------- #
# Pure chunk mathematics (no storage involved)
# --------------------------------------------------------------------------- #
def cascade_margin(split_level, wavelet=DEFAULT_WAVELET):
    """
    function: overlap margin (samples per side) that guarantees the trimmed
        chunk coefficients equal the full-record transform
    call: cascade_margin()
    Input:
        split_level: depth of the stage-1 decomposition
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        margin: samples per side, ``(filter_length - 1) * 2**split_level``
            (conservative; a multiple of ``2**split_level`` by construction)
    """
    filter_length = pywt.Wavelet(wavelet).dec_len
    return (filter_length - 1) * 2 ** int(split_level)


def chunk_wavedec(padded, margin, split_level, wavelet=DEFAULT_WAVELET):
    """
    function: wavedec of one padded chunk, trimmed of boundary-contaminated
        coefficients. ``padded`` must cover ``[start - margin, stop + margin)``
        of the record, where ``start``, ``margin`` and ``stop - start`` are
        multiples of ``2**split_level`` (the record being treated as periodic,
        i.e. reads wrap around the record ends).
    call: chunk_wavedec()
    Input:
        padded: 1D array, core chunk plus ``margin`` real samples on each side
        margin: samples per side (see cascade_margin)
        split_level: decomposition depth
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        details: dict {level j: trimmed cD_j} for j = 1..split_level
        carry: trimmed cA_split_level (input to the next cascade stage)
    """
    padded = np.asarray(padded, dtype='float64')
    if padded.ndim != 1:
        raise ValueError("Input must be a vector")
    split_level = int(split_level)
    margin = int(margin)
    step = 2 ** split_level
    core = padded.size - 2 * margin
    if margin % step:
        raise ValueError(
            f"margin ({margin}) must be a multiple of 2**split_level ({step}).")
    if core <= 0 or core % step:
        raise ValueError(
            f"chunk length ({core}) must be a positive multiple of "
            f"2**split_level ({step}).")

    coeffs = pywt.wavedec(padded, wavelet, level=split_level, mode=MODE)
    # coeffs = [cA_split, cD_split, cD_split-1, ..., cD_1]
    p = margin >> split_level
    carry = coeffs[0][p: p + (core >> split_level)]
    details = {}
    for i, j in enumerate(range(split_level, 0, -1), start=1):
        pj = margin >> j
        details[j] = coeffs[i][pj: pj + (core >> j)]
    return details, carry


def _read(source, i0, i1):
    """Read samples [i0, i1) from an array-like or a callable source."""
    if callable(source):
        out = source(i0, i1)
    elif hasattr(source, 'iloc'):   # pandas may slice by label otherwise
        out = source.iloc[i0:i1]
    else:
        out = source[i0:i1]
    out = np.asarray(out, dtype='float64')
    if out.shape != (i1 - i0,):
        # a read of the wrong length would shift every later coefficient
        raise ValueError(
            f"Reading samples [{i0}, {i1}) returned shape {out.shape}, "
            f"expected ({i1 - i0},).")
    return out


def _read_wrapped(source, i0, i1, n):
    """Read samples [i0, i1) treating the record of length *n* as periodic."""
    parts = []
    while i0 < i1:
        a = i0 % n
        b = min(a + (i1 - i0), n)
        parts.append(_read(source, a, b))
        i0 += b - a
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _fill_gaps(x, left=None, right=None):
    """Linear, index-based interpolation across non-finite samples.

    *left* and *right* are the (index, value) pairs bounding gaps that run
    past the ends of *x*, indices being relative to ``x[0]``. With no finite
    sample to interpolate from at all, *x* is filled with zeros.

    Returns (filled copy, gap mask).
    """
    x = np.array(x, dtype='float64')
    mask = ~np.isfinite(x)
    if mask.any():
        idx = np.arange(x.size)
        at, values = idx[~mask], x[~mask]
        if left is not None:
            at, values = np.r_[left[0], at], np.r_[left[1], values]
        if right is not None:
            at, values = np.r_[at, right[0]], np.r_[values, right[1]]
        x[mask] = np.interp(idx[mask], at, values) if at.size else 0.0
    return x, mask


def _nearest_finite(source, pos, step, n, block):
    """Nearest finite sample walking from *pos* in direction *step* (+1 or
    -1) around the periodic record of length *n*, *block* samples per read.

    Returns (position, value), the position unwrapped, or None if there is
    no finite sample in the record.
    """
    for offset in range(1, n + 1, block):
        width = min(block, n + 1 - offset)
        start = pos + offset if step > 0 else pos - offset - width + 1
        segment = _read_wrapped(source, start, start + width, n)
        hits = np.flatnonzero(np.isfinite(segment))
        if hits.size:
            k = int(hits[0] if step > 0 else hits[-1])
            return start + k, float(segment[k])
    return None


def _gap_bounds(source, pos, n, block, known):
    """The (left, right) samples bounding the gap that holds sample *pos*, as
    _nearest_finite returns them. A gap can outlast any window: *known* keeps
    the latest ones, so that an outage spanning many chunks is scanned once
    rather than once per chunk."""
    for bounds in known:
        if bounds[0] is None or bounds[0][0] < pos < bounds[1][0]:
            return bounds
    left = _nearest_finite(source, pos, -1, n, block)
    right = None if left is None else _nearest_finite(source, pos, 1, n, block)
    known.append((left, right))
    del known[:-2]              # a window reaches into two gaps at most
    return left, right


def _fill_window(raw, start, source, n, block, known):
    """_fill_gaps for the window *raw* beginning at sample *start*, looking
    in the record for what bounds the gaps running past the window's ends, so
    that the fill depends on the record alone and not on the chunk layout."""
    def bound(pos, side):
        found = _gap_bounds(source, pos, n, block, known)[side]
        return None if found is None else (found[0] - start, found[1])

    left = None if np.isfinite(raw[0]) else bound(start, 0)
    right = None if np.isfinite(raw[-1]) else bound(start + raw.size - 1, 1)
    return _fill_gaps(raw, left, right)


def _block_fraction(mask, width, nblocks=None):
    """Mean of *mask* over consecutive blocks of *width* (gap fraction per
    coefficient). A short tail is padded with its last value."""
    mask = np.asarray(mask, dtype='float64')
    if nblocks is None:
        nblocks = mask.size // width
    pad = nblocks * width - mask.size
    if pad > 0:
        fill = mask[-1] if mask.size else 1.0
        mask = np.concatenate([mask, np.full(pad, fill)])
    return mask[: nblocks * width].reshape(nblocks, width).mean(axis=1)


def _dilate_max(values, radius, periodic=False):
    """Running maximum over ±radius entries: spreads a coefficient's gap
    fraction over every neighbour whose filter support can reach it, so no
    gap-influenced coefficient reports qc == 0 (conservative by design).
    *periodic* wraps around the ends, as the transform of a whole record
    does."""
    if radius <= 0 or values.size == 0:
        return values
    out = values.copy()
    for s in range(1, int(radius) + 1):
        if periodic:
            np.maximum(out, np.roll(values, s), out=out)
            np.maximum(out, np.roll(values, -s), out=out)
        else:
            np.maximum(out[s:], values[:-s], out=out[s:])
            np.maximum(out[:-s], values[s:], out=out[:-s])
    return out


# --------------------------------------------------------------------------- #
# Coefficient store (zarr)
# --------------------------------------------------------------------------- #
def _create_array(group, name, dtype, chunks=_ZARR_CHUNK):
    """Create an empty, appendable 1D array (zarr 2/3 compatible)."""
    kwargs = dict(shape=(0,), chunks=(int(chunks),), dtype=dtype)
    if name in group:
        del group[name]
    if hasattr(group, 'create_array'):    # zarr >= 3
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)  # zarr 2


class CascadeStore:
    """
    Zarr-backed store of DWT coefficients: one group per variable, one 1D
    array per level.

    Layout::

        <path>/                 attrs: wavelet, mode, fs, split_level,
            |                          n_samples, chunk_size, margin, t0,
            |                          details_dtype, variables, level
            <var>/              attrs: n_done (samples processed so far)
                cD01 .. cD<split_level>     detail coefficients (stage 1)
                qc01 .. qc<split_level>     gap fraction per coefficient
                carry, qc_carry             cA_split_level (stage-1 output)
                cD<split_level+1> .. cD<level>, cA, qc.., qc_cA   (stage 2)

    Use :meth:`create` / :meth:`open`; process with :func:`cascade_run` and
    :func:`cascade_finish`.
    """

    def __init__(self, root):
        self.root = root

    @classmethod
    def create(cls, path, variables, n_samples, fs, wavelet=DEFAULT_WAVELET,
               split_level=9, chunk_size=None, margin=None, t0=None,
               details_dtype='float32', overwrite=False):
        """
        function: initialise an empty coefficient store
        call: CascadeStore.create()
        Input:
            path: directory for the zarr store
            variables: list of unique variable names (e.g. ['w', 'co2']), or
                a single name
            n_samples: record length on the nominal sampling grid; must be a
                positive multiple of ``2**split_level`` (choose the datetime
                range, or pad, accordingly)
            fs: sampling rate (Hz)
            wavelet: mother wavelet (comprehensible to pywt)
            split_level: stage-1 depth; scale ``2**split_level / fs`` seconds
            chunk_size: stage-1 chunk (samples); multiple of 2**split_level
            margin: overlap per side (samples); default cascade_margin()
            t0: timestamp of the first sample (anything np.datetime64 accepts)
            details_dtype: storage dtype of detail coefficients ('float32'
                halves the archive; use 'float64' for bit-exact round-trips).
                The carry is always stored in float64 since it feeds all
                deeper levels.
            overwrite: replace an existing store
        Return:
            store: CascadeStore
        """
        _require_zarr()
        if isinstance(variables, str):
            variables = [variables]
        variables = list(variables)
        if not variables or len(set(variables)) != len(variables):
            raise ValueError(
                f"variables must be unique names; got {variables}.")
        split_level = int(split_level)
        if split_level < 0:
            raise ValueError(
                f"split_level ({split_level}) must not be negative.")
        step = 2 ** split_level
        n_samples = int(n_samples)
        if n_samples <= 0 or n_samples % step:
            raise ValueError(
                f"n_samples ({n_samples}) must be a positive multiple of "
                f"2**split_level ({step}); pad the record once, globally.")
        if not fs > 0:
            raise ValueError(f"fs ({fs}) must be positive.")
        if chunk_size is None:
            chunk_size = max(step, 2 ** 21)
        chunk_size = int(chunk_size)
        if chunk_size <= 0 or chunk_size % step:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be a positive multiple of "
                f"2**split_level ({step}).")
        required = cascade_margin(split_level, wavelet)
        if margin is None:
            margin = required
        margin = int(-(-int(margin) // step) * step)  # round up to a multiple
        if margin < required:
            # an undersized margin silently contaminates the deep stage-1
            # levels while the shallow ones stay exact -- never allow it
            raise ValueError(
                f"margin ({margin}) is below the exactness bound "
                f"cascade_margin({split_level}, '{wavelet}') = {required}; "
                "omit margin to use the safe default.")
        if t0 is None:
            logger.warning(
                "CascadeStore.create: t0 not given; TIMESTAMP coordinates of "
                "all products will be float seconds since the record start.")

        root = zarr.open_group(path, mode='w' if overwrite else 'w-')
        root.attrs.update({
            'waveletec_cascade': 1,
            'wavelet': str(wavelet),
            'mode': MODE,
            'fs': float(fs),
            'split_level': split_level,
            'n_samples': n_samples,
            'chunk_size': chunk_size,
            'margin': margin,
            't0': (np.datetime_as_string(np.datetime64(t0))
                   if t0 is not None else None),
            'details_dtype': str(details_dtype),
            'variables': [str(v) for v in variables],
            'level': None,
        })
        for var in variables:
            g = root.create_group(var)
            g.attrs['n_done'] = 0
            for j in range(1, split_level + 1):
                _create_array(g, f'cD{j:02d}', details_dtype)
                _create_array(g, f'qc{j:02d}', 'float32')
            _create_array(g, 'carry', 'float64')
            _create_array(g, 'qc_carry', 'float32')
        return cls(root)

    @classmethod
    def open(cls, path, mode='r+'):
        _require_zarr()
        root = zarr.open_group(path, mode=mode)
        if not root.attrs.get('waveletec_cascade'):
            raise ValueError(f"{path} is not a waveletec cascade store.")
        return cls(root)

    @property
    def attrs(self):
        # through a fresh handle: zarr keeps the attributes it has loaded,
        # and another handle may have finished the store since
        root = zarr.open_group(self.root.store, mode='r', path=self.root.path)
        return dict(root.attrs)

    @property
    def variables(self):
        return list(self.root.attrs['variables'])

    def __getitem__(self, var):
        return self.root[var]

    def __repr__(self):
        a = self.attrs
        return (f"CascadeStore(variables={a['variables']}, "
                f"n_samples={a['n_samples']}, fs={a['fs']}, "
                f"wavelet='{a['wavelet']}', split_level={a['split_level']}, "
                f"level={a['level']})")


def _as_store(store, mode='r+'):
    if isinstance(store, CascadeStore):
        return store
    return CascadeStore.open(store, mode=mode)


# --------------------------------------------------------------------------- #
# Stage 1: chunked decomposition into the store
# --------------------------------------------------------------------------- #
def _reconcile(group, split_level, done):
    """Truncate stage-1 arrays to the lengths implied by the committed
    ``n_done``. The appends of a chunk land before the ``n_done`` commit, so
    a process that died mid-chunk leaves some arrays longer than ``n_done``
    implies; discarding that partial chunk makes the resume exact."""
    names = ([(f'cD{j:02d}', j) for j in range(1, split_level + 1)]
             + [(f'qc{j:02d}', j) for j in range(1, split_level + 1)]
             + [('carry', split_level), ('qc_carry', split_level)])
    for name, shift in names:
        arr = group[name]
        expected = done >> shift
        if arr.shape[0] > expected:
            logger.warning(
                "cascade_run: discarding %d partially-written value(s) of "
                "'%s' left by an interrupted run.",
                arr.shape[0] - expected, name)
            arr.resize((expected,))
        elif arr.shape[0] < expected:
            raise ValueError(
                f"Store is inconsistent: '{name}' holds {arr.shape[0]} "
                f"values but n_done={done} implies {expected}. "
                "Recreate the store.")


def cascade_run(store, sources):
    """
    function: stage 1 of the cascade. Walks the record chunk by chunk (with
        overlap margins of real data, wrapping at the record ends as
        ``mode='periodization'`` does globally), gap-fills each padded chunk,
        decomposes it to ``split_level``, trims, and appends coefficients and
        gap-fraction QC to the store. Resumable: already-processed chunks
        (tracked per variable via the ``n_done`` attribute) are skipped and a
        chunk half-written by an interrupted process is discarded on
        re-entry, so interrupted runs can simply be re-launched. Run at most
        ONE writer per variable at a time (parallelise across variables or
        stores, not within one variable).
    call: cascade_run()
    Input:
        store: CascadeStore (or path to one)
        sources: mapping {variable: source}; a source is either a 1D
            array-like on the nominal sampling grid (numpy, memmap, dask,
            xarray, pandas) or a callable ``f(i0, i1) -> array`` returning
            samples [i0, i1); its length must equal the store's
            ``n_samples``, and a read of any other length is an error.
            Gaps are non-finite samples: whatever their length, they are
            linearly interpolated between the finite samples around them
            (the record being periodic, as for the transform) and reported
            in the per-level ``qc`` arrays as the gap fraction under each
            coefficient, dilated by the filter reach so every
            gap-influenced coefficient has qc > 0 (conservative). A record
            with no finite sample at all is transformed as zeros.
    Return:
        store: CascadeStore
    """
    store = _as_store(store)
    a = store.attrs
    n, chunk, margin = a['n_samples'], a['chunk_size'], a['margin']
    split, wavelet = a['split_level'], a['wavelet']
    reach = pywt.Wavelet(wavelet).dec_len - 1   # qc dilation radius [blocks]

    unknown = set(sources) - set(store.variables)
    if unknown:
        raise ValueError(
            f"Variable(s) {sorted(unknown)} not in store {store.variables}.")

    for var, source in sources.items():
        if hasattr(source, '__len__') and len(source) != n:
            raise ValueError(
                f"Source for '{var}' has {len(source)} samples, "
                f"store expects {n}.")
        g = store[var]
        done = int(g.attrs['n_done'])
        _reconcile(g, split, done)
        if done >= n:
            logger.info("cascade_run: '%s' already complete, skipping.", var)
            continue
        known_gaps = []
        for s in range(done, n, chunk):
            stop = min(s + chunk, n)
            raw = _read_wrapped(source, s - margin, stop + margin, n)
            filled, mask = _fill_window(raw, s - margin, source, n, chunk,
                                        known_gaps)
            details, carry = chunk_wavedec(filled, margin, split, wavelet)
            # QC from the padded mask (so influence crossing chunk seams is
            # seen), dilated by the filter reach, then trimmed like the
            # coefficients themselves.
            for j in range(1, split + 1):
                g[f'cD{j:02d}'].append(details[j])
                qc = _dilate_max(_block_fraction(mask, 2 ** j), reach)
                p = margin >> j
                g[f'qc{j:02d}'].append(
                    qc[p: p + ((stop - s) >> j)].astype('float32'))
            g['carry'].append(carry)
            qc = _dilate_max(_block_fraction(mask, 2 ** split), reach)
            p = margin >> split
            g['qc_carry'].append(
                qc[p: p + ((stop - s) >> split)].astype('float32'))
            g.attrs['n_done'] = stop
            logger.debug("cascade_run: '%s' %d/%d samples.", var, stop, n)
        logger.info("cascade_run: '%s' done (%d samples).", var, n)
    return store


# --------------------------------------------------------------------------- #
# Stage 2: finish the cascade from the carried approximation
# --------------------------------------------------------------------------- #
def cascade_finish(store, level=None):
    """
    function: stage 2 of the cascade. Loads the carried approximation
        (``n_samples / 2**split_level`` values -- small even for multi-year
        records), decomposes it in memory down to *level*, and writes the
        deep detail coefficients plus the final approximation ``cA`` to the
        store. Idempotent: re-running replaces the stage-2 arrays, so it can
        be repeated cheaply to change *level*. A refused call (bad *level*,
        incomplete variable) leaves the store as it was.
    call: cascade_finish()
    Input:
        store: CascadeStore (or path to one)
        level: final decomposition level; default pywt.dwt_max_level
    Return:
        store: CascadeStore
    """
    store = _as_store(store)
    a = store.attrs
    n, split, wavelet = a['n_samples'], a['split_level'], a['wavelet']
    dtype = a['details_dtype']
    reach = pywt.Wavelet(wavelet).dec_len - 1

    # refuse before touching anything: a rejected call leaves a finished
    # store finished
    for var in store.variables:
        g = store[var]
        if int(g.attrs['n_done']) != n:
            raise ValueError(
                f"Variable '{var}' has {g.attrs['n_done']}/{n} samples; "
                "run cascade_run to completion before cascade_finish.")
        _reconcile(g, split, n)
    carry_size = n >> split
    max_extra = pywt.dwt_max_level(carry_size, wavelet)
    if level is None:
        level_ = split + max_extra
    else:
        level_ = int(level)
        if level_ <= split:
            raise ValueError(
                f"level ({level_}) must exceed split_level ({split}).")
        if level_ - split > max_extra:
            logger.warning(
                "Level %d exceeds dwt_max_level (%d) for the carried "
                "approximation; deepest levels are boundary-dominated.",
                level_, split + max_extra)
    if carry_size > 2 ** 27:
        logger.warning(
            "Carried approximation holds %d samples (%.1f GB in memory); "
            "consider a deeper split_level for records this long.",
            carry_size, carry_size * 8 / 2 ** 30)
    # parity of the halving chain: beyond the first odd length, pywt pads
    # and the transform becomes redundant (still equal to a full
    # pywt.wavedec, but no longer an orthonormal partition)
    even_depth = (carry_size & -carry_size).bit_length() - 1
    if level_ - split > even_depth:
        logger.warning(
            "cascade_finish: the halving chain hits an odd length after %d "
            "level(s); coefficients beyond level %d remain a valid pywt DWT "
            "but the per-scale covariance partition becomes approximate "
            "there. For an exact partition choose n_samples divisible by "
            "2**level.", even_depth, split + even_depth)

    # invalidate first: while stage-2 arrays are being rewritten, readers
    # fall back to the always-intact carry instead of trusting stale labels
    store.root.attrs['level'] = None

    for var in store.variables:
        g = store[var]
        carry = g['carry'][:]
        qc_carry = g['qc_carry'][:].astype('float64')

        # decompose BEFORE touching the previous stage-2 arrays, so a
        # failure here loses nothing: re-running completes the store
        coeffs = pywt.wavedec(carry, wavelet, level=level_ - split, mode=MODE)
        # coeffs = [cA_level, cD_level, ..., cD_split+1] (levels of the record)

        # drop stage-2 arrays from a previous (possibly deeper) finish
        for name in list(g.keys()):
            deep_detail = (name.startswith(('cD', 'qc'))
                           and name[2:].isdigit() and int(name[2:]) > split)
            if deep_detail or name in ('cA', 'qc_cA'):
                del g[name]

        _create_array(g, 'cA', 'float64')
        g['cA'].append(coeffs[0])
        _create_array(g, 'qc_cA', 'float32')
        g['qc_cA'].append(_dilate_max(_block_fraction(
            qc_carry, 2 ** (level_ - split), coeffs[0].size),
            reach, periodic=True).astype('float32'))
        for i, j in enumerate(range(level_, split, -1), start=1):
            _create_array(g, f'cD{j:02d}', dtype)
            g[f'cD{j:02d}'].append(coeffs[i])
            _create_array(g, f'qc{j:02d}', 'float32')
            g[f'qc{j:02d}'].append(_dilate_max(_block_fraction(
                qc_carry, 2 ** (j - split), coeffs[i].size),
                reach, periodic=True).astype('float32'))
        logger.info("cascade_finish: '%s' decomposed to level %d.",
                    var, level_)

    # commit last: 'level' only ever describes fully-written arrays
    store.root.attrs['level'] = level_
    return store


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #
def _maybe_dask(array):
    """Wrap a zarr array in dask when it is large; read small ones eagerly."""
    if array.shape[0] > _TIME_COORD_MAX:
        import dask.array as da
        return da.from_array(array, chunks=array.chunks)
    return array[:]


def _time_coord(attrs, level_step, count, offset=0):
    """Timestamps (or seconds since start) of coefficient/window centers
    spaced *level_step* samples, starting *offset* samples into the record."""
    seconds = (offset + (np.arange(count) + 0.5) * level_step) / attrs['fs']
    if attrs.get('t0') is None:
        return seconds
    t0 = np.datetime64(attrs['t0'], 'ns')
    return t0 + (seconds * 1e9).astype('timedelta64[ns]')


def open_level(store, var, level=None, qc=False):
    """
    function: read one level of coefficients back as an xarray.DataArray on
        its own (decimated) time axis. Large levels (> ~16M coefficients)
        are returned dask-backed and without a TIMESTAMP coordinate.
        With ``t0`` unset at create time, TIMESTAMP is float seconds since
        the record start rather than datetime64.
    call: open_level()
    Input:
        store: CascadeStore (or path to one)
        var: variable name
        level: detail level (1..final, where final is ``split_level`` until
            cascade_finish has run); None for the approximation -- the final
            cA when the store is finished, else the stage-1 carry
        qc: return the gap-fraction QC array instead of the coefficients
    Return:
        da: xarray.DataArray; details carry attrs natural_frequency =
            fs * 2**-level (Hz), the approximation ('<var>_lf') is labelled
            fs * 2**-(level+1) -- the slot universal_wt reserves for it
    """
    store = _as_store(store, mode='r')
    a = store.attrs
    final = a['level'] if a['level'] is not None else a['split_level']
    if level is None:
        if a['level'] is None:
            name, j = 'qc_carry' if qc else 'carry', a['split_level']
        else:
            name, j = 'qc_cA' if qc else 'cA', a['level']
        out_name = f'{var}_lf_qc' if qc else f'{var}_lf'
        freq = a['fs'] * 2.0 ** -(j + 1)     # the approximation's own slot
    else:
        j = int(level)
        if not 1 <= j <= final:
            raise ValueError(
                f"Level {j} is not available (store holds 1..{final}); "
                "deeper levels require cascade_finish(level >= "
                f"{j}).")
        name = f'{"qc" if qc else "cD"}{j:02d}'
        out_name = f'{var}_{name}'
        freq = a['fs'] * 2.0 ** -j
    arr = store[var][name]
    values = _maybe_dask(arr)
    coords = {}
    if arr.shape[0] <= _TIME_COORD_MAX:
        coords['TIMESTAMP'] = ('TIMESTAMP', _time_coord(a, 2 ** j, arr.shape[0]))
    out = xr.DataArray(values, dims=('TIMESTAMP',), coords=coords,
                       name=out_name)
    out.attrs.update({'level': j, 'natural_frequency': freq,
                      'array': name,
                      'wavelet': a['wavelet'], 'mode': a['mode']})
    return out


# --------------------------------------------------------------------------- #
# Products in coefficient space
# --------------------------------------------------------------------------- #
# windows longer than this many coefficients are reduced through the
# cumulative-integral path instead of a row-contiguous reshape, keeping
# dask task memory bounded for climatological windows
_RESHAPE_CAP = 2 ** 22


def _window_mean(values, level_step, window, nwin, n_samples):
    """Window means of the per-sample density of a per-coefficient series.

    Each coefficient of a level with *level_step* samples per coefficient
    spreads its value uniformly over its samples (the last coefficient
    overlapping the record end over what remains of it, so ceil-padded deep
    levels spread no mass beyond the record); windows of *window* samples
    then average that density. Exact for any window/level combination.
    Dask inputs are reduced lazily in both paths -- only per-window sums
    and a handful of edge values are ever materialised.
    """
    if window % level_step == 0 and window // level_step <= _RESHAPE_CAP:
        c = window // level_step
        v = values[: nwin * c]
        if hasattr(v, 'rechunk'):  # dask: align chunks to window boundaries
            v = v.rechunk((c * max(1, _RESHAPE_CAP // c),))
        out = np.asarray(v.reshape((nwin, c)).sum(axis=1), dtype='float64')
        return out / window
    # general path: cumulative integral of the density, evaluated at the
    # window edges (a lazy cumsum plus nwin+1 gathered values for dask)
    k_real = max(1, min(int(values.shape[0]), -(-n_samples // level_step)))
    v = values[:k_real].astype('float64')
    if hasattr(v, 'rechunk'):
        import dask.array as da
        cum = da.cumsum(v)
    else:
        v = np.asarray(v)
        cum = np.cumsum(v)
    edges = np.arange(nwin + 1, dtype='float64') * float(window)
    ki = np.minimum((edges // level_step).astype('int64'), k_real - 1)
    width = np.full(nwin + 1, float(level_step))
    last_width = float(n_samples - (k_real - 1) * level_step)
    width[ki == k_real - 1] = min(float(level_step), max(last_width, 1.0))
    frac = np.clip((edges - ki * level_step) / width, 0.0, 1.0)
    base = np.where(ki > 0, np.asarray(cum[np.maximum(ki - 1, 0)]), 0.0)
    s_edges = base + frac * np.asarray(v[ki])
    return np.diff(s_edges) / window


def scale_covariance(store, x, y, window=1800, levels=None):
    """
    function: per-scale covariance time series computed directly in
        coefficient space (no reconstruction): for each level and averaging
        window, ``sum(cD_x * cD_y) / n_window_samples``, coefficients that
        straddle a window edge being attributed fractionally. Summing the
        output over all levels plus the ``_lf`` term recovers, over the whole
        record, ``mean(x * y)`` exactly when ``n_samples`` is divisible by
        ``2**level`` (orthonormal periodized DWT); once the halving chain
        hits an odd length the deepest levels are redundant and the
        partition becomes approximate there (cascade_finish warns). The
        difference sits in the ~10 coefficients per level at the record
        ends, interior coefficients being unaffected, but it is not small:
        0.2 to 11% of the whole-record covariance on the 3-day FR-Gri
        sample. Per window, levels whose scale nears the window are only a
        coarse attribution of the covariance (a coefficient is spread evenly
        over its block), the whole-record means staying exact. Use demeaned
        or turbulent-fluctuation inputs for flux applications.
    call: scale_covariance()
    Input:
        store: CascadeStore (or path to one)
        x, y: variable names (y may equal x for a variance spectrum)
        window: averaging window in seconds (default 1800 = 30 min), or a
            pandas-style duration string such as '30min'
        levels: detail levels to include (default: all available)
    Return:
        ds: xarray.Dataset with
            '<x><y>'      (natural_frequency, TIMESTAMP) covariance per scale
            '<x><y>_lf'   (TIMESTAMP,) final-approximation (lowest frequency)
            '<x><y>_qc'   (natural_frequency, TIMESTAMP) gap fraction (max of
                          the two variables)
            '<x><y>_lf_qc' (TIMESTAMP,)
    """
    store = _as_store(store, mode='r')
    a = store.attrs
    n, fs = a['n_samples'], a['fs']
    final = a['level'] if a['level'] is not None else a['split_level']
    if levels is None:
        levels = list(range(1, final + 1))
    if not len(levels) or not all(1 <= j <= final for j in levels):
        raise ValueError(
            f"levels {list(levels)} are not available (store holds "
            f"1..{final}).")
    if isinstance(window, str):
        import pandas as pd
        window = pd.to_timedelta(window).total_seconds()
    w = int(round(window * fs))
    if not 0 < w <= n:
        raise ValueError(
            f"window ({window} s, {w} samples) must span at least one sample "
            f"and at most the {n}-sample record.")
    nwin = n // w
    if n % w:
        logger.warning(
            "Record (%d samples) is not a whole number of %d-sample windows; "
            "the tail is dropped.", n, w)

    gx, gy = store[x], store[y]
    for var, g in ((x, gx), (y, gy)):
        if int(g.attrs['n_done']) != n:
            raise ValueError(f"Variable '{var}' is not fully decomposed yet.")
    cov_rows, qc_rows = [], []
    for j in levels:
        name = f'cD{j:02d}'
        px = _maybe_dask(gx[name]).astype('float64')
        py = _maybe_dask(gy[name]).astype('float64')
        cov_rows.append(_window_mean(px * py, 2 ** j, w, nwin, n))
        qx = _maybe_dask(gx[f'qc{j:02d}']).astype('float64')
        qy = _maybe_dask(gy[f'qc{j:02d}']).astype('float64')
        qc_rows.append(_window_mean(
            np.maximum(qx, qy) * 2 ** j, 2 ** j, w, nwin, n))

    lf_name, lf_qc_name = ('cA', 'qc_cA') if a['level'] is not None \
        else ('carry', 'qc_carry')
    step = 2 ** final
    lfx = gx[lf_name][:].astype('float64')
    lfy = gy[lf_name][:].astype('float64')
    lf = _window_mean(lfx * lfy, step, w, nwin, n)
    lf_qc = _window_mean(np.maximum(
        gx[lf_qc_name][:], gy[lf_qc_name][:]).astype('float64') * step,
        step, w, nwin, n)

    xy = f'{x}{y}'
    timestamps = _time_coord(a, w, nwin, offset=-w / 2)  # window starts
    freqs = [fs * 2.0 ** -j for j in levels]
    ds = xr.Dataset(
        {
            xy: (('natural_frequency', 'TIMESTAMP'), np.vstack(cov_rows)),
            f'{xy}_qc': (('natural_frequency', 'TIMESTAMP'),
                         np.vstack(qc_rows)),
            f'{xy}_lf': (('TIMESTAMP',), lf),
            f'{xy}_lf_qc': (('TIMESTAMP',), lf_qc),
        },
        coords={'natural_frequency': freqs, 'TIMESTAMP': timestamps},
    )
    ds.attrs.update({'method': 'dwt', 'wavelet': a['wavelet'],
                     'mode': a['mode'], 'fs': fs,
                     'averaging_window [s]': window,
                     'lf_level': final})
    return ds


# --------------------------------------------------------------------------- #
# On-demand reconstruction (stage-1 levels)
# --------------------------------------------------------------------------- #
def reconstruct(store, var, levels=None, start=0, stop=None):
    """
    function: per-level time series (as waverec_2d rows) for the sample
        window [start, stop), reconstructed on demand from stored
        coefficients with the same overlap-and-trim logic as the forward
        pass -- the full-length dense representation never exists. Limited
        to stage-1 levels (<= split_level): deeper levels have filter
        supports far longer than any reasonable window; read those as
        coefficients with open_level instead.
    call: reconstruct()
    Input:
        store: CascadeStore (or path to one)
        var: variable name
        levels: detail levels to reconstruct (default: 1..split_level)
        start, stop: sample range on the nominal grid (default whole record;
            mind the memory, each level is a full-resolution series)
    Return:
        da: xarray.DataArray (natural_frequency, TIMESTAMP), one row per level
    """
    store = _as_store(store, mode='r')
    a = store.attrs
    n, split, wavelet = a['n_samples'], a['split_level'], a['wavelet']
    if levels is None:
        levels = list(range(1, split + 1))
    if max(levels) > split or min(levels) < 1:
        raise ValueError(
            f"levels must be within 1..split_level ({split}); "
            "use open_level for deeper (stage-2) levels.")
    stop = n if stop is None else int(stop)
    start = int(start)
    if not 0 <= start < stop <= n:
        raise ValueError(f"Invalid sample range [{start}, {stop}) for n={n}.")
    if (stop - start) * len(levels) > _TIME_COORD_MAX:
        raise ValueError(
            f"Reconstructing {len(levels)} level(s) over {stop - start} "
            f"samples would materialise "
            f"{(stop - start) * len(levels) * 8 / 2 ** 30:.1f} GB; "
            "request a bounded [start, stop) window (fail-fast guard at "
            f"{_TIME_COORD_MAX} total samples).")
    g = store[var]
    if int(g.attrs['n_done']) != n:
        raise ValueError(f"Variable '{var}' is not fully decomposed yet.")

    lmax = max(levels)
    step = 2 ** lmax
    margin = cascade_margin(lmax, wavelet)
    a0 = ((start - margin) // step) * step
    b0 = -(-(stop + margin) // step) * step
    span = b0 - a0

    rows = []
    for j in levels:
        cd = _read_wrapped(g[f'cD{j:02d}'], a0 >> j, b0 >> j, n >> j)
        coeffs = ([np.zeros(span >> lmax)]
                  + [cd if i == j else np.zeros(span >> i)
                     for i in range(lmax, 0, -1)])
        rec = pywt.waverec(coeffs, wavelet, mode=MODE)
        # copy so the full padded-span buffer is freed level by level
        rows.append(rec[start - a0: stop - a0].copy())

    out = xr.DataArray(
        np.vstack(rows), dims=('natural_frequency', 'TIMESTAMP'),
        coords={'natural_frequency': [a['fs'] * 2.0 ** -j for j in levels],
                'TIMESTAMP': _time_coord(a, 1, stop - start,
                                         offset=start - 0.5)},
        name=var)
    out.attrs.update({'method': 'dwt', 'wavelet': wavelet, 'mode': a['mode'],
                      'fs': a['fs']})
    return out
