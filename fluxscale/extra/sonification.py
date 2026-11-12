"""Turn any variable into sound. Experimental: the API may still change.

Two levels, each a step further from the samples and closer to the ear:

- **L0, audio file.** The signal written as audio and nothing more. By default
  each data sample is one audio sample.
- **L1, audible.** The same signal altered for human hearing. Playing a record
  faster is not enough: speed and pitch are tied together, so squeezing half an
  hour of 20 Hz data into 30 s leaves most of its variance below the threshold
  of hearing. The wavelet transform unties them. Each scale is expanded into the
  hearing range on its own, then equalised and weighted for equal loudness.
  Every sound is still a piece of the data, and fine eddies stay above large
  ones.

A third level, music, is in FluxSound (``pip install fluxsound``), with its
instruments, its harmony and its MIDI export. It is built on what this module
makes public: ``load`` and ``wavelet_bands`` to get the scales of a variable,
the hearing curves, ``master`` and ``as_sound``.

Gaps in the data are silent. Over a long record, calm passages stay softer than
turbulent ones but within reach of the ear (see ``loudness_range``). Grains are
seeded, so what moves in the sound is what moved in the signal.
"""

# built-in modules
import logging
import os

# 3rd party modules
import numpy as np
import pandas as pd
import pywt
import xarray as xr
from scipy import signal as spsignal
from scipy.io import wavfile
from scipy.ndimage import (map_coordinates, maximum_filter1d, spline_filter1d,
                           uniform_filter1d)

# project modules
from ..core.wavelet_functions import formula_to_vars, prepare_signal, universal_wt

logger = logging.getLogger(__name__)

__all__ = ["sonify", "write_wav", "integrated_loudness", "load", "wavelet_bands", "band_names",
           "master", "as_sound"]

SAMPLE_RATE = 44100
DEFAULT_DURATION = 30.0      # s, L1 and the music of FluxSound

AUDIBLE_RANGE = (20, 20000)  # Hz
PEAK_CEILING = 10 ** (-1 / 20)  # -1 dBFS
SLOPE_KEPT = 0.2             # share of the spectral slope left by equalisation
MAX_BOOST_DB = 30            # equalisation never lifts a band further than this
LOUDNESS_WINDOW = 3.0        # s, the short-term window of BS.1770
CONTROL_RATE = 200           # Hz, audio time; rate of the loudness curves

# L1
HEARING_BAND = (100, 8000)   # Hz, where the scales of the data are expanded to
GRAIN_CYCLES = 4             # length of a grain, in cycles of its own band
GRAIN_OVERLAP = 4
GRAIN_CHUNK = 2 ** 22        # samples read at a time, to bound memory
MAX_MODULATION_HZ = 8        # below the ~15 Hz where fluctuation turns rough


# --------------------------------------------------------------------------- #
# Hearing
# --------------------------------------------------------------------------- #
def erb_rate(hz):
    """Hz to ERB-rate: the number of auditory filters below *hz* (Glasberg & Moore)."""
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(hz, dtype=float))


def erb_spaced(n, low, high):
    """*n* frequencies from *low* to *high* (Hz), evenly spaced to the ear."""
    rate = np.linspace(erb_rate(low), erb_rate(high), n)
    return (10 ** (rate / 21.4) - 1) / 0.00437


def equal_loudness_gain(hz):
    """Gain lifting *hz* halfway to equal loudness (half an inverse A-weighting).

    Half, because A-weighting describes quiet sounds; at listening level the
    ear's bass loss is roughly half as steep. Below 100 Hz the boost stops
    growing: there it would only feed speakers that cannot play it, and the
    overtones of a low voice already carry its pitch.
    """
    f2 = np.maximum(np.asarray(hz, dtype=float), 100) ** 2
    a_weight = (12194 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194 ** 2))
    return (a_weight * 10 ** (2 / 20)) ** -0.5


def typical_level(rows):
    """Typical size of each row, robust enough that a spike cannot set the scale."""
    return np.array([np.quantile(np.abs(row), 0.75) for row in rows])


def equalised(typical, kept):
    """Level of each band after equalisation, relative to the strongest band.

    Flattens the spectrum, leaving a fraction *kept* of its slope, but boosts no
    band by more than ``MAX_BOOST_DB``: a nearly empty band holds little more
    than leakage and sensor noise, which should stay in the background.
    """
    top = typical.max()
    if top == 0:
        return np.zeros_like(typical)
    ratio = typical / top
    return np.minimum(ratio ** kept, ratio * 10 ** (MAX_BOOST_DB / 20))


# --------------------------------------------------------------------------- #
# From the caller's data to a clean signal
# --------------------------------------------------------------------------- #
def parse_level(level):
    text = str(level).strip().upper().lstrip("L")
    if text not in {"0", "1", "2"}:
        raise ValueError(f"Unknown level {level!r}: use 0, 1 or 2.")
    return int(text)


def check_duration(duration):
    if duration is not None and duration <= 0:
        raise ValueError(f"`duration` must be positive, got {duration}.")


def _pick(data, name):
    """``data[name]`` from a Dataset, DataFrame or dict, or fail with the options."""
    if name is None:
        raise ValueError("`variable` is required when `data` holds several variables.")
    if name not in data:
        available = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data)
        raise ValueError(f"Variable {name!r} not found. Available: {available}.")
    return data[name]


def _infer_fs(series):
    """Sampling rate from a 1-D time coordinate, or ``None`` if there is none."""
    if not isinstance(series, xr.DataArray) or series.ndim != 1:
        return None
    coord = series.coords.get(series.dims[0])
    if coord is None or coord.size < 2:
        return None
    step = np.median(np.diff(coord.values))
    if np.issubdtype(coord.dtype, np.datetime64):
        step = step / np.timedelta64(1, "s")
    elif not np.issubdtype(coord.dtype, np.number):
        return None
    return 1 / float(step) if step > 0 else None


def _clean(series):
    """Gap-filled 1-D values of *series* and the mask of what was missing."""
    values = np.asarray(series, dtype=float).ravel()
    values = np.where(np.isfinite(values), values, np.nan)
    if not np.isfinite(values).any():
        raise ValueError("No finite values to sonify.")
    if values.size < 4:
        raise ValueError(f"Need at least 4 samples to sonify, got {values.size}.")
    ready = prepare_signal(xr.DataArray(values, dims="t"))
    return ready["signal"].values, ready["signan"].values.astype(bool)


def wavelet_bands(values, fs, wavelet):
    """Wavelet bands of *values*, finest first, and their frequencies (Hz)."""
    depth = pywt.dwt_max_level(values.size, wavelet)
    if depth < 2:  # too short for this filter; Haar needs only 2 samples
        wavelet = "haar"
        depth = pywt.dwt_max_level(values.size, wavelet)
    result = universal_wt(xr.DataArray(values, dims="t"), method="dwt", fs=fs,
                          wavelet=wavelet, level=depth, coi=False)
    return result["wave"].values, result["natural_frequency"].values


def band_names(freqs):
    """Name of each band, coarsest first, as stems and MIDI tracks show them."""
    return [f"{f:g} Hz" for f in freqs[::-1]]


def _load_product(data, variable, fs, wavelet):
    """A product as its cross-scalogram, as in ``data_compute_product``.

    That is the same-scale product of the levels: it sums over scales to the
    product of the signals over the record. It is not the cross-level ``wco2``
    of ``data_run`` (see ``handlers.data_compute_flux``).
    """
    names = formula_to_vars(variable).xy
    series = [_pick(data, name) for name in names]
    fs = fs or _infer_fs(series[0]) or 1.0
    cleaned = [_clean(s) for s in series]
    bands = None
    parts = []
    for name, (values, _) in zip(names, cleaned):
        decomposed, freqs = wavelet_bands(values, fs, wavelet)
        parts.append(dict(name=name, values=values,
                          power=np.einsum("ij,ij->j", decomposed, decomposed)))
        bands = decomposed if bands is None else bands * decomposed
    source = dict(values=bands.sum(axis=0), bands=bands, freqs=freqs,
                  gaps=np.any([g for _, g in cleaned], axis=0), fs=fs, name=variable,
                  signed=True)
    if len(names) == 2:
        source["parts"] = parts
    return source


def _load_wavelet_domain(series, fs, name):
    """A variable that is already decomposed (e.g. from ``data_run``)."""
    freqs = series["natural_frequency"].values
    if np.any(np.diff(freqs) >= 0):
        raise ValueError(
            "Wavelet-domain variables must come from the discrete transform "
            "(`method='dwt'`), whose bands run from fine to coarse.")
    values = series.transpose("natural_frequency", ...).values
    bands = values.reshape(freqs.size, -1).astype(float)
    gaps = np.isnan(bands).all(axis=0)
    if gaps.all():
        raise ValueError("No finite values to sonify.")
    bands = np.nan_to_num(bands)
    return dict(values=bands.sum(axis=0), bands=bands, freqs=freqs, gaps=gaps,
                fs=fs or 2 * float(freqs.max()), name=name, signed=True)


def load(data, variable, fs, wavelet):
    """Resolve what the caller passed.

    Returns:
        dict with ``values`` (1-D), ``gaps`` (bool, same length), ``fs``, ``name``
        and, when the input already implies them, ``bands`` (finest first) and
        ``freqs``. A product of two variables adds ``parts``.
    """
    if variable is not None and "*" in variable:
        return _load_product(data, variable, fs, wavelet)
    is_table = isinstance(data, (xr.Dataset, pd.DataFrame, dict))
    series = _pick(data, variable) if is_table else data
    if isinstance(series, pd.Series):
        series = xr.DataArray(series)
    name = variable or getattr(series, "name", None) or "signal"
    if isinstance(series, xr.DataArray) and "natural_frequency" in series.dims:
        return _load_wavelet_domain(series, fs, name)
    values, gaps = _clean(series)
    return dict(values=values, gaps=gaps, fs=fs or _infer_fs(series) or 1.0, name=name,
                signed=False)


# --------------------------------------------------------------------------- #
# Shared by L1 and the music of FluxSound
# --------------------------------------------------------------------------- #
def to_control(series, n_control, window):
    """Moving-average *series* over *window* samples, sampled at *n_control* points."""
    smooth = uniform_filter1d(series, max(int(round(window)), 1), mode="nearest")
    return np.interp(np.linspace(0, series.size - 1, n_control),
                     np.arange(series.size), smooth)


def rms_curve(power, n_control, window):
    """Root of a smoothed power (running sums can dip just below zero in a gap)."""
    return np.sqrt(np.maximum(to_control(power, n_control, window), 0))


def presence(gaps, n_control):
    """Curve that is 1 where there are data and 0 across gaps, without clicks."""
    present = to_control(1.0 - gaps, n_control, gaps.size / n_control)
    return uniform_filter1d(np.clip(4 * present - 2.5, 0, 1), 4, mode="nearest")


def pan_gains(pan):
    """Constant-power left and right gains for *pan* in -1..1."""
    angle = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.cos(angle), np.sin(angle)


# --------------------------------------------------------------------------- #
# L0: audio file
# --------------------------------------------------------------------------- #
def _audify(source, duration, sample_rate, parts=None):
    """The waveform itself: mean removed, peak-normalised, edges declicked.

    Returns:
        Stereo audio, the audible fraction, and the same treatment of each of
        *parts* (series that sum to the signal), scaled to sum to that audio.
    """
    values = source["values"] - source["values"].mean()
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    audio_hz = np.arange(spectrum.size) / duration  # k cycles per record
    total = spectrum[1:].sum()
    audible = spectrum[(audio_hz >= AUDIBLE_RANGE[0]) & (audio_hz <= AUDIBLE_RANGE[1])].sum()
    fraction = float(audible / total) if total > 0 else 0.0
    if fraction < 0.5:
        logger.warning(
            "Only %.0f%% of the variance is audible at this speed; try level=1 "
            "(or level 2 of FluxSound), or a shorter duration.", 100 * fraction)
    if duration < 0.25:
        logger.warning("This record lasts only %.0f ms as audio; pass `duration` "
                       "to slow it down, or use level=1.", 1000 * duration)

    n_out = int(round(duration * sample_rate))
    taper = spsignal.windows.tukey(values.size, min(1.0, 0.04 / duration))
    present = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, values.size),
                        1.0 - source["gaps"])
    present = uniform_filter1d(present, int(0.005 * sample_rate) or 1, mode="nearest")

    def as_audio(series):
        wave = series * taper
        if n_out != series.size:
            wave = spsignal.resample(wave, n_out)
        return wave * present

    wave = as_audio(values)
    peak = np.abs(wave).max()
    scale = PEAK_CEILING / peak if peak > 0 else 1.0
    stereo = [np.column_stack([as_audio(p) * scale] * 2) for p in parts] if parts is not None else None
    return np.column_stack([wave * scale] * 2), fraction, stereo


# --------------------------------------------------------------------------- #
# L1: audible
# --------------------------------------------------------------------------- #
def _expand_band(band, cycles_per_sample, target, neighbourhood, n_out, sample_rate, rng):
    """Move one wavelet band up to *target* Hz without moving it in time.

    Short grains of the band are read fast enough to sound at *target*, while
    the place they are read from advances at the speed of the record, so pitch
    no longer depends on speed. A grain lasts a few cycles of its band, as a
    wavelet does. Grains come from around the current place, at least a period
    apart so that they do not repeat in phase and buzz, and within
    *neighbourhood* samples so that loudness cannot change faster than that.

    Returns:
        Stereo array of shape ``(n_out, 2)``, grains spread across the field.
    """
    step = target / sample_rate / cycles_per_sample  # data samples per audio sample
    hop = max(int(GRAIN_CYCLES * sample_rate / target / GRAIN_OVERLAP), 1)
    length = hop * GRAIN_OVERLAP
    n_grains = -(-n_out // hop) + GRAIN_OVERLAP
    places = ((np.arange(n_grains) - GRAIN_OVERLAP) * hop + length / 2) / n_out * band.size
    reach = max(1 / cycles_per_sample, neighbourhood)
    places = places + rng.uniform(-reach, reach, n_grains)
    left, right = pan_gains(rng.uniform(-0.7, 0.7, n_grains))

    window = spsignal.windows.hann(length, sym=False)
    offsets = (np.arange(length) - length / 2) * step
    spline = spline_filter1d(band, order=3, mode="mirror")
    rows = np.zeros((2, n_grains + GRAIN_OVERLAP, hop))
    chunk = max(GRAIN_CHUNK // length, 1)
    for first in range(0, n_grains, chunk):
        last = min(first + chunk, n_grains)
        reads = (places[first:last, None] + offsets).ravel()[None]
        grains = map_coordinates(spline, reads, order=3, mode="mirror", prefilter=False)
        grains = grains.reshape(last - first, length) * window
        for q in range(GRAIN_OVERLAP):  # overlap-add, one quarter of a grain at a time
            part = grains[:, q * hop:(q + 1) * hop]
            rows[0, first + q:last + q] += part * left[first:last, None]
            rows[1, first + q:last + q] += part * right[first:last, None]
    skip = GRAIN_OVERLAP * hop
    return rows.reshape(2, -1)[:, skip:skip + n_out].T


def _render_audible(bands, freqs, source, duration, sample_rate, rng):
    """L1: expand the bands over the hearing range, equalise them, weight them.

    Returns:
        The part of each band, coarsest first, and the frequency it was given.
    """
    bands, freqs = bands[::-1], freqs[::-1]  # coarsest first, like the pitches
    targets = erb_spaced(bands.shape[0], *HEARING_BAND)
    n_out = int(round(duration * sample_rate))
    typical = typical_level(bands)
    levels = equalised(typical, SLOPE_KEPT)
    neighbourhood = bands.shape[1] / duration / MAX_MODULATION_HZ / 2

    n_control = max(int(duration * CONTROL_RATE), 2)
    gate = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, n_control),
                     presence(source["gaps"], n_control))[:, None]
    parts = []
    for band, freq, target, size, level in zip(bands, freqs, targets, typical, levels):
        if size == 0:
            parts.append(np.zeros((n_out, 2)))
            continue
        softened = np.tanh(band / (5 * size))  # spikes saturate, the rest is untouched
        # a DWT band spans freq/2..freq, so its centre is 0.75 * freq
        expanded = _expand_band(softened, 0.75 * freq / source["fs"], target, neighbourhood,
                                n_out, sample_rate, rng)
        part = expanded * (level * equal_loudness_gain(target) / expanded.std())
        parts.append((part - part.mean(axis=0)) * gate)
    return parts, targets


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
def _k_weighting(sample_rate):
    """The two BS.1770 pre-filters, for any sample rate, as second-order sections.

    A high shelf for the head, then a high-pass for low-frequency insensitivity.
    These analogue parameters reproduce the 48 kHz coefficients of the standard.
    """
    k = np.tan(np.pi * 1681.9744509555319 / sample_rate)
    q = 0.7071752369554193
    high = 10 ** (3.99984385397 / 20)
    mid = high ** 0.499666774155
    shelf = [high + mid * k / q + k * k, 2 * (k * k - high), high - mid * k / q + k * k,
             1 + k / q + k * k, 2 * (k * k - 1), 1 - k / q + k * k]
    k = np.tan(np.pi * 38.13547087613982 / sample_rate)
    q = 0.5003270373253953
    scale = 1 + k / q + k * k
    high_pass = [scale, -2 * scale, scale, scale, 2 * (k * k - 1), 1 - k / q + k * k]
    sections = np.array([shelf, high_pass])
    return sections / sections[:, 3:4]


def _weighted_power(audio, sample_rate):
    """K-weighted power of every sample, summed over channels."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim == 1:
        audio = audio[:, None]
    return (spsignal.sosfilt(_k_weighting(sample_rate), audio, axis=0) ** 2).sum(axis=1)


def integrated_loudness(audio, sample_rate=SAMPLE_RATE):
    """Integrated programme loudness (ITU-R BS.1770-4), in LUFS.

    Args:
        audio: Array of shape ``(samples,)`` or ``(samples, channels)``.
        sample_rate: Sample rate in Hz.

    Returns:
        Loudness in LUFS; ``-inf`` for silence.
    """
    power = _weighted_power(audio, sample_rate)

    def lufs(mean_square):
        with np.errstate(divide="ignore"):
            return -0.691 + 10 * np.log10(mean_square)

    block = int(0.4 * sample_rate)
    if power.size < block:
        return float(lufs(power.mean()))
    running = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, power.size - block + 1, block // 4)
    blocks = (running[starts + block] - running[starts]) / block
    blocks = blocks[lufs(blocks) > -70]  # absolute gate
    if blocks.size == 0:
        return -np.inf
    blocks = blocks[lufs(blocks) > lufs(blocks.mean()) - 10]  # relative gate
    return float(lufs(blocks.mean()))


def _limit_gain(audio, sample_rate):
    """Look-ahead limiter: turns peaks down smoothly instead of clipping them."""
    reach = 2 * int(0.01 * sample_rate) + 1
    held = maximum_filter1d(np.abs(audio).max(axis=1), reach, mode="nearest")
    gain = PEAK_CEILING / np.maximum(held, PEAK_CEILING)
    return uniform_filter1d(gain, reach, mode="nearest")


def _swing_gain(audio, sample_rate, n_record, widest):
    """Gain that brings the slow swings of loudness within *widest* LU, keeping their order.

    A long record can hold a calm night and a turbulent noon, so far apart that
    the night is lost when the noon sounds right. Soft passages stay softer than
    loud ones, only by less, and notes keep their own dynamics because the gain
    moves over seconds.

    Args:
        audio: Stereo array.
        sample_rate: Sample rate in Hz.
        n_record: Samples that carry the record. Whatever follows, such as the
            last notes ringing out, has no say and takes the last gain.
        widest: Largest swing in LU, from the 5th to the 95th percentile.

    Returns:
        Gain per sample, or ``None`` if the record already fits.
    """
    window = int(LOUDNESS_WINDOW * sample_rate)
    if n_record < window:  # too short to have slow swings
        return None
    hop = max(window // 30, 1)
    power = _weighted_power(audio[:n_record], sample_rate)
    blocks = power[:power.size // hop * hop].reshape(-1, hop).mean(axis=1)
    if not blocks.any():
        return None
    # A gap has no loudness to keep, so windows average only the sound beside it.
    sounding = blocks > 1e-6 * blocks.max()
    share = uniform_filter1d(sounding.astype(float), 30, mode="nearest")
    known = share > 0.02
    mean = uniform_filter1d(blocks * sounding, 30, mode="nearest")
    level = 10 * np.log10(np.maximum(mean[known], 1e-30) / share[known])
    low, high = np.quantile(level, [0.05, 0.95])
    if high - low <= widest:
        return None
    gain = (np.clip(level, low, high) - (low + high) / 2) * (widest / (high - low) - 1)
    gain = uniform_filter1d(gain, 30, mode="nearest")
    gain = np.interp(np.arange(audio.shape[0]), (np.flatnonzero(known) + 0.5) * hop, gain)
    return 10 ** (gain / 20)


def master(audio, sample_rate, target, n_record, widest):
    """Narrow slow swings, fade the edges, reach *target* and keep the peaks legal.

    Returns:
        The mastered audio and the gain per sample that made it. Every step is a
        gain, so parts of the mix can take the same curve and still sum to it.
    """
    gain = np.ones(audio.shape[0])
    if widest is not None:
        swing = _swing_gain(audio, sample_rate, n_record, widest)
        if swing is not None:
            gain = swing
    fade = min(int(1.5 * sample_rate), audio.shape[0] // 4)
    if fade:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        gain[:fade] *= ramp
        gain[-fade:] *= ramp[::-1]
    for _ in range(3):  # limiting lowers the loudness a little, so converge
        measured = integrated_loudness(audio * gain[:, None], sample_rate)
        if not np.isfinite(measured):
            break
        gain = gain * 10 ** ((target - measured) / 20)
        gain = gain * _limit_gain(audio * gain[:, None], sample_rate)
    return audio * gain[:, None], gain


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def sonify(data, variable=None, level=1, duration=None, fs=None, loudness=-16.0,
           loudness_range=6.0, sample_rate=SAMPLE_RATE, wavelet="db6", seed=0, dst=None,
           stems=False):
    """Turn a variable into sound.

    Args:
        data: The signal: a 1-D array, ``pandas.Series`` or
            ``xarray.DataArray``; or a Dataset, DataFrame or dict holding
            *variable*. Multi-dimensional variables are read in time order, so
            ``(TIMESTAMP, ns)`` blocks play one after another. Variables with a
            ``natural_frequency`` dimension (``data_run`` output such as
            ``wco2`` or ``wco2-wh2o+``) are used as they are.
        variable: Name of the variable, or a product such as ``"w*co2"``.
        level: 0 audio file, 1 audible (the default; see the module docstring).
            Also accepts ``"L0"`` and ``"L1"``. Level 2, music, is in FluxSound.
        duration: Seconds of sound for the whole record. By default 30, or at
            level 0 one audio sample per data sample.
        fs: Sampling rate of the data (Hz). Only labels the output: the sound
            does not depend on it. Inferred from a time coordinate if possible.
        loudness: L1 target loudness in LUFS.
        loudness_range: L1 widest slow swing of loudness, in LU. Quiet passages
            stay quieter, but no further below the loud ones than this: 6 LU
            sounds about two thirds as loud. ``None`` keeps the full swing.
        sample_rate: Audio sample rate in Hz.
        wavelet: Mother wavelet for the decomposition.
        seed: Seed of the grains. The same data and seed always give the same
            sound.
        dst: Optional path of a ``.wav`` file to write (the full mix).
        stems: Return one track per wavelet scale instead of the mix, plus the
            trend at L0. They share one mastering, so they sum to the mix.

    Returns:
        xarray.DataArray with dims ``(time, channel)`` in the range -1..1, or
        ``(stem, time, channel)`` with *stems*.
        ``attrs`` records the mapping: ``level``, ``sample_rate``, ``variable``,
        ``fs`` and ``speedup``. L0 adds ``audible_fraction``, the share of the
        variance within hearing range. L1 adds ``band_frequency_hz`` and the
        ``voice_frequency_hz`` each band was given (coarsest first).

    Raises:
        ValueError: If the level or variable is unknown, or there are not enough
            finite values.
    """
    level = parse_level(level)
    if level == 2:
        raise ValueError("Level 2 (music) is in FluxSound: use fluxsound.sonify.")
    check_duration(duration)
    if loudness_range is not None and loudness_range <= 0:
        raise ValueError(f"`loudness_range` must be positive or None, got {loudness_range}.")

    source = load(data, variable, fs, wavelet)
    values = source["values"]
    if duration is None:
        duration = values.size / sample_rate if level == 0 else DEFAULT_DURATION
    rng = np.random.default_rng(seed)
    attrs = dict(level=level, sample_rate=sample_rate, variable=source["name"],
                 fs=source["fs"], speedup=values.size / source["fs"] / duration)

    parts = names = None
    if np.ptp(values) == 0:
        logger.warning("%s is constant: there is nothing to hear.", source["name"])
        audio = np.zeros((int(round(duration * sample_rate)), 2))
    elif level == 0:
        if stems:
            bands, freqs = wavelet_bands(values, source["fs"], wavelet)
            pieces = [values - bands.sum(axis=0) - values.mean()] + list(bands[::-1])
            names = ["trend"] + band_names(freqs)
        else:
            pieces = None
        audio, attrs["audible_fraction"], parts = _audify(source, duration, sample_rate, pieces)
    else:
        bands, freqs = wavelet_bands(values, source["fs"], wavelet)
        parts, voices = _render_audible(bands, freqs, source, duration, sample_rate, rng)
        names = band_names(freqs)
        audio, gain = master(np.sum(parts, axis=0), sample_rate, loudness,
                             int(round(duration * sample_rate)), loudness_range)
        parts = [part * gain[:, None] for part in parts]
        attrs["band_frequency_hz"] = freqs[::-1].tolist()
        attrs["voice_frequency_hz"] = np.asarray(voices).tolist()
    return as_sound(audio, source["name"], sample_rate, attrs, parts if stems else None, names, dst)


def as_sound(audio, name, sample_rate, attrs, parts=None, names=None, dst=None):
    """Audio as the ``xarray.DataArray`` that :func:`sonify` returns.

    Args:
        audio: The mix, of shape ``(samples, 2)``.
        name: Name of the variable.
        sample_rate: Audio sample rate in Hz.
        attrs: What to record of the mapping.
        parts: Stems of the shape of *audio*. If given, they are returned instead
            of the mix.
        names: Name of each of *parts*.
        dst: Optional path of a ``.wav`` file for the mix.

    Returns:
        xarray.DataArray with dims ``(time, channel)``, or ``(stem, time,
        channel)`` with *parts*.
    """
    time = np.arange(audio.shape[0]) / sample_rate
    if parts is not None:
        return xr.DataArray(
            np.array(parts, dtype="float32"), dims=("stem", "time", "channel"), name=name,
            coords={"stem": names, "time": time, "channel": ["left", "right"]}, attrs=attrs)
    result = xr.DataArray(
        audio.astype("float32"), dims=("time", "channel"), name=name,
        coords={"time": time, "channel": ["left", "right"]}, attrs=attrs)
    if dst:
        write_wav(result, dst)
    return result


def write_wav(audio, path, sample_rate=None):
    """Write audio to a 16-bit ``.wav`` file.

    Args:
        audio: Result of :func:`sonify`, or an array of samples in -1..1.
        path: Destination path; missing folders are created.
        sample_rate: Sample rate in Hz. Defaults to the one stored in ``audio``.
    """
    rate = sample_rate or getattr(audio, "attrs", {}).get("sample_rate", SAMPLE_RATE)
    samples = np.asarray(audio, dtype=float)
    if samples.any():
        # Triangular dither keeps quiet reverb tails from turning into distortion.
        rng = np.random.default_rng(0)
        samples = samples + (rng.random(samples.shape) - rng.random(samples.shape)) / 32767
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    if os.path.dirname(str(path)):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    wavfile.write(str(path), int(rate), pcm)
    logger.info("Saved %s (%.1f s).", path, pcm.shape[0] / rate)
