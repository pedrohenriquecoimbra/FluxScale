"""
Maximal overlap discrete wavelet transform (MODWT).

The discrete transform keeps one coefficient every ``2**level`` samples, so its
values depend on where that grid falls on the signal. The MODWT keeps a
coefficient at every sample: it has no grid, and the same turbulence gets the
same value wherever it happens.

Each level here is the signal filtered by the gain of the MODWT filter of that
level, without its phase, so that a coefficient sits at the time of what it
describes, whichever the wavelet. The powers of all the levels and of the
approximation add up to one at every frequency. So the levels split the
variance of a signal, and the covariance of two signals, over a whole record:
the sum over the levels of the mean of ``x_j * y_j``, plus that of their
approximations, is the mean of ``x * y``.

The levels do not add up to the signal, as they overlap in frequency. The
details of *details* (method 'moda') do: each is the signal filtered with the
square of the gain, as ``pywt.mra`` gives them, and the details and the smooth
add up to the signal. Same-scale products of these details lose about a third
of the covariance, but the cross-level flux of the pipeline (see
handlers.data_compute_flux) needs levels that add up to the signal, and closes
on them.

Filtering runs in the frequency domain, which treats a record as periodic. Past
*reach* a filter is below rounding, so a stretch of signal needs that many
neighbours on each side to get the result of a longer record. Where a signal is
flat a level is zero, but filtering leaves rounding noise there, whose sign
would then decide the class of a sample in conditional sampling: *clean* sets
it to zero.

The main functions are:
- transform
    function: levels and approximation of a whole signal
    call: transform(signal, level=15, wavelet='db6')
- details
    function: levels and smooth of a whole signal, which add up to it
    call: details(signal, level=15, wavelet='db6')
- reach
    function: neighbours a level needs on each side
    call: reach(15)
- detail_gain, approximation_gain
    function: frequency response of a level, for filtering a stretch of signal
    call: detail_gain(n, 15)
- clean
    function: set what is only rounding to zero
    call: clean(level, scale)
"""

# built-in modules
import functools

# 3rd party modules
import numpy as np
import pywt
from scipy import fft


DEFAULT_WAVELET = 'db6'
# On real data values are either rounding noise, around 1e-15 of the largest
# value of the signal, or above 1e-10: this falls in between
ROUNDING = 1e-12


def reach(level, wavelet=DEFAULT_WAVELET):
    """
    function: samples on each side past which the filter of *level* is below
        rounding. Measured: at this distance the impulse response is 2e-16 of
        its peak, at any level
    call: reach()
    Input:
        level: level of the filter
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        samples: int
    """
    return 2 * (pywt.Wavelet(wavelet).dec_len - 1) * 2 ** int(level)


@functools.lru_cache(maxsize=4)
def _filter_powers(n, wavelet):
    """Power of the MODWT filters at the n frequencies of a record of n samples."""
    filters = pywt.Wavelet(wavelet)
    return tuple(np.abs(fft.rfft(np.array(f) / np.sqrt(2), n)) ** 2
                 for f in (filters.dec_hi, filters.dec_lo))


def _dilated(power, n, level):
    """*power* at frequencies ``2**(level - 1)`` times those of a record of n samples."""
    bins = (np.arange(n // 2 + 1, dtype=np.int64) << (level - 1)) % n
    return power[np.minimum(bins, n - bins)]


def _low_pass(n, level, wavelet):
    """Power of the level-*level* approximation filter."""
    low = _filter_powers(n, wavelet)[1]
    out = np.ones(n // 2 + 1)
    for j in range(1, level + 1):
        out *= _dilated(low, n, j)
    return out


@functools.lru_cache(maxsize=32)
def detail_gain(n, level, wavelet=DEFAULT_WAVELET):
    """
    function: gain of *level* at the frequencies of ``scipy.fft.rfft`` of n samples
    call: detail_gain()
    Input:
        n: samples of the record
        level: level of the filter
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        gain: 1D array of n // 2 + 1 values
    """
    high = _filter_powers(n, wavelet)[0]
    return np.sqrt(_dilated(high, n, level) * _low_pass(n, level - 1, wavelet))


@functools.lru_cache(maxsize=4)
def approximation_gain(n, level, wavelet=DEFAULT_WAVELET):
    """Gain of what the levels up to *level* leave out (c.f. detail_gain)."""
    return np.sqrt(_low_pass(n, level, wavelet))


def clean(level, scale):
    """
    function: set what is only rounding to zero, in place
    call: clean()
    Input:
        level: array of a level
        scale: largest absolute value of the signal, for all of *level* or
            broadcast along it. Values fading out by a flat stretch sit right at
            the threshold, so for a result that does not depend on how a run is
            cut this must come from the measurements alone (c.f. grid.SampleBlock)
    Return:
        level
    """
    level[np.abs(level) < ROUNDING * scale] = 0
    return level


def _filter_bank(signal, level, wavelet, power):
    """Levels and approximation of a whole signal, with the gains to *power*."""
    values = np.asarray(signal, dtype='float64')
    n = values.size
    mean = values.mean()
    scale = np.abs(values).max()
    spectrum = fft.rfft(values - mean)
    out = np.empty((level, n))
    for j in range(1, level + 1):
        gain = detail_gain(n, j, wavelet)
        out[j - 1] = clean(fft.irfft(spectrum * (gain if power == 1 else gain ** power), n),
                           scale)
    gain = approximation_gain(n, level, wavelet)
    return out, fft.irfft(spectrum * (gain if power == 1 else gain ** power), n) + mean


def transform(signal, level, wavelet=DEFAULT_WAVELET):
    """
    function: MODWT of a whole signal, taken as periodic
    call: transform()
    Input:
        signal: 1D array
        level: deepest level
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        details: 2D array, one row per level, the first being the finest
        approximation: 1D array
    """
    return _filter_bank(signal, level, wavelet, 1)


def details(signal, level, wavelet=DEFAULT_WAVELET):
    """
    function: MODWT details of a whole signal, taken as periodic: the levels
        filtered with the square of the gain, which add up to the signal with
        the smooth (method 'moda'). Flat stretches are cleaned as in transform,
        so the sum holds to rounding
    call: details()
    Input:
        signal: 1D array
        level: deepest level
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        details: 2D array, one row per level, the first being the finest
        smooth: 1D array
    """
    return _filter_bank(signal, level, wavelet, 2)
