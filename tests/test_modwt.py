"""Tests for the MODWT.

What makes it worth having over the discrete transform: it has no grid, so a
shift of the signal only shifts the result, and its levels split the variance
and the covariance exactly. ``pywt.swt`` is the outside reference for the energy
of each level.
"""
import numpy as np
import pytest
import pywt
import xarray as xr

from fluxscale.core import modwt
from fluxscale.core.wavelet_functions import universal_wt

LEVEL = 6


def red_noise(n=2 ** 12, seed=1):
    white = np.random.default_rng(seed).standard_normal(n)
    return white + .05 * np.cumsum(white)


@pytest.mark.parametrize("n", [2 ** 12, 3000, 3001])
def test_levels_split_the_variance(n):
    x = red_noise(n)
    details, low = modwt.transform(x, LEVEL)
    np.testing.assert_allclose((details ** 2).mean(1).sum() + (low ** 2).mean(), (x ** 2).mean())


def test_levels_split_the_covariance():
    x, y = red_noise(seed=2), red_noise(seed=3) + red_noise(seed=2)
    (dx, lx), (dy, ly) = modwt.transform(x, LEVEL), modwt.transform(y, LEVEL)
    np.testing.assert_allclose((dx * dy).mean(1).sum() + (lx * ly).mean(), (x * y).mean())


def test_each_level_has_the_energy_of_pywt():
    x = red_noise()
    details, low = modwt.transform(x, LEVEL)
    reference = pywt.swt(x, "db6", level=LEVEL, norm=True, trim_approx=True)
    np.testing.assert_allclose((low ** 2).sum(), (reference[0] ** 2).sum())
    for level in range(1, LEVEL + 1):
        np.testing.assert_allclose((details[level - 1] ** 2).sum(),
                                   (reference[LEVEL - level + 1] ** 2).sum())


@pytest.mark.parametrize("shift", [1, 7, 100])
def test_a_shift_only_shifts_the_result(shift):
    x = red_noise()
    details, low = modwt.transform(x, LEVEL)
    moved, moved_low = modwt.transform(np.roll(x, shift), LEVEL)
    np.testing.assert_allclose(moved, np.roll(details, shift, axis=1), atol=1e-11)
    np.testing.assert_allclose(moved_low, np.roll(low, shift), atol=1e-11)


def test_a_coefficient_sits_at_the_time_of_what_it_describes():
    impulse = np.zeros(2 ** 14)
    impulse[2 ** 13] = 1
    details, low = modwt.transform(impulse, LEVEL)
    response = np.column_stack([details.T, low])
    np.testing.assert_allclose(response[1:], response[:0:-1], atol=1e-12)


def test_the_approximation_keeps_the_mean():
    x = red_noise() + 400
    details, low = modwt.transform(x, LEVEL)
    np.testing.assert_allclose(low.mean(), x.mean())
    np.testing.assert_allclose(details.mean(1), 0, atol=1e-10)


def test_a_flat_stretch_has_no_level_and_not_rounding_noise():
    x = red_noise(2 ** 14) + 400
    x[6000:9000] = x[6000]
    details, _ = modwt.transform(x, 2)
    assert (details[0, 7000:8000] == 0).all()
    assert (details[0, :5000] != 0).all()


def test_clean_leaves_what_is_not_rounding():
    level = np.array([1e-15, -1e-15, 1e-6, -2.0])
    np.testing.assert_array_equal(modwt.clean(level, 10), [0, 0, 1e-6, -2.0])


@pytest.mark.parametrize("wavelet", ["db6", "haar", "sym4"])
def test_gains_are_power_complementary(wavelet):
    n = 3000
    power = sum(modwt.detail_gain(n, j, wavelet) ** 2 for j in range(1, LEVEL + 1))
    np.testing.assert_allclose(power + modwt.approximation_gain(n, LEVEL, wavelet) ** 2, 1)


def test_gain_is_the_filter_cascade_evaluated_directly():
    n = 500
    frequency = np.arange(n // 2 + 1) / n
    filters = pywt.Wavelet("db6")

    def power(values, f):
        taps = np.array(values) / np.sqrt(2)
        response = np.exp(-2j * np.pi * np.outer(f, np.arange(taps.size))) @ taps
        return np.abs(response) ** 2

    expected = power(filters.dec_hi, frequency * 4)
    for j in (1, 2):
        expected = expected * power(filters.dec_lo, frequency * j)
    np.testing.assert_allclose(modwt.detail_gain(n, 3), np.sqrt(expected), atol=1e-12)


@pytest.mark.parametrize("level", [3, 6])
def test_past_its_reach_a_filter_is_below_rounding(level):
    n = 2 ** 18
    impulse = np.zeros(n)
    impulse[n // 2] = 1
    details, low = modwt.transform(impulse, level)
    reach = modwt.reach(level)
    for response in (details[-1], low):
        outside = np.r_[response[:n // 2 - reach], response[n // 2 + reach:]]
        assert np.abs(outside).max() < 1e-14 * np.abs(response).max()


def test_universal_wt_gives_the_levels_and_their_scales():
    signal = xr.DataArray(red_noise(), dims="z")
    result = universal_wt(signal, method="modwt", fs=20, level=LEVEL)
    details, low = modwt.transform(signal, LEVEL)
    np.testing.assert_array_equal(result.wave, details)
    np.testing.assert_array_equal(result.approximation, low)
    np.testing.assert_array_equal(result.natural_frequency, 20 * 2.0 ** -np.arange(1, LEVEL + 1))


def test_universal_wt_works_out_the_level_like_the_discrete_transform():
    signal = xr.DataArray(red_noise(), dims="z")
    assert universal_wt(signal, method="modwt", fs=20, f0=20 / 2 ** 4).sizes["natural_frequency"] == 4


@pytest.mark.parametrize("n", [2 ** 12, 3000, 3001])
def test_details_add_up_to_the_signal(n):
    x = red_noise(n) + 400
    details, smooth = modwt.details(x, LEVEL)
    np.testing.assert_allclose(details.sum(0) + smooth, x, rtol=1e-12)


def test_details_are_those_of_pywt_mra():
    x = red_noise()
    details, smooth = modwt.details(x, LEVEL)
    reference = pywt.mra(x, "db6", level=LEVEL, transform="swt", mode="periodization")
    np.testing.assert_allclose(smooth, reference[0], atol=1e-10)
    for level in range(1, LEVEL + 1):
        np.testing.assert_allclose(details[level - 1], reference[LEVEL - level + 1],
                                   atol=1e-10)


def test_details_of_a_flat_stretch_are_zero():
    x = red_noise(2 ** 14) + 400
    x[6000:9000] = x[6000]
    details, _ = modwt.details(x, 2)
    assert (details[0, 7000:8000] == 0).all()


def test_universal_wt_gives_the_details_with_moda():
    signal = xr.DataArray(red_noise(), dims="z")
    result = universal_wt(signal, method="moda", fs=20, level=LEVEL)
    details, smooth = modwt.details(signal, LEVEL)
    np.testing.assert_array_equal(result.wave, details)
    np.testing.assert_array_equal(result.approximation, smooth)
    np.testing.assert_array_equal(result.natural_frequency, 20 * 2.0 ** -np.arange(1, LEVEL + 1))
    coefficients = universal_wt(signal, method="modwt", fs=20, level=LEVEL)
    np.testing.assert_array_equal(coefficients.wave, modwt.transform(signal, LEVEL)[0])
