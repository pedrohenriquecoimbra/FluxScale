"""Tests for fluxscale.extra.sonification.

Whether something sounds good cannot be unit-tested. What can be tested is that
each level does what it claims: L0 is the signal, and L1 moves it into the
hearing range without moving it in time. Music, level 2, is tested in FluxSound.
"""
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import signal as spsignal
from scipy.io import wavfile

import fluxscale
from fluxscale import main as wmain
from fluxscale.extra import sonification as so
from fluxscale.extra.decomposition import decompose

# Numerical trouble here goes quiet instead of crashing: a NaN in one band once
# silenced every voice while the rest of the audio stayed finite.
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

RATE = 22050


def red_noise(n=2 ** 14, seed=0, slope=-5 / 3):
    """Turbulence-like signal: random phases under a power-law spectrum."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n)
    spectrum = np.zeros(freqs.size, dtype=complex)
    spectrum[1:] = freqs[1:] ** (slope / 2) * np.exp(2j * np.pi * rng.random(freqs.size - 1))
    signal = np.fft.irfft(spectrum, n)
    return signal / signal.std()


def correlated(r, n=2 ** 14, slope=-5 / 3):
    """Two variables with correlation *r* at every scale, as a table."""
    a, b = red_noise(n, 1, slope), red_noise(n, 2, slope)
    return {"a": a, "b": r * a + np.sqrt(1 - r ** 2) * b}


def share(audio, low, high, rate=RATE):
    """Fraction of the power of *audio* between *low* and *high* Hz."""
    freqs, power = spsignal.welch(np.asarray(audio).mean(axis=1), fs=rate, nperseg=4096)
    return power[(freqs >= low) & (freqs < high)].sum() / power.sum()


def rms(audio, start, stop, rate=RATE):
    """RMS of *audio* between two times in seconds."""
    return float(np.sqrt((np.asarray(audio)[int(start * rate):int(stop * rate)] ** 2).mean()))


BAND_CENTRE = 0.09375


# --------------------------------------------------------------------------- #
# L0: the signal as an audio file
# --------------------------------------------------------------------------- #
def test_level0_is_the_signal_itself():
    signal = red_noise() + 400  # a large mean, like CO2 in ppm
    audio = so.sonify(signal, level=0)
    assert audio.sizes["time"] == signal.size  # one audio sample per data sample
    np.testing.assert_array_equal(audio.values[:, 0], audio.values[:, 1])
    inner = slice(1000, -1000)  # away from the declicked edges
    assert np.corrcoef(audio.values[inner, 0], signal[inner])[0, 1] > 0.9999
    assert np.abs(audio.values).max() == pytest.approx(so.PEAK_CEILING, abs=1e-4)


def test_level0_warns_that_slowing_down_drops_the_signal_below_hearing(caplog):
    signal = red_noise(slope=-1)
    fast = so.sonify(signal, level=0)
    assert fast.attrs["audible_fraction"] > 0.5
    with caplog.at_level("WARNING"):
        slow = so.sonify(signal, level=0, duration=120, sample_rate=RATE)
    assert slow.attrs["audible_fraction"] < 0.5 < fast.attrs["audible_fraction"]
    assert "audible" in caplog.text
    assert slow.sizes["time"] == 120 * RATE


# --------------------------------------------------------------------------- #
# L1: altered for hearing
# --------------------------------------------------------------------------- #
def test_level1_reaches_the_hearing_range_that_level0_cannot():
    signal = red_noise()
    raw = so.sonify(signal, level=0, duration=10, sample_rate=RATE)
    heard = so.sonify(signal, level=1, duration=10, sample_rate=RATE)
    assert share(raw, 1000, 10000) < 1e-3
    assert share(heard, 1000, 10000) > 0.05
    assert share(raw, 0, 20) > 3 * share(heard, 0, 20)


def test_level1_keeps_fine_scales_above_large_ones():
    audio = so.sonify(red_noise(), level=1, duration=1, fs=20, sample_rate=RATE)
    bands, targets = audio.attrs["band_frequency_hz"], audio.attrs["voice_frequency_hz"]
    assert bands == sorted(bands) and targets == sorted(targets)
    assert targets[0] == pytest.approx(so.HEARING_BAND[0])
    assert targets[-1] == pytest.approx(so.HEARING_BAND[1])


def test_level1_equalises_without_erasing_the_colour_of_a_variable():
    def centroid(slope):
        audio = so.sonify(red_noise(slope=slope), level=1, duration=4, sample_rate=RATE)
        freqs, power = spsignal.welch(audio.values.mean(axis=1), fs=RATE, nperseg=4096)
        return (freqs * power).sum() / power.sum()

    steep, flat = centroid(-3), centroid(0)
    assert steep > 200  # unequalised, only the lowest band (100 Hz) would be left
    assert steep < 0.8 * flat


def test_level1_keeps_a_pure_tone_pure():
    # Equalisation must not lift wavelet leakage into a band of noise.
    tone = np.sin(2 * np.pi * BAND_CENTRE * np.arange(2 ** 14))
    audio = so.sonify(tone, level=1, duration=4, fs=1.0, sample_rate=RATE)
    bands = np.array(audio.attrs["band_frequency_hz"])
    target = audio.attrs["voice_frequency_hz"][int(np.argmin(np.abs(bands * 0.75 - BAND_CENTRE)))]
    assert share(audio.values, target / 2, target * 2) > 0.8


@pytest.mark.parametrize("level", [1])
def test_an_event_is_heard_when_and_where_it_happened(level):
    n, duration = 2 ** 14, 8
    t = np.arange(n)
    signal = 0.02 * red_noise(n)
    during = (t > 0.4 * n) & (t < 0.6 * n)
    signal[during] += np.sin(2 * np.pi * BAND_CENTRE * t[during])

    # Without levelling, which by design lets an event stand out by less.
    audio = so.sonify(signal, level=level, duration=duration, fs=1.0, sample_rate=RATE,
                      loudness_range=None)
    bands = np.array(audio.attrs["band_frequency_hz"])
    voice = audio.attrs["voice_frequency_hz"][int(np.argmin(np.abs(bands * 0.75 - BAND_CENTRE)))]
    sections = spsignal.butter(2, [voice / 1.15, voice * 1.15], "bandpass", fs=RATE, output="sos")
    filtered = spsignal.sosfilt(sections, audio.values, axis=0)
    assert rms(filtered, 3.6, 4.4) > 4 * rms(filtered, 1.5, 2.5)


@pytest.mark.parametrize("level", [0, 1])
def test_gaps_are_silent(level):
    signal = red_noise()
    signal[6000:10000] = np.nan  # from 2.9 s to 4.9 s of 8 s
    audio = so.sonify(signal, level=level, duration=8, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    assert rms(audio.values, 4.4, 4.8) < 1e-3 * rms(audio.values, 1.5, 2.5)


@pytest.mark.parametrize("level", [1])
def test_a_spike_does_not_drown_the_record(level):
    signal = red_noise()
    spiked = signal.copy()
    spiked[8000] = 1000.0
    clean = so.sonify(signal, level=level, duration=6, sample_rate=RATE)
    audio = so.sonify(spiked, level=level, duration=6, sample_rate=RATE)
    assert rms(audio.values, 0.5, 2) > 0.5 * rms(clean.values, 0.5, 2)


@pytest.mark.parametrize("level", [1])
def test_the_sound_depends_on_the_data_and_the_seed_only(level):
    signal = red_noise()
    kwargs = dict(level=level, duration=2, sample_rate=RATE)
    first = so.sonify(signal, seed=3, **kwargs).values
    np.testing.assert_array_equal(first, so.sonify(signal, seed=3, **kwargs).values)
    assert not np.allclose(first, so.sonify(signal, seed=4, **kwargs).values)
    assert not np.allclose(first, so.sonify(red_noise(seed=9), seed=3, **kwargs).values)


@pytest.mark.parametrize("level", [0, 1])
def test_fs_labels_the_output_without_changing_the_sound(level):
    signal = red_noise(2 ** 12)
    fast = so.sonify(signal, level=level, duration=2, fs=20, sample_rate=RATE)
    slow = so.sonify(signal, level=level, duration=2, fs=1 / 1800, sample_rate=RATE)
    np.testing.assert_array_equal(fast.values, slow.values)
    assert slow.attrs["speedup"] == pytest.approx(fast.attrs["speedup"] * 20 * 1800)


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
def test_k_weighting_reproduces_the_coefficients_of_the_standard():
    shelf, high_pass = so._k_weighting(48000)
    np.testing.assert_allclose(shelf, [1.53512485958697, -2.69169618940638, 1.19839281085285,
                                       1.0, -1.69065929318241, 0.73248077421585], rtol=1e-8)
    np.testing.assert_allclose(high_pass, [1.0, -2.0, 1.0,
                                           1.0, -1.99004745483398, 0.99007225036621], rtol=1e-8)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_loudness_meets_the_reference_of_the_standard(rate):
    # BS.1770: a full-scale 997 Hz tone in one channel reads -3.01 LKFS.
    tone = np.sin(2 * np.pi * 997 * np.arange(5 * rate) / rate)
    assert so.integrated_loudness(tone, rate) == pytest.approx(-3.01, abs=0.02)


def test_loudness_of_silence_is_minus_infinity():
    assert so.integrated_loudness(np.zeros((RATE, 2)), RATE) == -np.inf


@pytest.mark.parametrize("level", [1])
def test_loudness_and_peaks_are_mastered(level):
    audio = so.sonify(red_noise(), level=level, duration=6, loudness=-20, sample_rate=RATE)
    assert so.integrated_loudness(audio.values, RATE) == pytest.approx(-20, abs=0.5)
    assert np.abs(audio.values).max() <= so.PEAK_CEILING + 1e-4
    assert np.abs(audio.values[0]).max() < 1e-3 and np.abs(audio.values[-1]).max() < 1e-3


# --------------------------------------------------------------------------- #
# Quiet passages stay softer, but audible
# --------------------------------------------------------------------------- #
def noise_at(*levels_db, seconds=6.0):
    """Stereo noise holding each level in turn; ``None`` is a gap."""
    rng = np.random.default_rng(0)
    n = int(seconds * RATE)
    return np.concatenate([np.zeros((n, 2)) if db is None
                           else rng.standard_normal((n, 2)) * 10 ** (db / 20)
                           for db in levels_db])


def db(audio, start, stop):
    return 20 * np.log10(rms(audio, start, stop))


def narrowed(audio, n_record=None):
    gain = so._swing_gain(audio, RATE, n_record or audio.shape[0], 6.0)
    return audio if gain is None else audio * gain[:, None]


def test_a_wide_swing_is_narrowed_without_closing():
    audio = noise_at(-30, -10)
    level = narrowed(audio)
    assert db(audio, 8, 10) - db(audio, 2, 4) == pytest.approx(20, abs=0.5)
    assert db(level, 8, 10) - db(level, 2, 4) == pytest.approx(6, abs=1)


def test_soft_never_overtakes_loud():
    level = narrowed(noise_at(-40, -30, -20, -10))
    levels = [db(level, start, start + 2) for start in (2, 8, 14, 20)]
    assert levels == sorted(levels)


def test_a_record_within_the_limit_is_left_alone():
    assert so._swing_gain(noise_at(-12, -10), RATE, int(12 * RATE), 6.0) is None


def test_a_gap_neither_widens_the_range_nor_makes_a_sound():
    assert so._swing_gain(noise_at(-14, None, -10), RATE, int(18 * RATE), 6.0) is None
    level = narrowed(noise_at(-30, None, -10))
    assert not level[int(6.5 * RATE):int(11.5 * RATE)].any()
    assert db(level, 14, 16) - db(level, 2, 4) == pytest.approx(6, abs=1)


def test_sound_beside_a_gap_is_not_boosted():
    level = narrowed(noise_at(-30, -10, None, -10))
    assert db(level, 11, 11.9) == pytest.approx(db(level, 8, 10), abs=1)


def test_the_ring_out_has_no_say():
    assert so._swing_gain(noise_at(-12, -10, -60), RATE, int(12 * RATE), 6.0) is None


@pytest.mark.parametrize("kept, low, high", [(6.0, 3, 8), (None, 20, 40)])
def test_a_calm_half_stays_softer_by_as_much_as_asked(kept, low, high):
    signal = red_noise(slope=-1)
    signal[:signal.size // 2] *= 0.03
    audio = so.sonify(signal, level=1, duration=14, loudness_range=kept, sample_rate=RATE).values

    def loudness(start, stop):
        return so.integrated_loudness(audio[int(start * RATE):int(stop * RATE)], RATE)

    assert low < loudness(9, 12) - loudness(2, 5) < high


# --------------------------------------------------------------------------- #
# Stems: one track per scale, which sum to the mix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level, variable", [(0, "a"), (1, "a"), (1, "a*b")])
def test_stems_sum_to_the_mix(level, variable):
    kwargs = dict(variable=variable, level=level, duration=4, sample_rate=RATE)
    table = correlated(0.6)
    mix = so.sonify(table, **kwargs).values
    stems = so.sonify(table, stems=True, **kwargs)
    assert stems.dims == ("stem", "time", "channel")
    assert stems.shape[1:] == mix.shape
    np.testing.assert_allclose(stems.sum("stem").values, mix, atol=2e-4)


def test_stems_are_named_by_scale():
    kwargs = dict(duration=2, fs=20, stems=True, sample_rate=RATE)
    names = list(so.sonify(red_noise(), level=1, **kwargs).stem.values)
    assert names[-1] == "10 Hz" and all(name.endswith("Hz") for name in names)
    assert list(so.sonify(red_noise(), level=0, **kwargs).stem.values) == ["trend"] + names


def test_a_stem_holds_only_its_own_scale():
    tone = np.sin(2 * np.pi * BAND_CENTRE * np.arange(2 ** 14))
    stems = so.sonify(tone, level=1, duration=4, fs=1.0, stems=True, sample_rate=RATE)
    levels = (stems ** 2).mean(("time", "channel")).values
    assert stems.stem.values[levels.argmax()] == "0.125 Hz"


# --------------------------------------------------------------------------- #
# Any variable
# --------------------------------------------------------------------------- #
def test_accepts_any_container():
    values = red_noise(2 ** 11)
    series = pd.Series(values, name="co2")
    reference = so.sonify(values, level=0).values
    for data, variable in [(values.tolist(), None), (series, None),
                           (xr.DataArray(values, dims="t"), None),
                           ({"co2": values}, "co2"), (series.to_frame(), "co2"),
                           (series.to_xarray().to_dataset(), "co2")]:
        np.testing.assert_array_equal(so.sonify(data, variable, level=0).values, reference)


def test_blocks_play_one_after_another():
    values = red_noise(2 ** 11)
    blocks = xr.Dataset({"co2": (("TIMESTAMP", "ns"), values.reshape(4, -1))})
    np.testing.assert_array_equal(so.sonify(blocks, "co2", level=0).values,
                                  so.sonify(values, level=0).values)


def test_infers_the_sampling_rate_from_time():
    index = pd.date_range("2022-05-13", periods=2 ** 10, freq="50ms")
    audio = so.sonify(pd.Series(red_noise(2 ** 10), index=index), level=0)
    assert audio.attrs["fs"] == pytest.approx(20)
    assert so.sonify(red_noise(2 ** 10), level=0).attrs["fs"] == 1.0


def test_product_is_the_wavelet_flux_density():
    table = correlated(0.5, slope=0)  # flat, so that the scales hold all the variance
    source = so.load(table, "a*b", 1.0, "db6")
    # summed over scales and time it recovers the covariance
    assert source["values"].mean() == pytest.approx(np.cov(table["a"], table["b"])[0, 1], rel=0.05)
    assert so.sonify(table, "a*b", level=0).attrs["variable"] == "a*b"
    assert so.sonify(table, "a*b", level=1, duration=1, sample_rate=RATE).attrs["level"] == 1


@pytest.mark.filterwarnings("default::RuntimeWarning")  # data_run is not under test here
def test_sonifies_the_output_of_data_run():
    rng = np.random.default_rng(0)
    data = xr.Dataset({name: (("TIMESTAMP", "ns"), rng.standard_normal((2, 2 ** 10)))
                       for name in ("w", "co2")},
                      coords={"TIMESTAMP": pd.date_range("2022-05-13", periods=2, freq="30min"),
                              "ns": np.arange(2 ** 10)})
    result = fluxscale.data_run(data, ["w*co2"], fs=20, method="dwt", wavelet="db6", level=5)
    audio = so.sonify(result, "wco2", level=1, duration=1, sample_rate=RATE)
    assert audio.attrs["fs"] == 20  # read from the finest band
    assert np.isfinite(audio.values).all()


def test_rejects_bands_of_a_continuous_transform():
    # universal_wt labels those with log2 scales, which would invert the mapping
    bands = xr.DataArray(np.ones((3, 64)), dims=("natural_frequency", "t"),
                         coords={"natural_frequency": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="discrete"):
        so.sonify(bands)


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("n", [4, 5, 7, 16, 48])
def test_short_series_still_make_sound(level, n):
    audio = so.sonify(red_noise(64)[:n], level=level, duration=1, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    assert np.abs(audio.values).max() > 0


@pytest.mark.parametrize("level", [0, 1])
def test_a_constant_is_silent(level):
    audio = so.sonify(np.full(256, 400.0), level=level, duration=1, sample_rate=RATE)
    assert not audio.values.any()


def test_infinite_values_count_as_gaps():
    signal = red_noise(2 ** 10)
    signal[100] = np.inf
    assert np.isfinite(so.sonify(signal, level=0).values).all()


@pytest.mark.parametrize("call, message", [
    (lambda: so.sonify(np.full(8, np.nan)), "finite"),
    (lambda: so.sonify(np.arange(3.0)), "at least 4"),
    (lambda: so.sonify(red_noise(64), level=3), "level"),
    (lambda: so.sonify(red_noise(64), duration=0), "duration"),
    (lambda: so.sonify(red_noise(64), loudness_range=0), "loudness_range"),
    (lambda: so.sonify({"a": red_noise(64)}, "b"), "not found"),
    (lambda: so.sonify({"a": red_noise(64)}), "required"),
])
def test_bad_input_is_explained(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_levels_can_be_written_as_labels():
    signal = red_noise(2 ** 10)
    np.testing.assert_array_equal(so.sonify(signal, level="L0").values,
                                  so.sonify(signal, level=0).values)


def test_level_1_is_the_default():
    signal = red_noise(2 ** 10)
    kwargs = dict(duration=1, sample_rate=RATE)
    np.testing.assert_array_equal(so.sonify(signal, **kwargs).values,
                                  so.sonify(signal, level=1, **kwargs).values)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def test_wav_round_trip_creates_folders(tmp_path):
    path = tmp_path / "deep" / "er" / "co2.wav"
    audio = so.sonify(red_noise(2 ** 12), level=1, duration=1, sample_rate=RATE, dst=str(path))
    rate, pcm = wavfile.read(path)
    assert rate == RATE
    assert pcm.dtype == np.int16
    assert pcm.shape == audio.shape
    assert np.abs(pcm / 32767 - audio.values).max() < 1e-3


def test_silence_stays_digital_silence(tmp_path):
    path = tmp_path / "silence.wav"
    so.write_wav(np.zeros((100, 2)), path, sample_rate=RATE)
    assert not wavfile.read(path)[1].any()


def test_is_part_of_the_public_api():
    assert fluxscale.sonify is so.sonify
    assert {"sonify", "decompose"} <= set(fluxscale.__all__)


def test_command_line(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for stamp in ("20220513-1200", "20220513-1230"):
        rows = "\n".join(f"{a:.5f}  {b:.5f}" for a, b in rng.standard_normal((512, 2)))
        (tmp_path / f"{stamp}_raw.txt").write_text("header\n" * 9 + "w  co2\n" + rows + "\n")
    wav = tmp_path / "flux.wav"
    monkeypatch.setattr(sys, "argv", [
        "fluxscale-sonify", "-i", str(tmp_path / "*_raw.txt"), "-v", "w*co2", "-o", str(wav),
        "-d", "1", "-af", "20", "--sample_rate", str(RATE), "--seed", "3"])
    wmain.sonify()
    rate, pcm = wavfile.read(wav)
    assert rate == RATE and pcm.any()


def test_command_line_explains_a_pattern_that_matches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fluxscale-sonify", "-i", str(tmp_path / "nope*"),
                                      "-v", "co2", "-o", str(tmp_path / "x.wav")])
    with pytest.raises(FileNotFoundError, match="No file"):
        wmain.sonify()


# --------------------------------------------------------------------------- #
# Music and MIDI are in FluxSound
# --------------------------------------------------------------------------- #
def test_music_is_in_fluxsound(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="fluxsound.sonify"):
        so.sonify(red_noise(64), level=2)
    with pytest.raises(TypeError, match="instruments"):
        so.sonify(red_noise(64), instruments="harp")
    assert not hasattr(fluxscale, "export_midi") and not hasattr(so, "INSTRUMENTS")
    for more in (["-o", "x.mid"], ["-o", "x.wav", "-l", "2"]):
        monkeypatch.setattr(sys, "argv", ["fluxscale-sonify", "-i", str(tmp_path / "*"), "-v", "co2", *more])
        with pytest.raises(SystemExit):
            wmain.sonify()
    assert "fluxsound" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #
def test_scales_and_trend_add_up_to_the_signal():
    split = decompose(red_noise(2 ** 12), fs=20)
    np.testing.assert_allclose(split.scales.sum("frequency") + split.trend, split.signal, atol=1e-9)
    assert split.frequency.values[0] == 10
    assert split.share.sum() == pytest.approx(1)


def test_shares_of_a_flux_are_the_cospectrum_and_can_be_negative():
    a = red_noise(2 ** 12, 1, slope=0)
    table = {"a": a, "b": -a}
    split = decompose(table, "a*b", fs=20)
    assert split.attrs["product"]
    assert split.share.sum() == pytest.approx(1)
    assert (split.scales.mean("time") < 0).all()
