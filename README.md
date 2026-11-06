# FluxScale

FluxScale (formerly WaveletEC) is a Python module designed for processing and analyzing Eddy Covariance data using wavelet transforms. This module provides functionalities to handle EddyPro setup files, perform wavelet analysis, and integrate results.

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.agrformet.2025.110684.svg)](https://doi.org/10.1016/j.agrformet.2025.110684)

[![DOI](https://zenodo.org/badge/786866970.svg)](https://zenodo.org/doi/10.5281/zenodo.11071327)

## Citation

Pedro H H Coimbra, Benjamin Loubet, Olivier Laurent, Matthias Mauder, Bernard Heinesch, Jonathan Bitton, Nicolas Delpierre, Daniel Berveiller, Jérémie Depuydt, Pauline Buysse. Evaluation of a novel approach to partitioning respiration and photosynthesis using eddy covariance, wavelets and conditional sampling. https://doi.org/10.1016/j.agrformet.2025.110684

\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr

## Installation
### From pip

To install the FluxScale module, use [pip](https://pypi.org/project/fluxscale/):

Recommended step (optional):
```bash
conda create -n wavec
conda activate wavec
```
Install library:
```bash
pip install fluxscale
```

### From git

To install the FluxScale module, clone the repository and install the required dependencies:

```bash
git clone https://github.com/pedrohenriquecoimbra/wavelet-ec
cd wavelet-ec
```
Then one of the following options:
- ```pip install .```
- ```conda env create -f environment.yml```


## Usage

### (Recommended) Using EddyPro

1. Make sure [EddyPro®](https://www.licor.com/support/EddyPro/software.html) is installed.
1. Run EddyPro, saving level 6 raw data.
   - go to Advanced Settings (top menu) > Output Files (left menu) > Processed raw data (bottom);
   - select Time series on "level 6 (after time lag compensation)";
   - select all variables;
   - proceed as usual running on "Advanced Mode".
1. Run _fluxscale_:\
    This requires ```pip install fluxscale```
    ```python
    import fluxscale
    fluxscale.run_from_eddypro("PATH/TO/EDDYPRO/SETUP", ...):
    ```
    Or directly in the command line:
    ```bash
    python -m fluxscale.main --help
    python -m fluxscale.main [-args...]
    ```

### Running _fluxscale_

This requires ```pip install fluxscale```.
```python
import fluxscale
```
Then
```python
fluxscale.process(...)
```
or directly run from a DataFrame or dictionary:
```python
data = ...
fluxscale.data_run(data, ...)
```

### Low frequencies over long records (deconstructed cascade)

The regular pipeline processes one averaging window at a time, so it cannot
resolve scales longer than the window. For that, `fluxscale.core.cascade`
decomposes arbitrarily long records (years of 20 Hz data) with bounded
memory, in two stages: chunks are decomposed to `split_level` with overlap
margins and the coefficients appended to a [zarr](https://zarr.dev) store
(`pip install fluxscale[cascade]`); the carried approximation — a factor
`2**split_level` shorter — is then decomposed in memory down to any depth.
The result is exactly (machine precision) the DWT of the whole record.

```python
from fluxscale.core import cascade

store = cascade.CascadeStore.create(
    'FR-Gri_cascade.zarr', variables=['w', 'co2'], n_samples=n, fs=20,
    split_level=9, t0='2020-01-01T00:00:00')          # n multiple of 2**9
cascade.cascade_run(store, {'w': w, 'co2': co2})      # stage 1: chunked, resumable
cascade.cascade_finish(store)                         # stage 2: seconds, re-runnable

cascade.scale_covariance(store, 'w', 'co2', window=1800)  # cospectra per scale
cascade.open_level(store, 'co2', 21)   # ~29 h scale (needs >= ~2 weeks of data
                                       #  so cascade_finish reaches level 21)
```

Sources may be numpy/memmap/dask/xarray/pandas or a callable `f(i0, i1)`
reading samples from disk, which must return exactly `i1 - i0` samples; gaps
(NaN), however long, are linearly interpolated and reported per coefficient
in `qc` arrays. Levels sum to the whole-record covariance exactly only when
`n` is divisible by `2**level`: otherwise the record-end coefficients of the
deep levels are off (0.2 to 11% of the total on the 3-day FR-Gri sample),
interior coefficients being unaffected. See `tests/test_cascade.py` for
exactness guarantees.

### Saving Results

If you're using ```fluxscale.process```, just pass the ```output_folderpath``` parameter like this: ```fluxscale.process(..., output_folderpath='PATH/TO/OUTPUT/FOLDER')```. This will create a new subfolder called 'wavelet_full_cospectra' for you, and you'll find the sum of all your files neatly stored in the output_folderpath you specified.

Prefer using ```fluxscale.data_run```? No problem! Pass the ```dst``` parameter (a single output file) like this: ```fluxscale.data_run(data, ..., dst='PATH/TO/OUTPUT.nc')```.


### Listening to the Data

Any variable can be turned into sound with ```fluxscale.sonify```, at one of three levels. This part of the library is experimental: its API may still change.

| Level | | What it does |
| --- | --- | --- |
| 0 | Audio file | The signal itself, written as audio: one data sample per audio sample. |
| 1 | Audible | The same signal altered for human hearing: its wavelet scales are expanded into the hearing range, equalised and weighted for equal loudness. Every sound is still a piece of the data. |
| 2 | Music (default) | Each wavelet scale drives a voice on a consonant scale, and the value of the variable plays a melody over it. A flux is a duet. |

```python
import fluxscale

fluxscale.sonify(data, "co2", level=0, dst="co2_L0.wav")
fluxscale.sonify(data, "co2", level=1, dst="co2_L1.wav")
fluxscale.sonify(data, "co2", dst="co2_L2.wav")
fluxscale.sonify(data, "w*co2", duration=60, dst="flux.wav")
fluxscale.sonify(data, "w*co2", instruments=("cello", "marimba"), dst="flux.wav")

stems = fluxscale.sonify(data, "co2", stems=True)    # one track per wavelet scale, summing to the mix
split = fluxscale.decompose(data, "w*co2", fs=20)    # the scales themselves, as a Dataset
fluxscale.export_midi(data, "w*co2", "flux.mid")     # the same music for a DAW (pip install "fluxscale[midi]")
```

```data``` can be an array, a ```pandas.Series``` or an ```xarray.DataArray```, or a Dataset, DataFrame or dictionary holding the variable. Products such as ```"w*co2"``` and wavelet-domain outputs of ```data_run``` such as ```"wco2-wh2o+"``` work too.

Why level 1 is needed: speed and pitch are tied together in a waveform. Playing half an hour of the FR-Gri sample in 30 s leaves only 3% (```h2o```) to 46% (```w```) of its variance audible. Wavelets untie them, so that scales can be moved into the hearing range without changing how fast time runs.

A flux is the covariance between two variables, so at level 2 it is played by two instruments, one per variable, each following its own value. What you hear between them is the flux:

| You hear | It means |
| --- | --- |
| Two instruments (flute and bell by default) | The first and the second variable (e.g. ```w``` and ```co2```) |
| Moving together, or in opposite directions | Positive or negative covariance |
| Sweet intervals, or looser ones with seconds and sevenths | Strong local correlation, or none |
| Voices swelling underneath | The flux at each wavelet scale, low notes for large eddies |
| Lowest note: major or relative minor | Upward or downward flux |

For a day of ```w*co2``` at FR-Gri, the instruments drift apart at night, then lock into consonant contrary motion in a minor key from mid-morning to late afternoon, as photosynthesis takes CO2 down.

Instruments are chosen by name from ```fluxscale.extra.sonification.INSTRUMENTS```, one per variable: held ones (flute, clarinet, oboe, bassoon, saxophone, horn, trumpet, trombone, violin, viola, cello, double_bass, voice, organ, accordion) and struck or plucked ones (bell, piano, harp, harpsichord, guitar, electric_guitar, bass_guitar, pizzicato, marimba, vibraphone, glockenspiel, celesta, timpani, kalimba). They are simple synthesised sketches, not recordings. In a duet, one of each kind is easiest to tell apart.

Gaps in the data are silent. Over a long record, calm passages stay softer than turbulent ones, but by no more than ```loudness_range``` (6 LU by default, ```None``` keeps the full swing), so that a quiet night remains audible. The same data and seed always give the same sound. Other options are ```duration```, ```scale```, ```root```, ```loudness``` and ```seed``` (see ```help(fluxscale.sonify)```); the result is an ```xarray.DataArray``` whose ```attrs``` record the mapping. In a notebook:

```python
from IPython.display import Audio

audio = fluxscale.sonify(data, "w*co2")
Audio(audio.values.T, rate=audio.attrs["sample_rate"], normalize=False)
```

Or from the command line:

```bash
fluxscale-sonify -i "output/eddypro_raw_datasets/level_6/20220513-12*" -v "w*co2" -o flux.wav
fluxscale-sonify -i "output/eddypro_raw_datasets/level_6/20220513-12*" -v "w*co2" -o flux.mid
```

#### Working in a DAW

The instruments are simple sketches. For real ones, take the L2 composition to Reaper, Ableton or MuseScore as MIDI, with the stems as a reference (```pip install "fluxscale[midi]"```):

```python
from fluxscale.extra.sonification import write_wav

kwargs = dict(duration=30, scale="pentatonic", instruments=("cello", "marimba"))

fluxscale.export_midi(data, "w*co2", "daw/flux.mid", **kwargs)
stems = fluxscale.sonify(data, "w*co2", stems=True, **kwargs)
for name in stems.stem.values:
    write_wav(stems.sel(stem=name), f"daw/{name.replace(':', '')}.wav")
```

Tracks and stems have the same names and, with the same arguments, the same notes: one per wavelet scale (```0.078125 Hz```), the ```bass``` of a duet, and one per melody line (```line: co2```). Each line carries the General MIDI program of its instrument, a starting point for loading your own.

- **Lines** are the notes of the audio, with velocities from the activity of the variable.
- **Scales** stay on the note of their voice. A note sounds whenever the energy at that scale has moved, so the velocities trace it: a steady scale is one long note.
- **Time.** A melody slot is an eighth note, which makes 90 BPM unless the record has few samples, and the file carries its tempo. Import it when asked, keep the tracks separate, and place everything at the start of the project: tracks and stems then stay in time.
- **Not in the file.** Keys are equal-tempered, so the pure ratios of the audio become the nearest key. Pan, room and mastering stay in the stems.

Web apps built on this are in [FluxSound](https://github.com/pedrohenriquecoimbra/FluxSound).


### Output Format

The output file for the wavelet-based (co)spectra analysis is structured as follows:

```cs
1   wavelet_based_(co)spectra
2   --------------------------------------------------------------
3   TIMESTAMP_START = 2022-05-13 00:00:00
4   TIMESTAMP_END = 2022-05-13 00:30:00
5   N: 133
6   TIME_BUFFER [min] = nan
7   frequency [Hz]
8   y-axis -> nan
9   mother_wavelet -> dwt
10  acquisition_frequency [Hz] = 20.0
11  averaging_interval [Min] = 30min
12  natural_frequency,variable,value
13  3.814697265625e-05,co2,417.0002460141172
.   ...,...,...
.   5.0,co2,1.708199383374261e-05
.   10.0,co2,1.7947312058017124e-07
.   ...,...,...
```

#### Explanation of Fields:

- **`wavelet_based_(co)spectra`**: Indicates that the data pertains to wavelet-based (co)spectra analysis.

- **`TIMESTAMP_START` and `TIMESTAMP_END`**: These fields specify the start and end times of the data collection period.

- **`N`**: Represents the number of data points or samples included in the analysis.

- **`TIME_BUFFER [min]`**: Indicates the time buffer in minutes. A value of 0 means no buffer was applied.

- **`frequency [Hz]`**: Specifies the frequency of the data is in Hertz.

- **`y-axis_->_wavelet_coefficient_*_`**: Indicates that the y-axis represents the wavelet coefficients.

- **`mother_wavelet -> dwt`**: Specifies the type of mother wavelet used in the analysis, in this case, `dwt` (Discrete Wavelet Transform).

- **`acquisition_frequency [Hz]`**: The frequency at which data was acquired, given in Hertz.

- **`averaging_interval [Min]`**: The interval over which the data was averaged, specified in minutes.

- **`natural_frequency,variable,value`**: This line contains the actual data points:
  - **`natural_frequency`**: The frequency at which the data point was recorded.
  - **`variable`**: The variable being measured, such as `co2`.
  - **`value`**: The value of the variable at the specified natural frequency.


## Example

For an example follow the [example.ipynb](https://github.com/pedrohenriquecoimbra/wavelet-ec/blob/main/sample/FR-Gri_20220514/example.ipynb) file in [sample](https://github.com/pedrohenriquecoimbra/wavelet-ec/tree/main/sample/).

## License

This project is licensed under the EUPL-1.2 License. See the [LICENSE](LICENSE) file for details.
