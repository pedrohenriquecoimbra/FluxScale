# WaveletEC Module

WaveletEC is a Python module designed for processing and analyzing Eddy Covariance data using wavelet transforms. This module provides functionalities to handle EddyPro setup files, perform wavelet analysis, and integrate results.

[![DOI](https://zenodo.org/badge/DOI/10.1016/j.agrformet.2025.110684.svg)](https://doi.org/10.1016/j.agrformet.2025.110684)

[![DOI](https://zenodo.org/badge/786866970.svg)](https://zenodo.org/doi/10.5281/zenodo.11071327)

## Citation

Pedro H H Coimbra, Benjamin Loubet, Olivier Laurent, Matthias Mauder, Bernard Heinesch, Jonathan Bitton, Nicolas Delpierre, Daniel Berveiller, Jérémie Depuydt, Pauline Buysse. Evaluation of a novel approach to partitioning respiration and photosynthesis using eddy covariance, wavelets and conditional sampling. https://doi.org/10.1016/j.agrformet.2025.110684

\* corresponding author: pedro-henrique.herig-coimbra@inrae.fr

## Installation
### From pip

To install the WaveletEC module, use [pip](https://pypi.org/project/waveletec/):

Recommended step (optional):
```bash
conda create -n wavec
conda activate wavec
```
Install library:
```bash
pip install waveletec
```

### From git

To install the WaveletEC module, clone the repository and install the required dependencies:

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
1. Run _waveletec_:\
    This requires ```pip install waveletec```
    ```python
    import waveletec
    waveletec.run_from_eddypro("PATH/TO/EDDYPRO/SETUP", ...):
    ```
    Or directly in the command line:
    ```bash
    python -m waveletec.main --help
    python -m waveletec.main [-args...]
    ```

### Running _waveletec_

This requires ```pip install waveletec```.
```python
import waveletec
```
Then
```python
waveletec.process(...)
```
or directly run from a DataFrame or dictionary:
```python
data = ...
waveletec.data_run(data, ...)
```

### Low frequencies over long records (deconstructed cascade)

The regular pipeline processes one averaging window at a time, so it cannot
resolve scales longer than the window. For that, `waveletec.core.cascade`
decomposes arbitrarily long records (years of 20 Hz data) with bounded
memory, in two stages: chunks are decomposed to `split_level` with overlap
margins and the coefficients appended to a [zarr](https://zarr.dev) store
(`pip install waveletec[cascade]`); the carried approximation — a factor
`2**split_level` shorter — is then decomposed in memory down to any depth.
The result is exactly (machine precision) the DWT of the whole record.

```python
from waveletec.core import cascade

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

If you're using ```waveletec.process```, just pass the ```output_folderpath``` parameter like this: ```waveletec.process(..., output_folderpath='PATH/TO/OUTPUT/FOLDER')```. This will create a new subfolder called 'wavelet_full_cospectra' for you, and you'll find the sum of all your files neatly stored in the output_folderpath you specified.

Prefer using ```waveletec.data_run```? No problem! Pass the ```dst``` parameter (a single output file) like this: ```waveletec.data_run(data, ..., dst='PATH/TO/OUTPUT.nc')```.


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
