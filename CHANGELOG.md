# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0:
breaking changes bump the minor version).

## [Unreleased]

### Added

- `fluxscale gui`: a page to load EddyPro raw datasets (level 6), decompose
  them and look at any covariance, its conditional sampling, and NEE, GPP and
  Reco, by window and by scale. `-i FOLDER` opens it on a folder. The same page
  is built for GitHub Pages by `scripts/build_site.py`, where it runs fluxscale
  inside the browser. See `docs/WEB_TOOL.md` and `fluxscale.gui.tool`.
- `process(n_jobs=...)` (`-j`/`--n_jobs` on the command line) processes several
  periods at once in worker processes, through the new `joblib` dependency. The
  result is the same as one at a time, the default. Each worker holds a whole
  period in memory, and each period logs when it ends.
- `method="modwt"` (`--method modwt` on the command line): the maximal overlap
  discrete wavelet transform, which has no grid, so that the same turbulence gets
  the same value wherever it happens. Its levels split the variance and the
  covariance, but do not add up to the signal, and each sits at the time of what
  it describes. With the default `fixed_grid`, each period gets the neighbours
  each level needs, and the samples are kept in
  `<output_folderpath>/decomposition`, so that a period gets the same result,
  down to rounding, however the run is cut. See `fluxscale.core.modwt`.
- `fluxscale.core.cascade.reconstruct_approximation`: what the levels of
  `reconstruct` leave out.
- `method="moda"` (`--method moda`): MODWT details, the levels filtered with the
  square of the `modwt` gain, which add up to the signal with the approximation
  (`fluxscale.core.modwt.details`). It runs on the fixed grid from the same kept
  samples as `modwt`, and fluxes use the cross-level rule.
- `fluxscale.core.handlers.data_compute_flux`: the cross-level flux of a product,
  level by level, as a density per sample or as window means.
- `cov_<xy>_filled`: the covariance of the gap-filled signals in each window
  (ddof=0), which the levels of a flux and `<x*y>_lf` add up to.
- `attrs['flux_rule']`, `'cross_level'` or `'same_scale'`, on every result.
  `open_files_in_folder` raises when a folder holds files of both rules, and
  `process` warns when it writes into such a folder.
- `fluxscale.core.cascade.reconstruct_signal`: the gap-filled signal of a stretch
  of the cascade.
- `values=` on `data_conditional_sampling` and `data_conditional_sampling_mean`:
  what to put in the classes, whose signs still come from the products.

### Changed

- **Results differ from earlier versions: fluxes are now cross-level.** With
  `dwt` and `moda`, and files that are windows, a flux such as `wco2` at level j
  is `1/2 [cov(w_j, co2) + cov(w, co2_j)]` in each window, instead of the
  same-scale product `w_j * co2_j`. So the levels and `<x*y>_lf` add up to
  `cov_<xy>_filled` in every window, which the same-scale products did not. What
  changes: `<xy>` and `<xy>~` (`csp_` and `_int` forms too) for every product of
  two variables or more, `ww` and `co2co2` included, which can be negative at
  deep levels; `<x*y>_lf`, which is now the same rule on the approximations, and
  is added for every product, such as `w*h2o_lf`; and the classes, which keep the
  signs of the same-scale products but now hold the flux of each sample, so that
  they add up to `wco2` again, apart from samples where a sign is zero. With
  `moda` these are the gap-filled samples, whose levels are zero: in a gappy
  window the rest can be a large share of the flux, and the partition puts it in
  `Reco`.
  `cross_level=False` (`--no_cross_level`) gives the earlier results, bit for
  bit. Use a new output folder when changing it. `modwt` keeps same-scale
  products, as its coefficients do not add up to the signal, and so do `cwt` and
  `fcwt` for now, with a warning. Readers without windows also keep them, with a
  warning. See "What a flux at one scale is" in the README.
- `process` now decomposes on a fixed calendar grid by default, so that a period
  gets the same result however the run is cut into blocks and wherever it starts.
  **Results differ from earlier versions**, whose blocks sat at different places
  on the wavelet grid; `fixed_grid=False, cross_level=False` (`--no_fixed_grid
  --no_cross_level` on the command line) gives them back. Samples are numbered from the Unix epoch and placed by their
  timestamp: a missing file stays a gap, filled and flagged in `<var>_qc`, as do
  the ends of the data. The run is decomposed once with the cascade and kept in
  `<output_folderpath>/decomposition`, and each period reads it back one level at
  a time, so that memory follows the period and not the number of levels. Other
  methods than `dwt`, and files that are not one window of the expected length
  each, use regular blocks, with a warning.
- `zarr` is now a regular dependency.
- `process` raises at once on an unknown `reader_method`, instead of logging an
  error for each file and returning an empty Dataset.
- `fluxscale.core.handlers.open_files_in_folder` now reads the files into memory
  and concatenates them, instead of always opening them lazily with
  `xr.open_mfdataset`, which made one dask chunk per file and variable and was
  very slow for folders of many small files: a year of daily files takes seconds
  instead of minutes. The result is identical (same variables, dims, attrs;
  sorted filename order) but numpy-backed. Folders of more than `LOAD_LIMIT`
  (1 GiB) of files are still opened lazily, with a warning; `load=True` or
  `load=False` forces either way, and `chunks=` sets the dask chunks of either
  result. The merge rules are now pinned, so the result no longer depends on
  xarray's defaults.

### Known issues

- `cwt` and `fcwt`: `universal_wt` passes `log2` scales to the inverse transform,
  which expects scales in seconds, so each level carries a wrong factor. The
  levels still add up to the signal only because the approximation is what they
  leave out. This is why they keep same-scale products for now.

### Fixed

- A period whose result could not be saved no longer leaves an incomplete file,
  which made the next run skip it as already done.
- `process` without an `output_folderpath` returned an empty Dataset, as saving
  failed; it now returns the periods without saving them.
- A run stopped by an error or Ctrl-C no longer leaves `.inprogress` markers or
  incomplete files, which made the next run skip those periods as already done.

## [0.6.0] - 2026-11-06

### Added

- `data_conditional_sampling_mean` (`conditional_sampling_mean` in
  `fluxscale.extra.partitioning.coimbra_et_al_2025`): the means of the sign
  classes, without building one full-size array per class.
- `data_run(average=True)`: the result of `data_average_dims`, without building
  the conditional sampling at full resolution.
- `fluxscale.sonify` (`fluxscale.extra.sonification`) turns any variable into
  sound at three levels: 0, the signal as an audio file; 1, the signal altered
  for human hearing (wavelet scales expanded into the hearing range, equalised
  and weighted for equal loudness); 2, music, in which a flux such as `w*co2` is
  a duet of two instruments whose covariance is heard as harmony. Instruments
  are chosen from `INSTRUMENTS`, `loudness_range` keeps calm passages softer but
  audible, and `stems=True` returns one track per wavelet scale. Experimental:
  the API may still change. Uses only existing dependencies.
- `fluxscale.decompose`: a signal or flux split into wavelet scales, as a Dataset.
- `fluxscale.export_midi` writes the same composition as MIDI, for a DAW. Needs
  the new `midi` extra.
- `fluxscale-sonify` command-line entry point, for `.wav` and `.mid` output.

### Changed

- **Renamed to FluxScale.** Install `fluxscale` and `import fluxscale`. The
  commands become `fluxscale`, `fluxscale-exec`, `fluxscale-partition`,
  `fluxscale-integrate`, `fluxscale-eddypro_run` and `fluxscale-sonify`. A last
  `waveletec` release installs `fluxscale` and keeps old imports and commands
  working. Cascade stores written before the rename still open.
- `process` only keeps averages, so it now calls `data_run(average=True)`. On a
  day of 20 Hz data with nine covariance formulas it runs 2.8x faster and peaks
  at 1.4 GB instead of 4.9 GB. The conditional sampling in its output changes
  only by floating-point summation order (relative differences below 2e-14);
  everything else is bit-identical.

## [0.5.0] - 2026-06-04

### Removed (breaking)

- **Pandas is no longer monkeypatched on import.** Importing `waveletec` no
  longer attaches `DataFrame.to_file`, `pd.read_file`,
  `DataFrame.columnstartswith`, `DataFrame.columnsmatch`, or
  `DataFrame.columnsconditioned`. Use the module-level functions in
  `waveletec.core.addons` instead (`to_file(df, path, ...)`,
  `read_file(path, ...)`, `columnstartswith(df, prefix)`,
  `columnsmatch(df, pattern)`, `columnsconditioned(df, start, *patterns)`), or
  call `waveletec.core.addons.patch_pandas()` once to restore the old
  method/function forms.
- **`waveletec.core.addons` no longer runs side effects on import.** It no
  longer overrides `warnings.showwarning`, and no longer reads matplotlib
  styles from the current working directory (that block loaded nothing and
  depended on the process CWD). Call `addons.configure_warnings()` to opt into
  routing Python warnings through the package logger.
- **Wildcard imports no longer leak incidental names.** `import waveletec` and
  `from waveletec.core.handlers import *` / `from waveletec.core.addons import *`
  no longer expose imported modules such as `os`, `re`, `np`, `xr`, `pd`, or
  `matplotlib`. Each module now declares an explicit `__all__`.

### Added

- `waveletec.core.addons`: standalone `read_file`, `to_file`,
  `columnstartswith`, `columnsmatch`, `columnsconditioned`, plus opt-in
  `patch_pandas()` and `configure_warnings()`.
- Console logging for the command line: output now appears on the console at the
  `--verbosity` level and persists during processing, while the run-log file
  continues to capture full `DEBUG`.
- Explicit `__all__` on the top-level package, `waveletec.core.handlers`, and
  `waveletec.core.addons`.

### Changed

- `waveletec.core.commons.start_logging` now attaches a `DEBUG`-level file
  handler without removing an existing console handler (it previously called
  `logging.basicConfig(force=True)`, which suppressed console output during a
  run). It is idempotent across repeated calls.
- `waveletec.core.commons` imports `pandas` and `matplotlib` directly rather
  than inheriting them from the `addons` wildcard import.
- The command-line entry points share `_add_common_args()` and `_finalize()`
  helpers instead of duplicating argument parsing and logging setup.
- Packaging metadata: added `readme`, corrected the project homepage URL, and
  fixed the license/keyword metadata. README corrected (license is EUPL-1.2,
  command-line invocation, `data_run` usage, install commands, example links).

### Fixed

- `waveletec.main`: the `exec`/`run` entry point crashed with `TypeError`
  (`**args` on an `argparse.Namespace`); it now uses `**vars(args)`.
- `waveletec.main`: running `python -m waveletec.main` no longer fails
  (`__main__` now calls `main()` directly).
- `waveletec.core.handlers`: removed a duplicate, shadowed `open_files_in_folder`
  definition.
- `waveletec.extra.eddypro`: the whitespace-separator pattern is now a raw
  string (`r"\s+"`), clearing a `SyntaxWarning`.
- Replaced stray `print()` calls and root-logger `logging.<level>()` calls
  across the library with per-module loggers.

### Renamed

- `waveletec.main.exec` -> `waveletec.main.run` (avoids shadowing the `exec`
  builtin). The `waveletEC-exec` console command is unchanged.
