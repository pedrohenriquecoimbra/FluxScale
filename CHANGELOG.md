# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0:
breaking changes bump the minor version).

## [Unreleased]

### Added

- `data_conditional_sampling_mean` (`conditional_sampling_mean` in
  `waveletec.extra.partitioning.coimbra_et_al_2025`): the means of the sign
  classes, without building one full-size array per class.
- `data_run(average=True)`: the result of `data_average_dims`, without building
  the conditional sampling at full resolution.
- `waveletec.sonify` (`waveletec.extra.sonification`) turns any variable into
  sound at three levels: 0, the signal as an audio file; 1, the signal altered
  for human hearing (wavelet scales expanded into the hearing range, equalised
  and weighted for equal loudness); 2, music, in which a flux such as `w*co2` is
  a duet of two instruments whose covariance is heard as harmony. Instruments
  are chosen from `INSTRUMENTS`, `loudness_range` keeps calm passages softer but
  audible, and `stems=True` returns one track per wavelet scale. Experimental:
  the API may still change. Uses only existing dependencies.
- `waveletec.decompose`: a signal or flux split into wavelet scales, as a Dataset.
- `waveletec.export_midi` writes the same composition as MIDI, for a DAW. Needs
  the new `midi` extra.
- `waveletEC-sonify` command-line entry point, for `.wav` and `.mid` output.

### Changed

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
