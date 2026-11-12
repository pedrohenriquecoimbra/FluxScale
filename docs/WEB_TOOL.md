# The FluxScale page

One page, two ways to run it. Load EddyPro raw datasets (level 6), decompose
them, then look at any covariance, its conditional sampling, and NEE, GPP and
Reco, by window and by scale.

| | Online (GitHub Pages) | `fluxscale gui` |
|---|---|---|
| Where fluxscale runs | in the browser tab, with Pyodide | on your computer |
| Install | nothing | `pip install fluxscale` |
| Files | read in place by the browser, never uploaded | chosen in the page, or `-i FOLDER` |
| Speed, FR-Gri sample | about 80 s for 3 h of 20 Hz data | about 30 s for 4 h |
| Internet | needed (Pyodide comes from a CDN) | not needed |

Both call `fluxscale.gui.tool`, which calls `fluxscale.process` and
`data_partition`: the numbers are those of the package.

## Files

- `fluxscale/gui/tool.py`: `scan` (order the files, find the rate and the
  variables with data), `run`, `write_export`.
- `fluxscale/gui/__init__.py`: the local server and `fluxscale gui`. It listens
  on 127.0.0.1 only and refuses requests from other origins.
- `fluxscale/gui/web/`: the page. `app.js` picks its engine: the local server
  marks the page it serves (`data-engine="local"`); otherwise `worker.js` loads
  Pyodide.
- `scripts/build_site.py`: builds `_site/` with the wheels and a sample.
- `.github/workflows/pages.yml`: publishes `_site/`. One-time setup:
  Settings > Pages > Source: GitHub Actions.

## How a run goes

1. Files are ordered by the `YYYYMMDD-HHMM` their names start with. The sampling
   rate comes from the rows of the first file and the minutes between files.
2. The first run decomposes every variable with data except `u` and `v` (and any
   variable a formula names), by asking for its variance too. The decomposition
   stays in the work folder, so other covariances of those variables do not
   decompose again.
3. Fluxes are the sum of the scales shorter than the integration period.
4. `w*co2|w*h2o` gives the partition of Coimbra et al. (2025). Sampling by `co`
   or `ch4` gives the other partitions of `data_partition`.

## Limits

- Values are covariances in the units of the files. They are not multiplied by
  air density and have no spectral correction.
- A missing file leaves its window out.
- Only `dwt` and `moda` are offered: their levels add up to the covariance.
- Online, a day of 20 Hz data (48 files, about 400 MB) takes several minutes and
  was not tested; 3 h was. Use `fluxscale gui` for long periods.
- The Pyodide version is pinned in `web/worker.js`, and `PYODIDE_PYTHON` in
  `scripts/build_site.py` must follow it.
