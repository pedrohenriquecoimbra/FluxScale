"""Build the FluxScale page into _site/, ready to serve as static files.

The page is the one `fluxscale gui` serves, in fluxscale/gui/web. Served as
plain files it has no fluxscale behind it, so it runs Python inside the
browser (Pyodide) and needs what this script adds:

- wheels/: fluxscale built from this checkout, and the pure-Python
  dependencies Pyodide does not ship, so that a visitor never needs PyPI;
- sample/: a few raw datasets of the FR-Gri sample, for the page's example.

    pip install build
    python scripts/build_site.py
    python -m http.server -d _site      # to look at it
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "fluxscale" / "gui" / "web"
OUT = ROOT / "_site"
# Pure-Python dependencies that Pyodide does not ship. The others are loaded by web/worker.js.
PACKAGES = ["dask==2026.8.0", "regorator==0.2.0"]
PYODIDE_PYTHON = "3.14"     # of the Pyodide in web/worker.js
SAMPLE = ROOT / "sample" / "FR-Gri_20220514" / "output" / "eddypro_raw_datasets" / "level_6"
SAMPLE_FILES = [f"20220514-{hour}" for hour in ("1100", "1130", "1200", "1230", "1300", "1330")]


def run(*command):
    subprocess.run([sys.executable, "-m", *command], check=True, cwd=ROOT)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(WEB, OUT)
    (OUT / ".nojekyll").write_text("")

    wheels = OUT / "wheels"
    run("pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheels), "--quiet")
    run("pip", "download", *PACKAGES, "--no-deps", "--dest", str(wheels), "--quiet", "--only-binary", ":all:",
        "--python-version", PYODIDE_PYTHON, "--implementation", "py", "--abi", "none", "--platform", "any")
    names = sorted(path.name for path in wheels.glob("*.whl"))
    (wheels / "index.json").write_text(json.dumps(names))

    sample = OUT / "sample"
    sample.mkdir()
    copied = []
    for start in SAMPLE_FILES:
        for path in sorted(SAMPLE.glob(f"{start}_*.txt")):
            shutil.copy(path, sample / path.name)
            copied.append(path.name)
    if copied:
        (sample / "index.json").write_text(json.dumps(copied))
    else:       # the page hides its example when there is none
        sample.rmdir()

    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) / 1e6
    print(f"Built {OUT} ({size:.0f} MB): {', '.join(names)}; {len(copied)} sample files.")


if __name__ == "__main__":
    main()
