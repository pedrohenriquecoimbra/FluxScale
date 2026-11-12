// Runs fluxscale off the main thread, with Python loaded into the page by Pyodide.
// It must stay a module worker: importScripts from the CDN failed in testing (FluxPrint).
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

// What fluxscale imports and Pyodide ships. dask and regorator come as wheels with the page.
const PACKAGES = ["micropip", "numpy", "scipy", "pandas", "xarray", "pywavelets", "scikit-learn", "pyyaml",
  "zarr", "joblib", "matplotlib", "netcdf4", "toolz", "cloudpickle", "click", "fsspec"];
const INPUT = "/input";
const WORK = "/work";

globalThis.fsNotify = (text, done, total) => postMessage({ type: "progress", text, done, total });

async function start() {
  postMessage({ type: "progress", text: "Downloading Python into this page (about 60 MB, kept by your browser after the first visit)", done: 0, total: 0 });
  const python = await loadPyodide({ stdout: () => {}, stderr: () => {} });
  await python.loadPackage(PACKAGES, { messageCallback: () => {} });
  postMessage({ type: "progress", text: "Loading FluxScale", done: 0, total: 0 });
  const wheels = await (await fetch("wheels/index.json")).json();
  const urls = wheels.map((name) => new URL(`wheels/${name}`, self.location).href);
  python.FS.mkdirTree(INPUT);
  await python.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(urls)}, deps=False)
import json
from js import fsNotify
from fluxscale.gui import tool
tool.set_notify(fsNotify)
`);
  return python;
}

const ready = start();
let mounted = false;

// The files stay where the browser has them: Python reads them through this mount, without a copy in memory.
async function files({ id, list }) {
  const python = await ready;
  if (mounted) python.FS.unmount(INPUT);
  python.FS.mount(python.FS.filesystems.WORKERFS, { files: list }, INPUT);
  mounted = true;
  const result = JSON.parse(await python.runPythonAsync(`json.dumps(tool.scan(${JSON.stringify(INPUT)}))`));
  postMessage({ type: "result", id, result });
}

async function run({ id, options }) {
  const python = await ready;
  python.globals.set("options_text", JSON.stringify(options));
  const result = JSON.parse(await python.runPythonAsync(
    `json.dumps(tool.run(${JSON.stringify(INPUT)}, ${JSON.stringify(WORK)}, json.loads(options_text)))`));
  postMessage({ type: "result", id, result });
}

async function exportFile({ id, kind }) {
  const python = await ready;
  python.globals.set("export_kind", kind);
  const path = await python.runPythonAsync(`tool.write_export(${JSON.stringify(WORK)}, export_kind)`);
  const bytes = python.FS.readFile(path);
  postMessage({ type: "result", id, result: { name: path.split("/").pop(), bytes: bytes.buffer } }, [bytes.buffer]);
}

const JOBS = { files, run, export: exportFile };

self.onmessage = ({ data }) => {
  JOBS[data.type](data).catch((error) => postMessage({ type: "error", id: data.id, text: String(error.message || error).trim().split("\n").pop().slice(-600) }));
};

ready.then(
  () => postMessage({ type: "ready" }),
  (error) => postMessage({ type: "error", id: null, text: String(error).slice(-600) }),
);
