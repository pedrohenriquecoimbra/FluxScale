// The FluxScale page. It talks to one of two engines with the same four calls:
// the fluxscale behind `fluxscale gui` when the page is served from this computer,
// or Python inside this tab (worker.js) when it is served as plain files.

const $ = (id) => document.getElementById(id);
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4"];
const WIND = new Set(["u", "v"]);
const SVG = "http://www.w3.org/2000/svg";

const state = { engine: null, scan: null, formulas: [], result: null, views: [], view: 0, picked: null, busy: false, ran: null };

// ---------------------------------------------------------------- engines

async function localEngine(info) {
  const call = async (path, body) => {
    const response = await fetch(path, body === undefined ? {} : { method: "POST", body });
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error || response.statusText);
    return answer;
  };
  return {
    kind: "local",
    folder: info.folder,
    version: info.version,
    ready: Promise.resolve(),
    scan: () => call("api/scan", ""),
    async load(files, progress) {
      await call("api/clear", "");
      for (let i = 0; i < files.length; i++) {
        progress("Copying files", i, files.length);
        await call(`api/file?name=${encodeURIComponent(files[i].name)}`, files[i]);
      }
      return call("api/scan", "");
    },
    async run(options, progress) {
      await call("api/run", JSON.stringify(options));
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        const job = await call("api/status");
        if (job.state === "done") return job.result;
        if (job.state === "error") throw new Error(job.text);
        progress(job.text, job.done, job.total);
      }
    },
    async export(kind) {
      const link = Object.assign(document.createElement("a"), { href: `api/export?kind=${kind}`, download: "" });
      link.click();
    },
  };
}

function workerEngine() {
  const worker = new Worker("worker.js", { type: "module" });
  const waiting = new Map();
  let next = 1;
  let progress = () => {};
  let started;
  const ready = new Promise((resolve, reject) => { started = { resolve, reject }; });
  ready.catch(() => {});
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") progress(data.text, data.done, data.total);
    else if (data.type === "ready") started.resolve();
    else if (data.id === null) started.reject(new Error(data.text));
    else {
      const job = waiting.get(data.id);
      waiting.delete(data.id);
      if (data.type === "error") job.reject(new Error(data.text));
      else job.resolve(data.result);
    }
  };
  worker.onerror = (event) => started.reject(new Error(event.message || "The page could not start Python."));
  const call = (type, payload, report) => new Promise((resolve, reject) => {
    if (report) progress = report;
    const id = next++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ type, id, ...payload });
  });
  return {
    kind: "browser",
    ready,
    onStart(report) { progress = report; },
    load: (files, report) => call("files", { list: files }, report),
    run: (options, report) => call("run", { options }, report),
    async export(kind) {
      const file = await call("export", { kind });
      const url = URL.createObjectURL(new Blob([file.bytes]));
      Object.assign(document.createElement("a"), { href: url, download: file.name }).click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    },
  };
}

// `fluxscale gui` marks the page it serves; plain files have no server to ask.
async function chooseEngine() {
  if (document.documentElement.dataset.engine === "local") return localEngine(await (await fetch("api/info")).json());
  return workerEngine();
}

// ---------------------------------------------------------------- status

function progress(text, done = 0, total = 0) {
  $("progress").hidden = false;
  $("status").textContent = total ? `${text}, ${done} of ${total}` : text;
  $("fill").classList.toggle("busy", !total);
  $("fill").style.width = total ? `${(100 * done) / total}%` : "";
}

function idle() { $("progress").hidden = true; }

function complain(error) {
  idle();
  $("message").hidden = false;
  $("message").textContent = error.message || String(error);
}

function setBusy(busy) {
  state.busy = busy;
  $("run").disabled = busy;
  $("files").disabled = busy;
  $("sample").disabled = busy;
  if (busy) $("message").hidden = true;
}

// ---------------------------------------------------------------- files

const day = (iso) => iso.replace("T", " ").slice(0, 16);

async function loadFiles(files) {
  if (!files.length || state.busy) return;
  setBusy(true);
  try {
    await state.engine.ready;
    progress("Reading the files");
    showScan(await state.engine.load(files, progress));
  } catch (error) { complain(error); }
  setBusy(false);
}

function showScan(scan) {
  idle();
  state.scan = scan;
  state.ran = null;
  const notes = scan.rejected.map((file) => `${file.name} was left out: ${file.reason}.`);
  $("form").hidden = !scan.files.length;
  $("found").hidden = false;
  if (!scan.files.length) {
    $("found").textContent = "No raw dataset among these files.";
  } else {
    const hours = (scan.files.length * scan.fileduration) / 60;
    $("found").textContent = `${scan.files.length} files, ${day(scan.first)} to ${day(scan.last)} (${+hours.toFixed(1)} h), ` +
      `${scan.fs} Hz, with ${scan.variables.join(", ")}.`;
    if (scan.missing) notes.push(`${scan.missing} file${scan.missing > 1 ? "s are" : " is"} missing inside this period: those windows are left out.`);
    if (!/level 6/i.test(scan.level)) notes.push(`These files say “${scan.level}”. FluxScale expects level 6, after time lag compensation.`);
    const empty = scan.columns.filter((name) => !scan.variables.includes(name));
    if (empty.length) notes.push(`No data in ${empty.join(", ")}.`);
    $("order").replaceChildren(...scan.files.map((file) => Object.assign(document.createElement("li"), { textContent: file.name })));
    $("fs").value = scan.fs;
    $("duration").value = scan.fileduration;
    $("integration").value = scan.fileduration;
    defaultFormulas();
  }
  $("listing").hidden = !scan.files.length;
  $("checks").hidden = !notes.length;
  $("checks").replaceChildren(...notes.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
  $("run").textContent = "Decompose and compute";
  $("reuse").hidden = true;
}

// ---------------------------------------------------------------- formulas

function defaultFormulas() {
  const have = state.scan.variables;
  const scalars = have.filter((name) => name !== "w" && !WIND.has(name));
  state.formulas = have.includes("w") ? scalars.map((name) => `w*${name}`) : [];
  if (["w", "co2", "h2o"].every((name) => have.includes(name))) state.formulas.push("w*co2|w*h2o");
  const options = (selected) => have.map((name) => new Option(name, name, false, name === selected));
  $("cov-x").replaceChildren(...options(have.includes("w") ? "w" : have[0]));
  $("cov-y").replaceChildren(...options(scalars[0] || have[0]));
  showFormulas();
}

const pretty = (formula) => {
  const [of, ...by] = formula.split("|");
  return by.length ? `${of} by ${by.join(" and ")}` : of;
};

function showFormulas() {
  $("formulas").replaceChildren(...state.formulas.map((formula) => {
    const chip = Object.assign(document.createElement("button"), { type: "button", textContent: pretty(formula) });
    chip.setAttribute("aria-label", `Remove ${pretty(formula)}`);
    chip.onclick = () => { state.formulas = state.formulas.filter((f) => f !== formula); showFormulas(); };
    return chip;
  }));
  const plain = state.formulas.filter((formula) => !formula.includes("|"));
  $("cs-of").replaceChildren(...plain.map((f) => new Option(f, f, false, f === "w*co2")));
  $("cs-by").replaceChildren(...plain.map((f) => new Option(f, f, false, f === "w*h2o")));
  $("add-cs").disabled = plain.length < 2;
}

function addFormula(formula) {
  if (!state.formulas.includes(formula)) state.formulas.push(formula);
  showFormulas();
}

// ---------------------------------------------------------------- run

async function run(event) {
  event.preventDefault();
  if (state.busy) return;
  const scan = state.scan;
  const variables = [...new Set([...state.formulas.flatMap((f) => f.split(/[|*]/)), ...scan.variables.filter((name) => !WIND.has(name))])];
  const options = {
    formulas: state.formulas, variables, fs: +$("fs").value, fileduration: +$("duration").value,
    method: $("method").value, integration_period: +$("integration").value * 60, first: scan.first, last: scan.last,
  };
  const key = JSON.stringify([variables.slice().sort(), options.fs, options.fileduration, options.method]);
  setBusy(true);
  progress(state.ran === key ? "Computing covariances" : "Reading files");
  try {
    showResult(await state.engine.run(options, progress));
    state.ran = key;
    $("run").textContent = "Update";
    $("reuse").hidden = false;
    $("reuse").textContent = `Other covariances of ${variables.join(", ")} reuse this decomposition, so they take seconds.`;
    idle();
  } catch (error) { complain(error); }
  setBusy(false);
}

// ---------------------------------------------------------------- result

const classLabel = (name) => name.replace(/([+-])/g, "$1 ").replace(/-/g, "−").trim();

function buildViews(result) {
  const views = [];
  const asked = new Set(result.formulas.map((formula) => formula.split("|")[0]));
  for (const partition of result.partitions) {
    views.push({
      label: `NEE, GPP and Reco: ${partition.label}`, kind: "partition",
      series: partition.parts.map((part) => ({ label: part.name, values: part.flux, spectrum: part.cospectrum })),
    });
  }
  for (const product of result.products.filter((p) => asked.has(p.formula))) {
    views.push({
      label: `${product.formula}: wavelet flux against eddy covariance`, kind: "covariance", formula: product.formula,
      series: [
        { label: "Wavelet flux", values: product.flux, spectrum: product.cospectrum },
        { label: "Eddy covariance of each file", values: product.covariance, dashed: true },
      ],
    });
    for (const sampling of product.samplings) {
      views.push({
        label: `${product.formula} sampled by ${sampling.by.join(" and ")}`, kind: "sampling", formula: product.formula,
        series: sampling.classes.slice(0, SERIES.length).map((cls) => ({ label: classLabel(cls.name), values: cls.flux, spectrum: cls.cospectrum })),
        dropped: Math.max(sampling.classes.length - SERIES.length, 0),
      });
    }
  }
  for (const product of result.products.filter((p) => !asked.has(p.formula))) {
    views.push({
      label: `Variance of ${product.variables[0]} by scale`, kind: "variance", formula: product.formula,
      series: [{ label: `Variance of ${product.variables[0]}`, values: product.flux, spectrum: product.cospectrum }],
    });
  }
  views.forEach((view) => view.series.forEach((series, i) => { series.color = `var(${SERIES[i]})`; }));
  return views;
}

function showResult(result) {
  state.result = result;
  state.views = buildViews(result);
  state.times = result.time.map((text) => Date.parse(`${text}:00Z`));
  const keep = state.views.findIndex((view) => view.label === state.viewLabel);
  state.view = keep >= 0 ? keep : 0;
  state.picked = null;
  $("view").replaceChildren(...state.views.map((view, i) => new Option(view.label, i, false, i === state.view)));
  for (const id of ["views", "time-card", "scale-row", "table-card", "units", "results"]) $(id).hidden = false;
  $("empty").hidden = true;
  const stats = [
    ["Windows", result.time.length], ["Scales", result.frequency.length],
    ["Method", result.method === "dwt" ? "DWT" : "MODWT details"],
    ["Fluxes add up scales shorter than", `${+(result.integration_period / 60).toFixed(1)} min`],
    ["fluxscale", result.version],
  ];
  $("stats").replaceChildren(...stats.flatMap(([name, value]) => [
    Object.assign(document.createElement("dt"), { textContent: name }), Object.assign(document.createElement("dd"), { textContent: value })]));
  $("notes").replaceChildren(...result.warnings.map((text) => Object.assign(document.createElement("li"), { textContent: text })));
  draw();
}

// ---------------------------------------------------------------- charts

const fmt = (value) => {
  if (value === null || value === undefined || !Number.isFinite(value)) return "no value";
  if (value === 0) return "0";
  const size = Math.abs(value);
  return (size >= 1e4 || size < 1e-3 ? value.toExponential(2) : value.toPrecision(3)).replace("-", "−");
};
const hz = (f) => (f >= 1 ? `${+f.toPrecision(2)} Hz` : f >= 1e-3 ? `${+(f * 1e3).toPrecision(2)} mHz` : `${+(f * 1e6).toPrecision(2)} µHz`);
const period = (f) => { const s = 1 / f; return s < 90 ? `${+s.toPrecision(2)} s` : s < 5400 ? `${+(s / 60).toPrecision(2)} min` : `${+(s / 3600).toPrecision(2)} h`; };
const clock = (ms) => new Date(ms).toISOString().slice(0, 16).replace("T", " ");

function el(name, attributes = {}, text) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function ticks(low, high, count = 5) {
  const span = high - low || Math.abs(high) || 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function extent(lists) {
  let low = 0, high = 0;       // a flux reads against zero, so zero is always in view
  for (const list of lists) for (const v of list || []) if (v !== null) { if (v < low) low = v; if (v > high) high = v; }
  if (low === high) high = low + 1;
  const pad = (high - low) * 0.06;
  return [low < 0 ? low - pad : 0, high > 0 ? high + pad : 0];
}

function path(points) {
  let d = "", pen = false;
  for (const point of points) {
    if (!point) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${point[0].toFixed(1)},${point[1].toFixed(1)}`;
    pen = true;
  }
  return d;
}

function tip(event, title, rows) {
  const box = $("tip");
  box.replaceChildren(Object.assign(document.createElement("b"), { textContent: title }), ...rows.map(([color, label, value]) => {
    const row = document.createElement("span");
    if (color) row.append(Object.assign(document.createElement("i"), { style: `border-color:${color}` }));
    row.append(label, Object.assign(document.createElement("em"), { textContent: value }));
    return row;
  }));
  box.hidden = false;
  const { width, height } = box.getBoundingClientRect();
  box.style.left = `${Math.min(event.clientX + 14, innerWidth - width - 8)}px`;
  box.style.top = `${Math.max(event.clientY - height - 10, 8)}px`;
}
const untip = () => { $("tip").hidden = true; };

// A frame with a y axis: returns the svg, the plot box and the y scale.
function frame(host, height, [low, high]) {
  const width = Math.max(host.clientWidth, 280);
  const box = { left: 56, right: width - 26, top: 8, bottom: height - 24 };
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" });
  const y = (v) => box.bottom - ((v - low) / (high - low)) * (box.bottom - box.top);
  for (const v of ticks(low, high)) {
    svg.append(el("line", { class: v === 0 ? "zero" : "grid", x1: box.left, x2: box.right, y1: y(v), y2: y(v) }));
    svg.append(el("text", { x: box.left - 6, y: y(v) + 4, "text-anchor": "end" }, fmt(v)));
  }
  host.replaceChildren(svg);
  return { svg, box, y, width };
}

function drawTime(view) {
  const host = $("time-plot");
  const times = state.times;
  const { svg, box, y } = frame(host, 250, extent(view.series.map((s) => s.values)));
  const t0 = times[0], t1 = times[times.length - 1];
  const step = state.result.fileduration * 60e3;
  const x = (t) => (t1 === t0 ? (box.left + box.right) / 2 : box.left + ((t - t0) / (t1 - t0)) * (box.right - box.left));
  const every = Math.max(1, Math.ceil(times.length / Math.max(2, Math.floor((box.right - box.left) / 90))));
  times.forEach((t, i) => {
    if (i % every) return;
    const label = clock(t);
    svg.append(el("text", { x: x(t), y: box.bottom + 16, "text-anchor": "middle" }, t1 - t0 > 36 * 3600e3 ? label.slice(5) : label.slice(11)));
  });
  if (state.picked !== null) svg.append(el("line", { class: "picked", x1: x(times[state.picked]), x2: x(times[state.picked]), y1: box.top, y2: box.bottom }));
  for (const series of view.series) {
    if (!series.values) continue;
    const points = series.values.map((v, i) => (v === null ? null : [x(times[i]), y(v)]));
    // a missing file has no window: the line stops there instead of crossing the gap
    const drawn = points.flatMap((p, i) => (i && times[i] - times[i - 1] > 1.5 * step ? [null, p] : [p]));
    svg.append(el("path", { class: "line", d: path(drawn), stroke: series.color, ...(series.dashed ? { "stroke-dasharray": "6 4" } : {}) }));
    if (times.length <= 60) for (const p of points) if (p) svg.append(el("circle", { cx: p[0], cy: p[1], r: 4, fill: series.color }));
  }
  const cursor = el("line", { class: "cursor", y1: box.top, y2: box.bottom, visibility: "hidden" });
  const hit = el("rect", { class: "hit", x: box.left, y: box.top, width: box.right - box.left, height: box.bottom - box.top });
  svg.append(cursor, hit);
  const nearest = (event) => {
    const rect = svg.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * svg.viewBox.baseVal.width;
    let best = 0;
    times.forEach((t, i) => { if (Math.abs(x(t) - px) < Math.abs(x(times[best]) - px)) best = i; });
    return best;
  };
  hit.onpointermove = (event) => {
    const i = nearest(event);
    cursor.setAttribute("x1", x(times[i])); cursor.setAttribute("x2", x(times[i])); cursor.setAttribute("visibility", "visible");
    tip(event, `${clock(times[i])}, end of the window`, view.series.filter((s) => s.values).map((s) => [s.color, s.label, fmt(s.values[i])]));
  };
  hit.onpointerleave = () => { cursor.setAttribute("visibility", "hidden"); untip(); };
  hit.onclick = (event) => { state.picked = nearest(event); draw(); };
}

function meanOver(spectrum) {
  return spectrum[0].map((_, j) => {
    const values = spectrum.map((row) => row[j]).filter((v) => v !== null);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  });
}

// Scales run down the side, fast eddies on top, as in the map beside it; what each carries runs across.
function drawScale(view) {
  const host = $("scale-plot");
  const frequency = state.result.frequency;
  const lines = view.series.filter((s) => s.spectrum).map((s) => ({ ...s, values: state.picked === null ? meanOver(s.spectrum) : s.spectrum[state.picked] }));
  const width = Math.max(host.clientWidth, 280), height = 250;
  const box = { left: 56, right: width - 26, top: 8, bottom: height - 24 };
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" });
  const [low, high] = extent(lines.map((s) => s.values));
  const x = (v) => box.left + ((v - low) / (high - low)) * (box.right - box.left);
  const row = (box.bottom - box.top) / frequency.length;
  const order = frequency.map((f, j) => j).sort((a, b) => frequency[b] - frequency[a]);     // fastest first
  const place = new Map(order.map((j, k) => [j, box.top + (k + 0.5) * row]));
  const y = (j) => place.get(j);
  for (const v of ticks(low, high, Math.max(2, Math.floor((box.right - box.left) / 80)))) {
    svg.append(el("line", { class: v === 0 ? "zero" : "grid", x1: x(v), x2: x(v), y1: box.top, y2: box.bottom }));
    svg.append(el("text", { x: x(v), y: box.bottom + 16, "text-anchor": "middle" }, fmt(v)));
  }
  order.forEach((j, k) => { if (k % 3 === 0) svg.append(el("text", { x: box.left - 6, y: y(j) + 4, "text-anchor": "end" }, period(frequency[j]))); });
  // the integration limit sits between the last scale a flux adds up and the first it leaves out
  const limit = 1 / state.result.integration_period;
  const kept = order.filter((j) => frequency[j] >= limit).length;
  if (kept < frequency.length) {
    const at = box.top + kept * row;
    svg.append(el("line", { class: "cursor", x1: box.left, x2: box.right, y1: at, y2: at }));
  }
  for (const series of lines) {
    const points = order.map((j) => (series.values[j] === null ? null : [x(series.values[j]), y(j)]));
    svg.append(el("path", { class: "line", d: path(points), stroke: series.color }));
    for (const p of points) if (p) svg.append(el("circle", { cx: p[0], cy: p[1], r: 4, fill: series.color }));
  }
  const cursor = el("line", { class: "cursor", x1: box.left, x2: box.right, visibility: "hidden" });
  const hit = el("rect", { class: "hit", x: box.left, y: box.top, width: box.right - box.left, height: box.bottom - box.top });
  svg.append(cursor, hit);
  host.replaceChildren(svg);
  hit.onpointermove = (event) => {
    const rect = svg.getBoundingClientRect();
    const py = ((event.clientY - rect.top) / rect.height) * height;
    const j = order[Math.max(0, Math.min(order.length - 1, Math.floor((py - box.top) / row)))];
    cursor.setAttribute("y1", y(j)); cursor.setAttribute("y2", y(j)); cursor.setAttribute("visibility", "visible");
    tip(event, `Eddies of about ${period(frequency[j])} (${hz(frequency[j])})`, lines.map((s) => [s.color, s.label, fmt(s.values[j])]));
  };
  hit.onpointerleave = () => { cursor.setAttribute("visibility", "hidden"); untip(); };
}

function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
function rgb(name) {
  const probe = Object.assign(document.createElement("i"), { style: `color:var(${name})` });
  document.body.append(probe);
  const parts = getComputedStyle(probe).color.match(/[\d.]+/g).slice(0, 3).map(Number);
  probe.remove();
  return parts;
}

function drawMap(view) {
  const host = $("map-plot");
  const series = view.series.find((s) => s.spectrum);
  const frequency = state.result.frequency;
  const times = state.times;
  const width = Math.max(host.clientWidth, 280), height = 226;
  const box = { left: 56, right: width - 12, top: 8, bottom: height - 24 };
  const ratio = window.devicePixelRatio || 1;
  const canvas = Object.assign(document.createElement("canvas"), { width: width * ratio, height: height * ratio });
  canvas.style.height = `${height}px`;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${series.label} by window and scale`);
  const pen = canvas.getContext("2d");
  pen.scale(ratio, ratio);
  const sizes = series.spectrum.flat().filter((v) => v !== null).map(Math.abs).sort((a, b) => a - b);
  const top = sizes[Math.floor(sizes.length * 0.98)] || sizes[sizes.length - 1] || 1;
  const [negative, zero, positive, ink, ground] = ["--pole-negative", "--pole-zero", "--pole-positive", "--muted", "--ground"].map(rgb);
  const cw = (box.right - box.left) / times.length, ch = (box.bottom - box.top) / frequency.length;
  series.spectrum.forEach((row, i) => row.forEach((v, j) => {
    const t = v === null ? 0 : Math.max(-1, Math.min(1, v / top));
    pen.fillStyle = v === null ? `rgb(${ground})` : `rgb(${mix(zero, t < 0 ? negative : positive, Math.abs(t))})`;
    pen.fillRect(box.left + i * cw, box.top + j * ch, Math.ceil(cw), Math.ceil(ch));
  }));
  pen.fillStyle = `rgb(${ink})`;
  pen.font = "11px sans-serif";
  pen.textAlign = "right";
  frequency.forEach((f, j) => { if (j % 3 === 0) pen.fillText(period(f), box.left - 6, box.top + (j + 0.5) * ch + 4); });
  pen.textAlign = "center";
  const every = Math.max(1, Math.ceil(times.length / Math.max(2, Math.floor((box.right - box.left) / 90))));
  const long = times[times.length - 1] - times[0] > 36 * 3600e3;
  times.forEach((t, i) => { if (i % every === 0) pen.fillText(long ? clock(t).slice(5) : clock(t).slice(11), box.left + (i + 0.5) * cw, box.bottom + 16); });
  if (state.picked !== null) {
    pen.strokeStyle = `rgb(${rgb("--ink")})`;
    pen.lineWidth = 1.5;
    pen.strokeRect(box.left + state.picked * cw, box.top, cw, box.bottom - box.top);
  }
  const bar = Object.assign(document.createElement("div"), { className: "scalebar" });
  bar.append(`−${fmt(top)}`, document.createElement("i"), `+${fmt(top)}`);
  host.replaceChildren(canvas, bar);
  const cell = (event) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((event.clientX - rect.left - box.left) / cw), j = Math.floor((event.clientY - rect.top - box.top) / ch);
    return i >= 0 && i < times.length && j >= 0 && j < frequency.length ? [i, j] : null;
  };
  canvas.onpointermove = (event) => {
    const at = cell(event);
    if (!at) return untip();
    tip(event, `${clock(times[at[0]])}, eddies of about ${period(frequency[at[1]])}`, [[null, series.label, fmt(series.spectrum[at[0]][at[1]])]]);
  };
  canvas.onpointerleave = untip;
  canvas.onclick = (event) => { const at = cell(event); if (at) { state.picked = at[0]; draw(); } };
  canvas.style.cursor = "crosshair";
  return series;
}

function drawTable(view) {
  const columns = view.series.filter((s) => s.values);
  const head = document.createElement("tr");
  head.append(...["End of the window", ...columns.map((s) => s.label)].map((text) => Object.assign(document.createElement("th"), { textContent: text })));
  const rows = state.times.map((t, i) => {
    const row = document.createElement("tr");
    row.append(...[clock(t), ...columns.map((s) => fmt(s.values[i]))].map((text) => Object.assign(document.createElement("td"), { textContent: text })));
    return row;
  });
  $("table").replaceChildren(head, ...rows);
}

function draw() {
  const view = state.views[state.view];
  if (!view) return;
  state.viewLabel = view.label;
  $("legend").replaceChildren(...view.series.map((series) => {
    const item = document.createElement("span");
    item.append(Object.assign(document.createElement("i"), { className: series.dashed ? "dashed" : "", style: `border-color:${series.color}` }), series.label);
    return item;
  }));
  const titles = {
    partition: ["NEE, GPP and Reco in each window", "GPP is the downward CO2 carried with upward water vapour, plus half of the rest of the downward CO2. Reco is NEE minus GPP."],
    covariance: [`${view.formula} in each window`, "The wavelet flux adds up the scales shorter than the integration period. The eddy covariance is the plain covariance within each file."],
    sampling: [`${view.formula} in each window, by class`, `Each sample goes to a class by the signs of the products at its scale.${view.dropped ? ` ${view.dropped} more classes are in the downloads.` : ""}`],
    variance: [`${view.series[0].label} in each window`, "The variance carried by the scales shorter than the integration period."],
  }[view.kind];
  $("time-title").textContent = titles[0];
  $("time-note").textContent = `${titles[1]} Click a window to see its scales.`;
  drawTime(view);
  $("scale-title").textContent = state.picked === null ? "By scale, mean of all windows" : `By scale, window ending ${clock(state.times[state.picked])}`;
  $("scale-note").textContent = `What each scale carries: eddy duration down, as in the map. A flux adds up the scales above the dashed line (${period(1 / state.result.integration_period)}).`;
  $("all-windows").hidden = state.picked === null;
  drawScale(view);
  const mapped = drawMap(view);
  $("map-title").textContent = `${mapped.label}, every window and scale`;
  $("map-note").textContent = "Windows across, eddy duration down. Colour is what the scale carries, around zero.";
  drawTable(view);
  $("units").textContent = "Values are covariances in the units of the files: for w*co2, m s-1 times the unit of co2 in the raw data " +
    "(µmol mol-1 for a mixing ratio). They are not multiplied by air density, and have no spectral correction.";
}

// ---------------------------------------------------------------- start

async function trySample() {
  setBusy(true);
  try {
    const names = await (await fetch("sample/index.json")).json();
    const files = [];
    for (let i = 0; i < names.length; i++) {
      progress("Downloading the sample", i, names.length);
      files.push(new File([await (await fetch(`sample/${names[i]}`)).blob()], names[i]));
    }
    setBusy(false);
    await loadFiles(files);
  } catch (error) { complain(error); setBusy(false); }
}

async function start() {
  const engine = state.engine = await chooseEngine();
  const local = engine.kind === "local";
  $("where").textContent = local ? "It runs on this computer." : "It runs in this browser tab: nothing is installed, and your files are not uploaded.";
  $("privacy").textContent = local
    ? "This page is served by the fluxscale installed on this computer, which does the computing."
    : "Your files stay on your computer: Python runs inside this tab. A day of 20 Hz data takes a few minutes; `fluxscale gui` runs the same page faster on your computer.";
  $("foot").textContent = local ? `fluxscale ${engine.version}, on this computer.` : "Runs with Pyodide. For long periods, pip install fluxscale and run: fluxscale gui";

  $("files").onchange = (event) => loadFiles([...event.target.files]);
  const drop = $("drop");
  drop.ondragover = (event) => { event.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (event) => { event.preventDefault(); drop.classList.remove("over"); loadFiles([...event.dataTransfer.files]); };
  $("sample").onclick = trySample;
  $("add-cov").onclick = () => addFormula(`${$("cov-x").value}*${$("cov-y").value}`);
  $("add-cs").onclick = () => { if ($("cs-of").value !== $("cs-by").value) addFormula(`${$("cs-of").value}|${$("cs-by").value}`); };
  $("form").onsubmit = run;
  $("view").onchange = (event) => { state.view = +event.target.value; draw(); };
  $("all-windows").onclick = () => { state.picked = null; draw(); };
  $("downloads").onclick = (event) => {
    const kind = event.target.dataset.kind;
    if (kind) state.engine.export(kind).catch(complain);
  };
  let pending;
  new ResizeObserver(() => { clearTimeout(pending); pending = setTimeout(draw, 120); }).observe($("stage"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

  if (local) {
    if (engine.folder) engine.scan().then(showScan, complain);
    return;
  }
  fetch("sample/index.json").then((response) => { $("sample").hidden = !response.ok; }, () => {});
  engine.onStart(progress);
  engine.ready.then(() => { if (!state.busy) idle(); }, complain);
}

start();
