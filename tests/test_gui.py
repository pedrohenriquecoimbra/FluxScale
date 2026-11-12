"""The page's engine (fluxscale.gui.tool) and the server of `fluxscale gui`."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from fluxscale import gui
from fluxscale.gui import tool

FS = 2              # Hz: small files, same code path
MINUTES = 30
HEADER = ("Raw dataset level 6\nLegend:\n" + "legend\n" * 6 + "\n"
          "   u                        v                        w                        ts"
          "                       co2                      h2o                      ch4\n")


def write_files(folder, labels, seed=0):
    """Raw datasets as EddyPro writes them: w carries co2 down and h2o up, and ch4 is absent."""
    rng = np.random.default_rng(seed)
    n = FS * MINUTES * 60
    for label in labels:
        w = rng.normal(0, 0.3, n)
        u, v = rng.normal(2, 0.5, n), rng.normal(0, 0.5, n)
        ts = 290 + 0.5 * w + rng.normal(0, 0.1, n)
        co2 = 400 - 2 * w + rng.normal(0, 0.5, n)
        h2o = 10 + 0.5 * w + rng.normal(0, 0.1, n)
        rows = np.column_stack([u, v, w, ts, co2, h2o, np.full(n, -9999.0)])
        with open(folder / f"{label}_raw_dataset_test_adv.txt", "w") as file:
            file.write(HEADER)
            np.savetxt(file, rows, fmt="%24.16f")


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    folder = tmp_path_factory.mktemp("level_6")
    # written out of order, with 1130 missing
    write_files(folder, ["20220514-1200", "20220514-1030", "20220514-1100", "20220514-1230"])
    (folder / "notes.txt").write_text("not a raw dataset")
    return folder


def test_scan_orders_files_and_reads_what_they_hold(raw):
    found = tool.scan(str(raw))
    assert [file["label"][11:16] for file in found["files"]] == ["10:30", "11:00", "12:00", "12:30"]
    assert found["rejected"][0]["name"] == "notes.txt"
    assert found["fs"] == FS and found["fileduration"] == MINUTES
    assert found["missing"] == 1
    assert "ch4" in found["columns"] and "ch4" not in found["variables"]
    assert found["variables"] == ["u", "v", "w", "ts", "co2", "h2o"]
    json.dumps(found)


def test_scan_says_when_a_file_is_not_a_raw_dataset(tmp_path):
    (tmp_path / "20220514-1200_other.txt").write_text("time,value\n1,2\n" * 20)
    with pytest.raises(ValueError, match="does not look like an EddyPro raw dataset"):
        tool.scan(str(tmp_path))


@pytest.fixture(scope="module")
def result(raw, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    found = tool.scan(str(raw))
    options = dict(formulas=["w*co2", "w*h2o", "w*co2|w*h2o"], variables=["w", "co2", "h2o"], fs=found["fs"],
                   fileduration=found["fileduration"], first=found["first"], last=found["last"])
    return raw, work, options, tool.run(str(raw), str(work), options)


def test_run_gives_fluxes_classes_and_a_partition(result):
    _, _, _, out = result
    json.dumps(out)     # nothing JSON cannot carry, such as NaN
    assert out["rule"] == "cross_level"
    assert [t[11:] for t in out["time"]] == ["10:30", "11:00", "12:00", "12:30"]     # no window for the missing file
    products = {product["formula"]: product for product in out["products"]}
    assert {"w*co2", "w*h2o", "w*w", "co2*co2"} <= set(products)
    wco2 = products["w*co2"]
    assert len(wco2["cospectrum"]) == 4 and len(wco2["cospectrum"][0]) == len(out["frequency"])
    present = range(4)
    for i in present:               # co2 goes down with w, and the wavelet flux follows the covariance
        assert wco2["flux"][i] < 0
        assert wco2["flux"][i] == pytest.approx(wco2["covariance"][i], rel=0.2)
    classes = [cls["name"] for cls in wco2["samplings"][0]["classes"]]
    assert sorted(classes) == ["wco2+wh2o+", "wco2+wh2o-", "wco2-wh2o+", "wco2-wh2o-"]

    parts = {part["name"]: part for part in out["partitions"][0]["parts"]}
    for i in present:
        assert parts["NEE"]["flux"][i] == pytest.approx(wco2["flux"][i])
        assert parts["GPP"]["flux"][i] + parts["Reco"]["flux"][i] == pytest.approx(parts["NEE"]["flux"][i])
        assert parts["GPP"]["flux"][i] < 0
    assert len(parts["GPP"]["cospectrum"][0]) == len(out["frequency"])


def test_other_covariances_reuse_the_decomposition(result, caplog):
    raw, work, options, first = result
    again = dict(options, formulas=options["formulas"] + ["w*h2o|w*co2"])
    with caplog.at_level("INFO", logger="fluxscale"):
        out = tool.run(str(raw), str(work), again)
    assert any("Using the decomposition" in record.getMessage() for record in caplog.records)
    products = {product["formula"]: product for product in out["products"]}
    assert products["w*h2o"]["samplings"][0]["by"] == ["w*co2"]
    before = {product["formula"]: product for product in first["products"]}
    assert products["w*co2"]["flux"] == pytest.approx(before["w*co2"]["flux"], nan_ok=True)


def test_exports(result):
    _, work, _, _ = result
    csv = tool.write_export(str(work), "csv")
    with open(csv) as file:
        header = file.readline().strip().split(",")
    assert header[0] == "TIMESTAMP" and {"wco2", "NEE", "GPP", "Reco"} <= set(header)
    assert tool.write_export(str(work), "netcdf").endswith(".nc")
    with pytest.raises(ValueError):
        tool.write_export(str(work), "xlsx")


def test_run_needs_a_covariance(raw, tmp_path):
    with pytest.raises(ValueError, match="at least one covariance"):
        tool.run(str(raw), str(tmp_path), dict(formulas=[], fs=FS, fileduration=MINUTES))


@pytest.fixture
def server(raw):
    session = gui._Session(str(raw))
    hosts = set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gui._handler(session, hosts))
    port = httpd.server_address[1]
    hosts.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()
    session.close()


def fetch(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def test_server_serves_the_page_and_scans_the_folder(server, raw):
    page = fetch(server + "/").decode()
    assert 'data-engine="local"' in page and "FluxScale" in page
    assert b"chooseEngine" in fetch(server + "/app.js")
    assert json.loads(fetch(server + "/api/info"))["folder"] == str(raw)
    assert len(json.loads(fetch(server + "/api/scan", data=b""))["files"]) == 4


def test_server_takes_uploads_by_their_base_name(server, raw):
    content = next(raw.glob("20220514-1200*")).read_bytes()
    fetch(server + "/api/clear", data=b"")
    fetch(server + "/api/file?name=../../20220514-1200_x.txt", data=content)
    found = json.loads(fetch(server + "/api/scan", data=b""))
    assert [file["name"] for file in found["files"]] == ["20220514-1200_x.txt"]


def test_server_refuses_other_origins_and_paths_outside_the_page(server):
    with pytest.raises(urllib.error.HTTPError) as refused:
        fetch(server + "/api/info", headers={"Origin": "http://example.org"})
    assert refused.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as missing:
        fetch(server + "/../__init__.py")
    assert missing.value.code == 404
