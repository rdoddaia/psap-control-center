"""Unit tests for scripts.gpu_utilization_14d_report (offline, mocked I/O)."""

import argparse
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from unittest import mock

import pytest

import scripts.gpu_utilization_14d_report as report

# Arbitrary cluster names used as data keys in these tests.
FA = "cluster-a"
DC = "cluster-b"
T0 = 1_700_000_000
DAYS, STEP = 14, 3600


def _series(host=None, gpu=None, values=(50.0, 60.0)):
    metric = {}
    if host is not None:
        metric["host"] = host
    if gpu is not None:
        metric["gpu"] = gpu
    return {
        "metric": metric,
        "values": [[T0 + i * STEP, v] for i, v in enumerate(values)],
    }


def _sample_data():
    return {
        FA: {
            "cluster": [_series()],
            "active": [_series(values=(8.0, 8.0))],
            "node": [_series(host="fa-node-abc")],
            "gpu": [
                _series(host="fa-node-abc", gpu="0"),
                _series(host="fa-node-abc", gpu="1"),
            ],
        },
        DC: {
            "cluster": [_series()],
            "active": [_series(values=(4.0, 4.0))],
            "node": [_series(host="dc-node-xyz")],
            "gpu": [
                _series(host="dc-node-xyz", gpu="0"),
                _series(host="dc-node-xyz", gpu="1"),
            ],
        },
        "_meta": {
            "schema_version": report.REPORT_SCHEMA_VERSION,
            "days": DAYS,
            "step_seconds": STEP,
            "generated": "2023-11-14T22:13:20+00:00",
            "window_start": T0 - DAYS * 86400,
            "window_end": T0,
        },
    }


def _cfg(port=9090, hostlabel="hostname"):
    return {
        "kubeconfig": "/tmp/kc", "port": port, "hostlabel": hostlabel,
        "days": DAYS, "step": STEP,
    }


class FakeResponse:
    def __init__(self, payload):
        self._stream = io.StringIO(json.dumps(payload))

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc):
        return False


def _prom_payload():
    return {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"hostname": "node-abc-123"},
                    "values": [[T0, "50"], [T0 + STEP, "NaN"]],
                }
            ]
        },
    }


class TestQueriesFor:
    def test_queries(self):
        assert report.queries_for("hostname") == {
            "cluster": "avg(DCGM_FI_DEV_GPU_UTIL)",
            "active": f"count(DCGM_FI_DEV_FB_USED > {report.FB_ACTIVE_THRESHOLD_MIB})",
            "node": "avg by (hostname)(DCGM_FI_DEV_GPU_UTIL)",
            "gpu": "avg by (hostname, gpu)(DCGM_FI_DEV_GPU_UTIL)",
        }

    def test_alternate_hostlabel(self):
        q = report.queries_for("Hostname")
        assert q["node"] == "avg by (Hostname)(DCGM_FI_DEV_GPU_UTIL)"
        assert q["gpu"] == "avg by (Hostname, gpu)(DCGM_FI_DEV_GPU_UTIL)"


class TestNodeShort:
    @pytest.mark.parametrize(
        ("hostname", "expected"),
        [
            ("diadochos-hqxzk-gpu-h100-gjfjh", "h100-gjfjh"),
            ("a-b-c", "b-c"),
            ("a-b", "a-b"),
            ("single", "single"),
        ],
    )
    def test_node_short(self, hostname, expected):
        assert report.node_short(hostname) == expected


class TestParseClusters:
    def _args(self, **kw):
        base = {
            "cluster": None, "kubeconfig": None, "port": None,
            "hostlabel": None, "no_fetch": False, "only": None,
            "days": DAYS, "step": STEP, "out_dir": None,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    def test_empty(self):
        assert report.parse_clusters(argparse.ArgumentParser(), self._args()) == {}

    def test_single_cluster_defaults(self):
        clusters = report.parse_clusters(
            argparse.ArgumentParser(),
            self._args(cluster=["c1"], kubeconfig=["/tmp/a.kc"]),
        )
        assert clusters == {"c1": {"kubeconfig": "/tmp/a.kc",
                                   "port": report.DEFAULT_PORT,
                                   "hostlabel": report.DEFAULT_HOSTLABEL}}

    def test_multiple_clusters(self):
        clusters = report.parse_clusters(
            argparse.ArgumentParser(),
            self._args(cluster=["c1", "c2"], kubeconfig=["/tmp/a.kc", "/tmp/b.kc"],
                       port=[9090, 9093], hostlabel=["hostname", "Hostname"]),
        )
        assert clusters["c1"]["port"] == 9090
        assert clusters["c2"]["port"] == 9093
        assert clusters["c2"]["hostlabel"] == "Hostname"

    def test_duplicate_names_error(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1", "c1"], kubeconfig=["/tmp/a.kc", "/tmp/b.kc"]),
            )

    def test_kubeconfig_count_mismatch_error(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1", "c2"], kubeconfig=["/tmp/a.kc"]),
            )

    def test_missing_kubeconfig_error(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1"]),
            )

    def test_multi_cluster_requires_port(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1", "c2"], kubeconfig=["/tmp/a.kc", "/tmp/b.kc"]),
            )

    def test_invalid_hostlabel_error(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1"], kubeconfig=["/tmp/a.kc"],
                           hostlabel=["hostname) or vector(1"]),
            )

    def test_invalid_port_error(self):
        with pytest.raises(SystemExit):
            report.parse_clusters(
                argparse.ArgumentParser(),
                self._args(cluster=["c1"], kubeconfig=["/tmp/a.kc"], port=[70000]),
            )


class TestClusterNames:
    def test_excludes_meta(self):
        assert report.cluster_names({"a": {}, "_meta": {}, "b": {}}) == ["a", "b"]


class TestMetadata:
    def test_legacy_metadata_derives_original_window(self):
        data = _sample_data()
        data["_meta"].pop("window_start")
        data["_meta"].pop("window_end")
        meta = report.metadata_from_data(data)
        expected_end = int(datetime.fromisoformat(meta["generated"]).timestamp())
        assert meta["window_end"] == expected_end
        assert meta["window_start"] == expected_end - DAYS * 86400

    def test_invalid_metadata_is_rejected(self):
        data = _sample_data()
        data["_meta"]["step_seconds"] = None
        with pytest.raises(ValueError, match="invalid values"):
            report.metadata_from_data(data)


class TestSummarize:
    def test_math_and_meta_exclusion(self):
        data = {
            FA: {
                "cluster": [{"metric": {}, "values": [[1, 50.0], [2, None], [3, 100.0]]}],
                "active": [{"metric": {}, "values": [[1, 8.0], [2, 8.0]]}],
                "node": [_series(host="node-a")],
                "gpu": [
                    _series(host="node-a", gpu=str(index)) for index in range(4)
                ],
            },
            "_meta": {},
        }
        rows = report.summarize(data, DAYS, STEP)
        assert len(rows) == 1
        row = rows[0]
        assert row["cluster"] == FA
        assert row["gpus"] == 4  # derived from gpu series count
        assert row["avg_util_pct"] == 75.0
        assert row["peak_1h_avg_pct"] == 100.0
        assert row["active_gpu_hours"] == 16.0
        assert row["pct_window_active"] == round(100 * 16 / (DAYS * 24 * 4), 1)

    def test_empty_prometheus_results_are_actionable(self):
        data = _sample_data()
        data[FA]["cluster"] = []
        with pytest.raises(ValueError, match=f"{FA}: Prometheus returned no cluster series"):
            report.summarize(data, DAYS, STEP)


class TestFetchCluster:
    def _run(self, payload=None, forwards=None, cfg=None):
        if forwards is None:
            forwards = {FA: mock.Mock()}
        if payload is None:
            payload = _prom_payload()
        if cfg is None:
            cfg = _cfg()
        requests = []

        def fake_urlopen(req, timeout=None, context=None):
            requests.append(req)
            return FakeResponse(payload)

        with mock.patch.object(report.subprocess, "run") as run, mock.patch.object(
            report.urllib.request, "urlopen", side_effect=fake_urlopen
        ), mock.patch.object(report.json, "load", side_effect=lambda _r: payload):
            run.return_value = mock.Mock(stdout="token-123\n")
            result = report.fetch_cluster(FA, cfg, forwards)
        return result, requests

    def test_fetches_all_queries_and_normalizes(self):
        result, requests = self._run()
        assert set(result) == {"cluster", "active", "node", "gpu"}
        assert len(requests) == 4
        assert all(req.headers["Authorization"] == "Bearer token-123" for req in requests)
        assert all(f"step={STEP}" in req.full_url for req in requests)
        assert all("https://localhost:9090" in req.full_url for req in requests)
        series = result["cluster"][0]
        assert series["metric"]["host"] == "node-abc-123"
        assert "hostname" not in series["metric"]
        assert series["values"] == [[T0, 50.0], [T0 + STEP, None]]

    def test_alternate_hostlabel_normalized(self):
        result, _ = self._run(cfg=_cfg(hostlabel="Hostname"))
        # metric label "hostname" is absent, so no normalization happens,
        # but queries used the caller-provided label
        assert "host" not in result["cluster"][0]["metric"]

    def test_reuses_existing_forward(self):
        forwards = {FA: mock.Mock()}
        with mock.patch.object(
            report, "start_forward"
        ) as start, mock.patch.object(report.subprocess, "run"), mock.patch.object(
            report.urllib.request, "urlopen",
            side_effect=lambda *a, **k: FakeResponse(_prom_payload()),
        ), mock.patch.object(report.json, "load", side_effect=lambda _r: _prom_payload()):
            report.fetch_cluster(FA, _cfg(), forwards)
        start.assert_not_called()

    def test_starts_managed_forward(self):
        cfg = _cfg(port=9093)
        with mock.patch.object(report, "start_forward") as start, mock.patch.object(
            report.subprocess, "run"
        ), mock.patch.object(
            report.urllib.request, "urlopen",
            side_effect=lambda *a, **k: FakeResponse(_prom_payload()),
        ), mock.patch.object(report.json, "load", side_effect=lambda _r: _prom_payload()):
            report.fetch_cluster(FA, cfg, forwards={})
        start.assert_called_once_with("/tmp/kc", 9093)

    def test_error_status_raises(self):
        payload = {"status": "error", "data": {}}
        with pytest.raises(RuntimeError, match="failed"):
            self._run(payload=payload)

    def test_empty_token_is_rejected(self):
        with pytest.raises(RuntimeError, match="empty token"):
            report.fetch_cluster(
                FA, _cfg(), {FA: mock.Mock()}, token_provider=lambda _kc: ""
            )


class TestStartForward:
    def test_success(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        with mock.patch.object(report, "ensure_port_available"), \
                mock.patch.object(report.subprocess, "Popen", return_value=proc) as popen, \
                mock.patch.object(report.socket, "create_connection") as conn:
            result = report.start_forward("/tmp/kc", 9090)
        assert result is proc
        popen.assert_called_once_with(
            ["kubectl", "--kubeconfig", "/tmp/kc", "-n", "openshift-monitoring",
             "port-forward", "--address", "127.0.0.1", "svc/prometheus-k8s", "9090:9091"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        conn.assert_called_once_with(("127.0.0.1", 9090), timeout=2)

    def test_early_exit_raises(self):
        proc = mock.Mock()
        proc.poll.return_value = 0
        with mock.patch.object(report, "ensure_port_available"), \
                mock.patch.object(report.subprocess, "Popen", return_value=proc), \
                mock.patch.object(report.socket, "create_connection",
                                  side_effect=OSError):
            with pytest.raises(RuntimeError, match="exited early"):
                report.start_forward("/tmp/kc", 9090)

    def test_timeout_terminates_and_raises(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        t0 = time.time()
        with mock.patch.object(report, "ensure_port_available"), \
                mock.patch.object(report.subprocess, "Popen", return_value=proc), \
                mock.patch.object(report.socket, "create_connection",
                                  side_effect=OSError), \
                mock.patch.object(report.time, "time",
                                  side_effect=[t0, t0 + 46]), \
                mock.patch.object(report.time, "sleep"):
            with pytest.raises(RuntimeError, match="did not become ready"):
                report.start_forward("/tmp/kc", 9090)
        proc.terminate.assert_called_once_with()

    def test_rejects_port_already_in_use(self):
        with mock.patch.object(report, "ensure_port_available",
                               side_effect=RuntimeError("already in use")), \
                mock.patch.object(report.subprocess, "Popen") as popen:
            with pytest.raises(RuntimeError, match="already in use"):
                report.start_forward("/tmp/kc", 9090)
        popen.assert_not_called()


class TestToFrame:
    def test_long_form_drops_none(self):
        data = _sample_data()
        data[FA]["node"][0]["values"].append([T0 + 2 * STEP, None])
        df = report.to_frame(
            data, "node",
            lambda name, m: f"{report.node_short(m['host'])} ({name})",
        )
        assert list(df.columns) == ["cluster", "time", "label", "value"]
        assert len(df) == 4  # 2 nodes x 2 hourly points, None dropped
        assert df["value"].notna().all()
        assert set(df["label"]) == {
            f"{report.node_short('fa-node-abc')} ({FA})",
            f"{report.node_short('dc-node-xyz')} ({DC})",
        }
        assert df["time"].iloc[0] == datetime.fromtimestamp(T0, tz=timezone.utc)
        assert set(df["cluster"]) == {FA, DC}

    def test_gpu_labels(self):
        data = _sample_data()
        df = report.to_frame(
            data, "gpu",
            lambda name, m: f"{report.node_short(m['host'])} gpu{m['gpu']}",
        )
        assert set(df["label"]) == {
            f"{report.node_short('fa-node-abc')} gpu0",
            f"{report.node_short('fa-node-abc')} gpu1",
            f"{report.node_short('dc-node-xyz')} gpu0",
            f"{report.node_short('dc-node-xyz')} gpu1",
        }


class TestCharts:
    def test_cluster_figure_traces_and_axes(self):
        fig = report.chart_cluster(_sample_data(), DAYS, STEP)
        assert len(fig.data) == 4  # 2 util + 2 active
        active = [t for t in fig.data if t.yaxis == "y2"]
        assert len(active) == 2
        assert all(t.line.dash == "dot" for t in active)
        assert all("active GPUs" in t.name for t in active)
        assert fig.layout.yaxis2.overlaying == "y"

    def test_node_figure(self):
        fig = report.chart_node(_sample_data(), DAYS, STEP)
        assert len(fig.data) == 2  # one node per cluster
        names = {t.name for t in fig.data}
        assert f"{report.node_short('fa-node-abc')} ({FA})" in names
        assert f"{report.node_short('dc-node-xyz')} ({DC})" in names

    def test_gpu_figure_facets_and_annotation(self):
        data = _sample_data()
        fig = report.chart_gpu(data, DAYS, STEP)
        assert len(fig.data) == 4  # 2 GPUs per cluster
        texts = {a.text for a in fig.layout.annotations}
        assert f"{FA} — 2 GPUs" in texts
        assert f"{DC} — 2 GPUs" in texts

    def test_gpu_figure_single_cluster_no_facets(self):
        data = {FA: _sample_data()[FA], "_meta": {}}
        fig = report.chart_gpu(data, DAYS, STEP)
        assert len(fig.data) == 2
        assert not any(a.text == FA for a in fig.layout.annotations)

    def test_titles_use_days_and_step(self):
        fig = report.chart_cluster(_sample_data(), 7, 900)
        assert "last 7 days" in fig.layout.title.text
        assert "900s buckets" in fig.layout.title.text


class TestRender:
    def _render(self, tmp_path, monkeypatch, data=None):
        out = tmp_path / "report.html"
        if data is None:
            data = _sample_data()
        with mock.patch.object(
            report.plotly.offline, "get_plotlyjs", return_value="/*plotlyjs*/"
        ):
            report.render(data, DAYS, STEP, out)
        return out.read_text()

    def test_summary_table_and_figures(self, tmp_path, monkeypatch):
        html = self._render(tmp_path, monkeypatch)
        assert "<table" in html
        assert FA in html and DC in html
        for div in ("c1", "c2", "c3"):
            assert f'id="{div}"' in html
            assert f'Plotly.newPlot("{div}"' in html
        assert "/*plotlyjs*/" in html
        assert "2 GPUs" in html  # per-cluster GPU count derived from data

    def test_summary_header_uses_configured_bucket(self, tmp_path, monkeypatch):
        out = tmp_path / "report.html"
        with mock.patch.object(
            report.plotly.offline, "get_plotlyjs", return_value="/*plotlyjs*/"
        ):
            report.render(_sample_data(), DAYS, 900, out)
        assert "Peak 900s bucket avg" in out.read_text()

    def test_single_cluster(self, tmp_path, monkeypatch):
        data = {FA: _sample_data()[FA], "_meta": {}}
        html = self._render(tmp_path, monkeypatch, data)
        assert FA in html and DC not in html


class TestMain:
    def _fetch_stub(self, fetched):
        def fake(name, cfg, forwards, token_provider):
            fetched.append(name)
            return _sample_data()[name]

        return fake

    def test_fetch_path_writes_data_and_renders(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--cluster", FA, "--kubeconfig", "/tmp/a.kc",
             "--cluster", DC, "--kubeconfig", "/tmp/b.kc",
             "--port", "9090", "--port", "9093",
             "--out-dir", str(tmp_path)],
        )
        fetched = []
        with mock.patch.object(report, "fetch_cluster", self._fetch_stub(fetched)):
            report.main()
        assert fetched == [FA, DC]
        payload = json.loads((tmp_path / f"gpu-utilization-{DAYS}d-data.json").read_text())
        assert payload["_meta"]["days"] == DAYS
        assert payload["_meta"]["step_seconds"] == STEP
        assert "window_start" in payload["_meta"]
        assert "window_end" in payload["_meta"]
        assert (tmp_path / f"gpu-utilization-{DAYS}d-report.html").exists()

    def test_fetch_path_creates_output_directory(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "new" / "reports"
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--cluster", FA, "--kubeconfig", "/tmp/a.kc",
             "--out-dir", str(out_dir)],
        )
        with mock.patch.object(report, "fetch_cluster", self._fetch_stub([])):
            report.main()
        assert (out_dir / f"gpu-utilization-{DAYS}d-data.json").exists()
        assert (out_dir / f"gpu-utilization-{DAYS}d-report.html").exists()

    def test_only_fetches_selected_cluster(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--cluster", FA, "--kubeconfig", "/tmp/a.kc",
             "--cluster", DC, "--kubeconfig", "/tmp/b.kc",
             "--port", "9090", "--port", "9093",
             "--only", DC, "--out-dir", str(tmp_path)],
        )
        fetched = []
        with mock.patch.object(report, "fetch_cluster", self._fetch_stub(fetched)):
            report.main()
        assert fetched == [DC]

    def test_only_unknown_cluster_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--cluster", FA, "--kubeconfig", "/tmp/a.kc",
             "--only", "nope", "--out-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit):
            report.main()

    def test_no_fetch_renders_saved_data(self, tmp_path, monkeypatch):
        (tmp_path / f"gpu-utilization-{DAYS}d-data.json").write_text(json.dumps(_sample_data()))
        monkeypatch.setattr(sys, "argv", ["prog", "--no-fetch", "--out-dir", str(tmp_path)])
        report.main()
        assert (tmp_path / f"gpu-utilization-{DAYS}d-report.html").exists()

    def test_no_fetch_uses_saved_step_and_timestamp(self, tmp_path, monkeypatch):
        data = _sample_data()
        data["_meta"]["step_seconds"] = 900
        path = tmp_path / f"gpu-utilization-{DAYS}d-data.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr(sys, "argv", ["prog", "--no-fetch", "--out-dir", str(tmp_path)])
        report.main()
        html = (tmp_path / f"gpu-utilization-{DAYS}d-report.html").read_text()
        assert "Peak 900s bucket avg" in html
        assert "Generated 2023-11-14 22:13 UTC" in html

    def test_no_fetch_rejects_mismatched_step(self, tmp_path, monkeypatch):
        data = _sample_data()
        data["_meta"]["step_seconds"] = 900
        (tmp_path / f"gpu-utilization-{DAYS}d-data.json").write_text(json.dumps(data))
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--no-fetch", "--step", "3600", "--out-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit):
            report.main()

    def test_no_fetch_with_only_filters(self, tmp_path, monkeypatch):
        (tmp_path / f"gpu-utilization-{DAYS}d-data.json").write_text(json.dumps(_sample_data()))
        monkeypatch.setattr(
            sys, "argv", ["prog", "--no-fetch", "--only", FA, "--out-dir", str(tmp_path)]
        )
        report.main()
        html = (tmp_path / f"gpu-utilization-{DAYS}d-report.html").read_text()
        assert FA in html and DC not in html

    def test_no_clusters_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--out-dir", str(tmp_path)])
        with pytest.raises(SystemExit):
            report.main()

    @pytest.mark.parametrize(("flag", "value"), [("--days", "0"), ("--step", "-1")])
    def test_non_positive_window_options_error(self, tmp_path, monkeypatch, flag, value):
        monkeypatch.setattr(
            sys, "argv", ["prog", flag, value, "--out-dir", str(tmp_path)]
        )
        with pytest.raises(SystemExit):
            report.main()
