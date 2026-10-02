#!/usr/bin/env python3
"""Build a multi-day GPU utilization report for caller-specified OpenShift clusters.

Pulls DCGM_FI_DEV_GPU_UTIL / DCGM_FI_DEV_FB_USED from each cluster's OpenShift
Prometheus through a local kubectl port-forward and renders a self-contained
Plotly HTML report with cluster-level, per-node, and per-GPU time series.

Cluster topology (name, kubeconfig, port, DCGM hostname label) is entirely
supplied by the caller, e.g.:

  python3 gpu_utilization_report.py \
    --cluster my-cluster-a --kubeconfig /path/a.kc --port 9090 --hostlabel hostname \
    --cluster my-cluster-b --kubeconfig /path/b.kc --port 9093 --hostlabel Hostname \
    --days 14 --step 3600 --out-dir /tmp/reports

  python3 gpu_utilization_report.py --no-fetch   # re-render from saved data
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import socket
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly
import plotly.express as px
from plotly.graph_objects import Figure

ROOT = Path(__file__).resolve().parent

DEFAULT_DAYS = 14
DEFAULT_STEP = 3600
DEFAULT_PORT = 9090
DEFAULT_HOSTLABEL = "hostname"
FB_ACTIVE_THRESHOLD_MIB = 1024
REPORT_SCHEMA_VERSION = 1
PROMETHEUS_LABEL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integer report options."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def port_number(value: str) -> int:
    """Argparse type for valid TCP ports."""
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def build_metadata(days: int, step: int, *, window_end: int | None = None) -> dict:
    """Create immutable collection metadata shared by every cluster query."""
    end = int(time.time()) if window_end is None else int(window_end)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "step_seconds": step,
        "window_start": end - days * 86400,
        "window_end": end,
        "active_threshold_mib": FB_ACTIVE_THRESHOLD_MIB,
    }


def metadata_from_data(data: dict, *, expected_days: int | None = None,
                       expected_step: int | None = None) -> dict:
    """Validate and return metadata used to render a saved report faithfully."""
    meta = data.get("_meta")
    if not isinstance(meta, dict):
        raise ValueError("saved report data is missing _meta")
    required = ("days", "step_seconds", "generated")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"saved report metadata is missing: {', '.join(missing)}")
    try:
        days = int(meta["days"])
        step = int(meta["step_seconds"])
        generated_dt = datetime.fromisoformat(str(meta["generated"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("saved report metadata contains invalid values") from exc
    if days <= 0 or step <= 0:
        raise ValueError("saved report days and step_seconds must be greater than zero")
    if expected_days is not None and expected_days != days:
        raise ValueError(f"saved report uses --days {days}, not {expected_days}")
    if expected_step is not None and expected_step != step:
        raise ValueError(f"saved report uses --step {step}, not {expected_step}")
    normalized = dict(meta)
    normalized["days"] = days
    normalized["step_seconds"] = step
    if generated_dt.tzinfo is None:
        generated_dt = generated_dt.replace(tzinfo=timezone.utc)
    try:
        end = int(normalized.get("window_end", generated_dt.timestamp()))
        start = int(normalized.get("window_start", end - days * 86400))
    except (TypeError, ValueError) as exc:
        raise ValueError("saved report metadata contains an invalid window") from exc
    normalized["window_end"] = end
    normalized["window_start"] = start
    return normalized


def queries_for(hostlabel: str) -> dict:
    return {
        "cluster": "avg(DCGM_FI_DEV_GPU_UTIL)",
        "active": f"count(DCGM_FI_DEV_FB_USED > {FB_ACTIVE_THRESHOLD_MIB})",
        "node": f"avg by ({hostlabel})(DCGM_FI_DEV_GPU_UTIL)",
        "gpu": f"avg by ({hostlabel}, gpu)(DCGM_FI_DEV_GPU_UTIL)",
    }


def node_short(hostname: str) -> str:
    parts = hostname.rsplit("-", 3)
    return "-".join(parts[-2:]) if len(parts) >= 2 else hostname


def parse_clusters(parser: argparse.ArgumentParser, args) -> dict:
    """Validate the repeated --cluster/--kubeconfig/--port/--hostlabel options."""
    if not args.cluster:
        return {}
    n = len(args.cluster)
    if len(args.cluster) != len(set(args.cluster)):
        parser.error("duplicate --cluster name")
    for flag, values in (
        ("--kubeconfig", args.kubeconfig),
        ("--port", args.port),
        ("--hostlabel", args.hostlabel),
    ):
        if values and len(values) != n:
            parser.error(f"{flag} must be given once per --cluster ({n} needed, got {len(values)})")
    if n > 1 and not args.port:
        parser.error("--port is required when passing more than one --cluster")
    kubeconfigs = args.kubeconfig or [None] * n
    ports = args.port or [DEFAULT_PORT] * n
    hostlabels = args.hostlabel or [DEFAULT_HOSTLABEL] * n
    clusters = {}
    for name, kc, port, hostlabel in zip(args.cluster, kubeconfigs, ports, hostlabels):
        if not kc:
            parser.error(f"--kubeconfig is required for cluster {name}")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            parser.error(f"invalid --port for cluster {name}: {port}")
        if not PROMETHEUS_LABEL_RE.fullmatch(hostlabel):
            parser.error(f"invalid --hostlabel for cluster {name}: {hostlabel}")
        clusters[name] = {"kubeconfig": kc, "port": port, "hostlabel": hostlabel}
    return clusters


def ensure_port_available(port: int) -> None:
    """Refuse to reuse an untrusted process already bound to the local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"local port {port} is already in use; choose another --port"
            ) from exc


def stop_forward(proc: subprocess.Popen) -> None:
    """Terminate and reap a managed port-forward process."""
    if proc.poll() is not None:
        proc.wait()
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def start_forward(kc: str, port: int) -> subprocess.Popen:
    ensure_port_available(port)
    proc = subprocess.Popen(
        ["kubectl", "--kubeconfig", kc, "-n", "openshift-monitoring", "port-forward",
         "--address", "127.0.0.1", "svc/prometheus-k8s", f"{port}:9091"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                if proc.poll() is None:
                    return proc
                proc.wait()
                raise RuntimeError(f"port-forward on :{port} exited early")
        except OSError:
            if proc.poll() is not None:
                proc.wait()
                raise RuntimeError(f"port-forward on :{port} exited early")
            time.sleep(1)
    stop_forward(proc)
    raise RuntimeError(f"port-forward on :{port} did not become ready in 45s")


def create_prometheus_token(kubeconfig: str) -> str:
    """CLI token provider; workers may inject a different credential source."""
    return subprocess.run(
        ["kubectl", "--kubeconfig", kubeconfig, "-n", "openshift-monitoring",
         "create", "token", "prometheus-k8s", "--duration=1800s"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def fetch_cluster(name: str, cfg: dict, forwards: dict,
                  token_provider: Callable[[str], str] = create_prometheus_token) -> dict:
    port = cfg["port"]
    if name not in forwards:
        forwards[name] = start_forward(cfg["kubeconfig"], port)
    token = token_provider(cfg["kubeconfig"])
    if not token:
        raise RuntimeError(f"{name}: Prometheus token provider returned an empty token")
    end = cfg.get("window_end", int(time.time()))
    start = cfg.get("window_start", end - cfg["days"] * 86400)
    result = {}
    for key, query in queries_for(cfg["hostlabel"]).items():
        url = (
            f"https://localhost:{port}/api/v1/query_range?"
            + urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": cfg["step"]})
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            payload = json.load(resp)
        if payload.get("status") != "success":
            raise RuntimeError(f"{name} query '{query}' failed: {payload}")
        series = []
        for item in payload["data"]["result"]:
            metric = dict(item["metric"])
            if cfg["hostlabel"] in metric:
                metric["host"] = metric.pop(cfg["hostlabel"])
            vals = [[float(t), None if v == "NaN" else float(v)] for t, v in item["values"]]
            series.append({"metric": metric, "values": vals})
        result[key] = series
        print(f"  {name} {key}: {len(series)} series")
    return result


def cluster_names(data: dict) -> list[str]:
    return [k for k in data if k != "_meta"]


def validate_report_data(data: dict) -> None:
    """Fail with an actionable message instead of crashing during rendering."""
    names = cluster_names(data)
    if not names:
        raise ValueError("report data contains no clusters")
    for name in names:
        cluster_data = data.get(name)
        if not isinstance(cluster_data, dict):
            raise ValueError(f"{name}: cluster data is not an object")
        for key in ("cluster", "active", "node", "gpu"):
            series = cluster_data.get(key)
            if not isinstance(series, list) or not series:
                raise ValueError(f"{name}: Prometheus returned no {key} series")
            if key in ("cluster", "active") and not series[0].get("values"):
                raise ValueError(f"{name}: Prometheus returned no {key} values")
        for key in ("node", "gpu"):
            for series in cluster_data[key]:
                metric = series.get("metric", {})
                if "host" not in metric:
                    raise ValueError(
                        f"{name}: {key} series is missing the configured hostname label"
                    )
        for series in cluster_data["gpu"]:
            if "gpu" not in series.get("metric", {}):
                raise ValueError(f"{name}: GPU series is missing the gpu label")


def summarize(data: dict, days: int, step: int) -> list[dict]:
    validate_report_data(data)
    rows = []
    for name in cluster_names(data):
        util = data[name]["cluster"][0]["values"]
        active = data[name]["active"][0]["values"]
        utils = [v for _, v in util if v is not None]
        counts = [v for _, v in active if v is not None]
        if not utils:
            raise ValueError(f"{name}: cluster utilization contains no numeric values")
        if not counts:
            raise ValueError(f"{name}: active GPU series contains no numeric values")
        gpus = max(len(data[name]["gpu"]), 1)
        gpu_hours = sum(counts) * (step / 3600)
        rows.append({
            "cluster": name,
            "gpus": gpus,
            "avg_util_pct": round(sum(utils) / len(utils), 1),
            "peak_1h_avg_pct": round(max(utils), 1),
            "active_gpu_hours": round(gpu_hours, 0),
            "pct_window_active": round(100 * gpu_hours / (days * 24 * gpus), 1),
        })
    return rows


def step_label(step: int) -> str:
    return f"{step // 3600}h" if step % 3600 == 0 else f"{step}s"


def to_frame(data: dict, key: str, label_fn) -> pd.DataFrame:
    """Flatten raw query results into a long-form DataFrame for plotly express."""
    rows = []
    for name in cluster_names(data):
        for series in data[name][key]:
            label = label_fn(name, series["metric"])
            for t, v in series["values"]:
                if v is None:
                    continue
                rows.append({
                    "cluster": name,
                    "time": datetime.fromtimestamp(t, tz=timezone.utc),
                    "label": label,
                    "value": v,
                })
    return pd.DataFrame(rows, columns=["cluster", "time", "label", "value"])


def chart_cluster(data: dict, days: int, step: int) -> Figure:
    names = cluster_names(data)
    util = to_frame(data, "cluster", lambda name, _m: name)
    active = to_frame(data, "active", lambda name, _m: name)
    fig = px.line(
        util, x="time", y="value", color="cluster",
        labels={"value": "Avg GPU utilization %", "cluster": "", "time": ""},
    )
    fig_active = px.line(
        active, x="time", y="value", color="cluster",
        labels={"value": "", "cluster": "", "time": ""},
    )
    fig_active.for_each_trace(
        lambda t: t.update(yaxis="y2", name=f"{t.name} active GPUs", line_dash="dot")
    )
    fig.add_traces(fig_active.data)
    fig.update_layout(
        title=f"Cluster-level GPU utilization — last {days} days ({step_label(step)} buckets, UTC)",
        yaxis=dict(title="Avg GPU utilization %", range=[0, 105]),
        yaxis2=dict(title="Active GPUs (VRAM > 1 GiB)", overlaying="y", side="right", range=[-2, 40], showgrid=False),
        legend=dict(orientation="h", y=1.12),
        height=420,
    )
    return fig


def chart_node(data: dict, days: int, step: int) -> Figure:
    df = to_frame(data, "node", lambda name, m: f"{node_short(m['host'])} ({name})")
    fig = px.line(
        df, x="time", y="value", color="label",
        labels={"value": "Avg GPU utilization %", "label": "", "time": ""},
        title=f"Per-node average GPU utilization — last {days} days ({step_label(step)} buckets, UTC)",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", y=1.12),
        height=460,
    )
    return fig


def chart_gpu(data: dict, days: int, step: int) -> Figure:
    names = cluster_names(data)
    df = to_frame(data, "gpu", lambda name, m: f"{node_short(m['host'])} gpu{m['gpu']}")
    faceted = len(names) > 1
    fig = px.line(
        df, x="time", y="value", color="label",
        facet_row="cluster" if faceted else None,
        labels={"value": "Avg GPU utilization %", "label": "", "time": ""},
        title=f"Per-GPU average utilization — last {days} days ({step_label(step)} buckets, UTC)",
    )
    if faceted:
        for ax in range(2, len(names) + 1):
            fig.update_layout(**{f"xaxis{ax}_matches": "x", f"xaxis{ax}_visible": False})
        for row, name in enumerate(names, start=1):
            fig.update_yaxes(title_text=f"{name} avg util %", range=[0, 105], row=row, col=1)
        counts = df.groupby("cluster")["label"].nunique().to_dict()
        for a in fig.layout.annotations:
            text = a.text.split("=", 1)[1] if "=" in a.text else a.text
            if text in counts:
                a.text = f"{text} — {counts[text]} GPUs"
                a.font = dict(size=13, weight="bold")
    fig.update_layout(
        legend=dict(orientation="h", y=1.12),
        height=950,
    )
    return fig


def render(data: dict, days: int, step: int, out_path: Path, *,
           generated_at: str | None = None, window_start: int | None = None,
           window_end: int | None = None) -> None:
    summaries = summarize(data, days, step)
    rows_html = "".join(
        "<tr><td>{cluster}</td><td>{gpus}</td><td>{avg_util_pct}%</td><td>{peak_1h_avg_pct}%</td>"
        "<td>{active_gpu_hours:.0f}</td><td>{pct_window_active}%</td></tr>".format(
            **{**row, "cluster": html_lib.escape(str(row["cluster"]))}
        )
        for row in summaries
    )
    generated_dt = (
        datetime.fromisoformat(generated_at)
        if generated_at else datetime.now(timezone.utc)
    )
    if generated_dt.tzinfo is None:
        generated_dt = generated_dt.replace(tzinfo=timezone.utc)
    generated = generated_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    start_ts = window_start if window_start is not None else time.time() - days * 86400
    end_ts = window_end if window_end is not None else time.time()
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    names = cluster_names(data)
    cluster_desc = " and ".join(
        f"<b>{html_lib.escape(str(n))}</b> ({len(data[n]['gpu'])} GPUs)" for n in names
    )
    figures = {
        "c1": chart_cluster(data, days, step),
        "c2": chart_node(data, days, step),
        "c3": chart_gpu(data, days, step),
    }
    plotly_js = plotly.offline.get_plotlyjs()
    figure_scripts = "\n".join(
        f'<div id="{div}" style="margin-bottom: 32px;"></div>\n'
        f'<script>Plotly.newPlot("{div}", {fig.to_json()}, {{responsive: true}});</script>'
        for div, fig in figures.items()
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GPU utilization {days}d report</title>
<script>{plotly_js}</script>
</head><body style="font-family: -apple-system, sans-serif; max-width: 1200px; margin: 24px auto; color: #111;">
<h1>GPU utilization — last {days} days</h1>
<p>Clusters: {cluster_desc}.
Window: {start} to {end}, {step_label(step)} buckets.
"Active GPU" = VRAM usage above {FB_ACTIVE_THRESHOLD_MIB} MiB. Generated {generated}.</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; margin-bottom: 24px;">
<tr style="background:#eee;"><th>Cluster</th><th>GPUs</th><th>Window avg util</th><th>Peak {step_label(step)} bucket avg</th><th>Active GPU-hours</th><th>% of GPU-time active</th></tr>
{rows_html}
</table>
{figure_scripts}
</body></html>"""
    out_path.write_text(html)
    print(out_path)


def generate_report(
    clusters: dict,
    *,
    days: int,
    step: int,
    out_dir: Path,
    fetcher: Callable = fetch_cluster,
    token_provider: Callable[[str], str] = create_prometheus_token,
) -> dict:
    """Collect and render a report through injectable I/O boundaries.

    Callers can supply a cluster-aware ``fetcher`` and token provider while
    reusing the validation, metadata, artifact, and rendering behavior. The
    CLI continues to use managed kubectl port-forwards.
    """
    if days <= 0 or step <= 0:
        raise ValueError("days and step must be greater than zero")
    if not clusters:
        raise ValueError("at least one cluster is required")

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"gpu-utilization-{days}d-data.json"
    out_path = out_dir / f"gpu-utilization-{days}d-report.html"
    metadata = build_metadata(days, step)
    configured_clusters = {
        name: {
            **cfg,
            "days": days,
            "step": step,
            "window_start": metadata["window_start"],
            "window_end": metadata["window_end"],
        }
        for name, cfg in clusters.items()
    }

    forwards: dict[str, subprocess.Popen] = {}
    data = {}
    try:
        for name, cfg in configured_clusters.items():
            print(f"Fetching {name} ...")
            data[name] = fetcher(name, cfg, forwards, token_provider)
        data["_meta"] = metadata
        validate_report_data(data)
        data_path.write_text(json.dumps(data, indent=1))
        print(data_path)
    finally:
        for proc in forwards.values():
            stop_forward(proc)

    render(
        data, days, step, out_path,
        generated_at=metadata["generated"],
        window_start=metadata["window_start"],
        window_end=metadata["window_end"],
    )
    return {"data_path": data_path, "report_path": out_path, "metadata": metadata}


def render_saved_report(
    data_path: Path,
    out_path: Path,
    *,
    only: str | None = None,
    expected_days: int | None = None,
    expected_step: int | None = None,
) -> dict:
    """Render persisted data using its original collection semantics."""
    data = json.loads(data_path.read_text())
    metadata = metadata_from_data(
        data, expected_days=expected_days, expected_step=expected_step
    )
    if only:
        if only not in data:
            raise ValueError(f"--only {only} is not present in saved report data")
        data = {only: data[only], "_meta": metadata}
    validate_report_data(data)
    render(
        data, int(metadata["days"]), int(metadata["step_seconds"]), out_path,
        generated_at=str(metadata["generated"]),
        window_start=metadata.get("window_start"),
        window_end=metadata.get("window_end"),
    )
    return {"data_path": data_path, "report_path": out_path, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", action="append", metavar="NAME",
                        help="cluster name; repeat together with --kubeconfig for multiple clusters")
    parser.add_argument("--kubeconfig", action="append", metavar="PATH",
                        help="kubeconfig path for the matching --cluster")
    parser.add_argument("--port", action="append", type=port_number, metavar="PORT",
                        help="local port for the matching --cluster Prometheus port-forward")
    parser.add_argument("--hostlabel", action="append", metavar="LABEL",
                        help="DCGM hostname metric label for the matching --cluster")
    parser.add_argument("--no-fetch", action="store_true", help="render from saved data JSON")
    parser.add_argument("--only", metavar="NAME", help="fetch/render a single cluster only")
    parser.add_argument("--days", type=positive_int, help=f"look-back window in days (default: {DEFAULT_DAYS})")
    parser.add_argument("--step", type=positive_int, help=f"query step in seconds (default: {DEFAULT_STEP})")
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="directory for data JSON and HTML report")
    args = parser.parse_args()

    days = args.days or DEFAULT_DAYS
    step = args.step or DEFAULT_STEP
    out_dir = args.out_dir
    data_path = out_dir / f"gpu-utilization-{days}d-data.json"
    out_path = out_dir / f"gpu-utilization-{days}d-report.html"

    clusters = parse_clusters(parser, args)

    if args.no_fetch:
        try:
            render_saved_report(
                data_path, out_path, only=args.only,
                expected_days=args.days, expected_step=args.step,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        return

    if not clusters:
        parser.error("at least one --cluster/--kubeconfig pair is required (or use --no-fetch)")
    if args.only:
        clusters = {n: c for n, c in clusters.items() if n == args.only}
        if not clusters:
            parser.error(f"--only {args.only} not among the --cluster names")

    try:
        generate_report(
            clusters, days=days, step=step, out_dir=out_dir,
            fetcher=fetch_cluster,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
