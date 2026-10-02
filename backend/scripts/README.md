# GPU utilization reporting

`gpu_utilization_14d_report.py` is a standalone adapter around reusable report
generation functions. It is intentionally not imported by the Control Center
API process: Plotly, pandas, and the cluster-query tooling belong in a report
worker or an operator's local environment rather than the production API image.

## Local setup

Install the report-only dependencies from `backend/`:

```bash
python3 -m pip install -r requirements-report.txt
```

The CLI also requires `kubectl` and a kubeconfig for every selected cluster.
Its current token provider creates a 30-minute token for the
`prometheus-k8s` service account in `openshift-monitoring`; therefore the
kubeconfig identity must be authorized to create that service-account token.
Use only a narrowly scoped identity approved for this purpose.

The script creates and owns every local port-forward. It refuses to reuse a
port that is already occupied so a bearer token is never sent to an unknown
local listener.

Example:

```bash
python3 scripts/gpu_utilization_14d_report.py \
  --cluster my-cluster --kubeconfig /path/to/cluster.kubeconfig \
  --port 9090 --hostlabel hostname \
  --days 14 --step 3600 --out-dir /tmp/gpu-reports
```

Re-rendering uses the saved collection metadata, including the original query
step and collection window:

```bash
python3 scripts/gpu_utilization_14d_report.py \
  --no-fetch --days 14 --out-dir /tmp/gpu-reports
```

## Reusable orchestration boundary

The module exposes two orchestration functions:

- `generate_report(...)` accepts resolved cluster configuration, an injectable
  cluster fetcher, and an injectable token provider. This keeps report
  collection testable without coupling the renderer to the CLI's `kubectl`
  process behavior.
- `render_saved_report(...)` recreates an artifact from persisted data while
  preserving the original collection metadata.
