# Running the project on Windows

Tested on Windows 11 Pro, PowerShell 7, Docker Desktop 4.x (WSL 2 engine), Python 3.12,
Intel Core i7-8565U (4 cores / 8 threads), 24 GB RAM, no GPU.

## Steps

```powershell
py -3.12 -m venv .venv                       # any Python >= 3.10
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[embed,dev]"
.\.venv\Scripts\python.exe -m pytest -q      # unit + end-to-end tests, no Elasticsearch
docker compose up -d elasticsearch
$env:ES_URL = "http://localhost:9200"; .\.venv\Scripts\python.exe -m pytest -q -m integration; Remove-Item Env:ES_URL
```

Or everything at once: `powershell -ExecutionPolicy Bypass -File scripts\run_week1.ps1`.

## Issues met and fixes

| Issue | Cause | Fix |
|---|---|---|
| `py -3.11` not found | only Python 3.10 and 3.12 installed | `scripts/run_week1.ps1` now picks the first of 3.12 / 3.11 / 3.10 listed by `py --list` |
| `Activate.ps1` blocked or not persistent | execution policy; agents and CI run each command in a fresh shell | call `.venv\Scripts\python.exe` / `cer.exe` directly |
| `docker info`: cannot connect to `dockerDesktopLinuxEngine` | WSL was not installed, so Docker Desktop had no Linux engine | `wsl --install` (admin PowerShell), reboot, start Docker Desktop and wait for "Engine running" |
| `docker info` returns HTTP 500 / hangs right after installing WSL | WSL installed but Windows not rebooted yet | reboot |
| Re-running `cer ingest` with the same `index_version` deleted the index the alias was serving | `create_index` dropped any existing index of that name | the served index is now protected: bump `backend.index_version` (an unserved leftover of an interrupted build is still rebuilt) |
| Integration test cleanup failed: `Wildcard expressions or all indices are not allowed` | Elasticsearch 8 sets `action.destructive_requires_name=true` | delete test indexes by explicit name |
| Elasticsearch unreachable the next morning (`cer-es` exited with code 255, no OOM) | Docker Desktop / WSL restarted (sleep, update) and the container had no restart policy | `restart: unless-stopped` in `docker-compose.yml`; indexes live on the `esdata` volume and survive |

## Timings on this machine

| Step | Duration |
|---|---|
| `pip install -e ".[embed,dev]"` (PyTorch CPU, sentence-transformers) | ~22 min (slow connection) |
| `pytest -q` (25 tests) | ~30–95 s |
| first `docker compose up -d elasticsearch` (image pull, slow connection) | ~10 min, then ~30 s to green |
| `pytest -m integration` | ~16 s |
| `cer ingest` NFCorpus (3,633 docs, 4,811 chunks, e5-base on CPU) | 70 min (1.1 chunks/s, 99% embedding), + ~18 min first model download |
| `cer eval` NFCorpus, 3 systems x 323 queries | 150 s |
| `cer ingest` SciFact (5,183 docs, 6,428 chunks) | 86 min (1.2 chunks/s) |
| `cer eval` SciFact, 3 systems x 300 queries | 154 s |
