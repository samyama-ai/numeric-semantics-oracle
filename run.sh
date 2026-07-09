#!/usr/bin/env bash
# One-command reproduction of every number in the paper.
#   ./run.sh
# Requires: docker + docker compose, python3. No cloud, no GPU, no paid resources.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
[ -x "$HOME/projects/venv/bin/python" ] && PY="$HOME/projects/venv/bin/python"

echo "==> installing python deps"
"$PY" -m pip install -q -r requirements.txt

echo "==> starting engines (PostgreSQL 17, MySQL 8.4, ClickHouse 25.3)"
docker compose up -d
for _ in $(seq 1 60); do
  n=$(docker compose ps --format '{{.Health}}' | grep -c healthy || true)
  [ "$n" -ge 3 ] && break
  sleep 3
done
docker compose ps --format '{{.Service}}: {{.Health}}'

mkdir -p results

echo
echo "==> [1/3] classification map + boundaries  (NC2 hard gate: expect 0 anomalies)"
"$PY" src/run_map.py

echo
echo "==> [2/3] E2: condition numbers of real TPC-H aggregate queries"
"$PY" src/e2_realdata.py

echo
echo "==> [3/3] E3: kappa distribution of SQLancer-style random columns"
"$PY" src/e3_fuzzer_kappa.py

echo
echo "==> done. results/ holds map.json, e3.json and the logs."
echo "    stop the engines with: docker compose down"
