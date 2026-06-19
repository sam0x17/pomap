#!/usr/bin/env bash
# One-shot: collect the COMPLETE benchmark dataset for THIS machine, spec-named.
# Produces (in the repo root):
#   bench-<spec>-g4.txt / bench-<spec>-g2.txt   main suite @ GROWTH=4 and =2
#   cold-<spec>.csv                             cold-access matrix
#   perf-<spec>.csv                             perf counters (Linux only)
# Copy those files back; the spec-based names already identify the machine.
#
# Run hygiene first (for stable numbers): governor=performance, turbo off, idle box.
#   Linux:  sudo cpupower frequency-set -g performance
# Each sub-step pins itself to a core (taskset -c 2) when available.
#
# Usage: scripts/collect_all.sh      (~20-30 min)
set -uo pipefail   # NOT -e: an optional failing step (perf) must not abort the rest
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/.."

marker="$(mktemp)"
echo "Collecting the full benchmark dataset for this machine (~20-30 min)…"
echo

echo "==> [1/3] main suite, both growth factors  (run_bench.sh)"
"$here/run_bench.sh" || echo "    run_bench.sh failed (continuing)"
echo

echo "==> [2/3] cold-access matrix  (run_cold.sh)"
"$here/run_cold.sh" || echo "    run_cold.sh failed (continuing)"
echo

echo "==> [3/3] perf counters  (perf_cold.sh; Linux only)"
if command -v perf >/dev/null 2>&1; then
  "$here/perf_cold.sh" || echo "    perf_cold.sh failed/blocked (continuing)"
else
  echo "    perf not present — skipped (hardware counters need a Linux box)"
fi

echo
echo "==> files produced this run — copy these back:"
find . -maxdepth 1 -newer "$marker" \
  \( -name 'bench-*-g[24].txt' -o -name 'cold-*.csv' -o -name 'perf-*.csv' \) 2>/dev/null \
  | sort | sed 's|^\./|    |'
rm -f "$marker"
