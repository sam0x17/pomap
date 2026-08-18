#!/usr/bin/env bash
# Capture the main `pomap_bench` suite for this machine at BOTH growth factors,
# spec-named, so each architecture's dataset is complete.
#
#   bench-<spec>-g4.txt   full suite, GROWTH=4 (default)
#   bench-<spec>-g2.txt   GROWTH=2 — only insert_allocate + memory_footprint
#                         (everything else provisions via with_capacity and is
#                          growth-independent, so re-running it would be identical;
#                          set FULL=1 to run the whole suite at G2 anyway).
#
# Pairs with scripts/run_cold.sh (cold matrix) and scripts/perf_cold.sh (counters),
# both of which are growth-independent — run them once each.
#
# Usage: scripts/run_bench.sh        # both growth factors
#        FULL=1 scripts/run_bench.sh # also run the full suite at G2 (redundant)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -r /proc/cpuinfo ]; then
  cpu_raw="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  cores="$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo)"
  ram_gb="$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"
elif command -v sysctl >/dev/null 2>&1; then
  cpu_raw="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  cores="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
  ram_gb="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
else
  cpu_raw="$(hostname)"; cores='?'; ram_gb='?'
fi
cpu="$(printf '%s' "$cpu_raw" | sed -E 's/\((R|TM)\)//g; s/[0-9]+-Core//; s/ Processor//; s/ CPU//; s/@.*//' | tr -s ' ' | sed 's/^ *//; s/ *$//')"
slug="$(printf '%s_%sc_%sGB' "$cpu" "$cores" "$ram_gb" | tr ' /' '__' | tr -cd '[:alnum:]._-')"

pin=""; command -v taskset >/dev/null 2>&1 && pin="taskset -c 2"
echo "Hygiene: pin (${pin:-none}), governor=performance, turbo off, idle box for stable numbers."

g4="bench-${slug}-g4.txt"
g2="bench-${slug}-g2.txt"

echo "[1/2] GROWTH=4 (default): full suite -> ${g4}  (~10 min)"
$pin cargo bench --bench pomap_bench >"${g4}" 2>&1 || true

if [ "${FULL:-0}" = "1" ]; then
  echo "[2/2] GROWTH=2: full suite -> ${g2}"
  $pin cargo bench --bench pomap_bench --features growth2 >"${g2}" 2>&1 || true
else
  echo "[2/2] GROWTH=2: insert_allocate + memory only -> ${g2}  (rest is growth-independent)"
  $pin cargo bench --bench pomap_bench --features growth2 -- 'insert_allocate|memory_footprint' >"${g2}" 2>&1 || true
fi

echo "wrote ${g4} ${g2}"
echo "extract: awk '/^[a-z_]+\\/[a-z0-9_]+/{n=\$1} /time:/{for(i=1;i<=NF;i++)if(\$i==\"time:\"){print n\"\\t\"\$(i+2)\" \"\$(i+3);break}}' ${g4} | grep -v report"
