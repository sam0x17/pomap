#!/usr/bin/env bash
# Run the cold-access experiment matrix and save machine-collatable CSV.
# The output filename self-describes the machine: "<cpu>_<cores>c_<ramGB>GB.csv"
# (e.g. AMD_EPYC_9845_8c_16GB.csv, Apple_M3_Max_16c_128GB.csv). Pins to a
# non-zero core when taskset is available.
#
# Usage: scripts/run_cold.sh            # auto spec-based name
#        scripts/run_cold.sh out.csv    # explicit path override
set -euo pipefail
# NOTE: GROWTH-independent — every map is built via with_capacity (no growth), so
# GROWTH=2 and GROWTH=4 give identical results here. The growth dial is captured
# by scripts/run_bench.sh (main suite). No --features growth2 needed.
cd "$(dirname "$0")/.."

# ---- gather platform specs (portable: Linux /proc, else macOS sysctl) ----
if [ -r /proc/cpuinfo ]; then
  cpu_raw="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  cores="$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo)"
  ram_gb="$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"
elif command -v sysctl >/dev/null 2>&1; then
  cpu_raw="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  cores="$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
  mem_b="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  ram_gb="$(( mem_b / 1073741824 ))"
else
  cpu_raw="$(hostname)"; cores='?'; ram_gb='?'
fi

# Clean the CPU string: strip (R)/(TM), "NN-Core", "Processor"/"CPU", "@ freq".
cpu="$(printf '%s' "$cpu_raw" \
  | sed -E 's/\((R|TM)\)//g; s/[0-9]+-Core//; s/ Processor//; s/ CPU//; s/@.*//' \
  | tr -s ' ' | sed 's/^ *//; s/ *$//')"

# Spec slug, safe for filenames.
slug="$(printf '%s_%sc_%sGB' "$cpu" "$cores" "$ram_gb" | tr ' /' '__' | tr -cd '[:alnum:]._-')"
out="${1:-cold-${slug}.csv}"

# ---- platform metadata header ----
{
  echo "# cpu: ${cpu_raw}"
  echo "# cores: ${cores}"
  echo "# ram_gb: ${ram_gb}"
  echo "# uname: $(uname -a)"
  echo "# rustc: $(rustc --version 2>/dev/null || echo unknown)"
  echo "# date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out"

# ---- build first (keep compile noise out of the timed run) ----
cargo bench --bench cold --no-run >/dev/null 2>&1

# ---- run pinned + isolated; harness stdout is pure CSV ----
run="cargo bench --bench cold"
if command -v taskset >/dev/null 2>&1; then
  run="taskset -c 2 ${run}"
fi
${run} 2>/dev/null >> "$out"

echo "wrote ${out}"
echo "rows: $(grep -c '^[0-9]' "$out")"
echo
echo "NOTE: this is the wall-clock matrix. For hardware counters (cache/TLB"
echo "      misses per op, IPC) run the OTHER script:  scripts/perf_cold.sh"
