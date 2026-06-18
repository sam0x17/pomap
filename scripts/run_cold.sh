#!/usr/bin/env bash
# Run the cold-access experiment matrix and save machine-collatable CSV with a
# platform header. Pins to a non-zero core when taskset is available.
#
# Usage: scripts/run_cold.sh            # writes cold-<host>.csv
#        scripts/run_cold.sh out.csv    # custom path
set -euo pipefail
cd "$(dirname "$0")/.."

host="$(hostname | tr ' /' '__')"
out="${1:-cold-${host}.csv}"

# ---- platform metadata header (best-effort, portable) ----
{
  echo "# host: ${host}"
  echo "# uname: $(uname -a)"
  if [ -r /proc/cpuinfo ]; then
    echo "# cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    echo "# cores: $(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)"
  elif command -v sysctl >/dev/null 2>&1; then
    echo "# cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "# cores: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi
  echo "# rustc: $(rustc --version 2>/dev/null || echo unknown)"
  echo "# date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out"

# ---- build first (keep compile noise out of the timed run) ----
cargo bench --bench cold --no-run >/dev/null 2>&1

# ---- run, pinned + isolated; CSV (and the column header) go to stdout ----
run="cargo bench --bench cold"
if command -v taskset >/dev/null 2>&1; then
  run="taskset -c 2 ${run}"
fi
# harness stdout is pure CSV; cargo's Compiling/Finished noise is on stderr.
${run} 2>/dev/null >> "$out"

echo "wrote ${out}"
echo "rows: $(grep -c '^[0-9]' "$out")"
