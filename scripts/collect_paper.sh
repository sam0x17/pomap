#!/usr/bin/env bash
# One-shot PER-MACHINE collection of the complete paper dataset, spec-named.
# Run on an idle box (governor=performance, turbo off where controllable).
#
#   scripts/collect_paper.sh            (~60-90 min)
#
# Produces (repo root; copy them all back):
#   loop-<spec>-r{1,2,3}.txt    STEP 1 — optimization A/B: live engine vs the
#                               frozen PRE-optimization snapshot, in-run.
#                               THE x86 REGRESSION GUARD: cand/base > ~1.03
#                               on any group means an M5-kept optimization
#                               regresses on this architecture.
#   bench-<spec>-g4.txt         main suite @ GROWTH=4
#   bench-<spec>-g2.txt         GROWTH=2 insert_allocate + memory
#   cold-<spec>.csv             cold-access matrix
#   perf-<spec>.csv             hardware counters (Linux only)
#   family-<spec>.txt           all seven engine designs, one harness
#   string-<spec>.txt           String-payload main suite
#   iter-<spec>-r{1,2,3}.txt    whole-map iteration vs hashbrown/std/BTreeMap
#   audit-<spec>.txt            methodology probes (thermal state, RNG dilution)
#
# Summaries print as steps finish (python3 optional but recommended).
set -uo pipefail   # NOT -e: optional steps (perf) must not abort the rest
cd "$(dirname "$0")/.."

# --- spec slug (same convention as run_bench.sh) ---
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

echo "== collect_paper: ${slug} =="
echo "   pin: ${pin:-none} | rustc: $(rustc --version 2>/dev/null || echo '?')"
echo

echo "==> [1/8] optimization A/B (loop_bench x3) — the cross-arch regression guard"
for i in 1 2 3; do
  $pin cargo bench --bench loop_bench > "loop-${slug}-r${i}.txt" 2>&1 || true
done
if command -v python3 >/dev/null 2>&1; then
  python3 scripts/summarize_loop.py "loop-${slug}-r1.txt" "loop-${slug}-r2.txt" "loop-${slug}-r3.txt" || true
fi
echo

echo "==> [2/8] main suite, both growth factors (run_bench.sh)"
scripts/run_bench.sh || echo "    run_bench.sh failed (continuing)"
echo

echo "==> [3/8] cold-access matrix (run_cold.sh)"
scripts/run_cold.sh || echo "    run_cold.sh failed (continuing)"
echo

echo "==> [4/8] perf counters (perf_cold.sh; Linux only)"
if command -v perf >/dev/null 2>&1; then
  scripts/perf_cold.sh || echo "    perf_cold.sh failed (continuing)"
else
  echo "    perf not present — skipped"
fi
echo

echo "==> [5/8] family bench (7 engines, ~25 min)"
$pin cargo bench --bench family_bench > "family-${slug}.txt" 2>&1 || true
grep -c "sanity ok" "family-${slug}.txt" | xargs -I{} echo "    sanity gates passed: {}/7"
echo

echo "==> [6/8] String-payload suite"
$pin cargo bench --bench pomap_bench --features bench-string > "string-${slug}.txt" 2>&1 || true
echo

echo "==> [7/8] iteration bench x3"
for i in 1 2 3; do
  $pin cargo bench --bench iter_bench > "iter-${slug}-r${i}.txt" 2>&1 || true
done
echo

echo "==> [8/9] set-algebra + bulk-build bench (union_bench)"
$pin cargo bench --bench union_bench > "union-${slug}.txt" 2>&1 || true
echo

echo "==> [9/9] methodology probes (audit_bench)"
$pin cargo bench --bench audit_bench > "audit-${slug}.txt" 2>&1 || true
echo

echo "== DONE — copy these back: =="
ls -1 "loop-${slug}"-r*.txt "bench-${slug}"-g*.txt "cold-${slug}.csv" \
      "family-${slug}.txt" "string-${slug}.txt" "iter-${slug}"-r*.txt \
      "union-${slug}.txt" "audit-${slug}.txt" 2>/dev/null
ls -1 "perf-"*.csv 2>/dev/null | tail -1
