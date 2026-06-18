#!/usr/bin/env bash
# Bind the cold-read wall-clock ratios to hardware counters via `perf stat`
# (Linux only). Runs the minimal `cold_perf` driver under perf for pomap /
# hashbrown / std × {get_hit, get_miss}, normalizes counters to per-op, and writes
# perf-<cpu>_<Nc>.csv plus a human-readable table.
#
# Tunables (env): VW=value_words(1|2|4|8)  WS=ws_mb  PASSES=passes  EVENTS=...
# Usage: scripts/perf_cold.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v perf >/dev/null 2>&1 || {
  echo "perf not found. Install linux-tools (e.g. apt install linux-tools-\$(uname -r))."
  exit 1
}
# perf_event_paranoid > 1 blocks unprivileged `perf stat` hardware counters.
par="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)"
if [ "$par" -gt 1 ] 2>/dev/null && [ "$(id -u)" != 0 ]; then
  echo "WARNING: perf_event_paranoid=${par} will block hardware counters."
  echo "  Fix (one of): sudo sysctl kernel.perf_event_paranoid=1"
  echo "                sudo $0          # run the whole script as root"
  echo "Continuing — but expect empty counter rows if it is not lowered."
  echo
fi

VW="${VW:-1}"           # value words: 1=8B 2=16B 4=32B 8=64B
WS="${WS:-256}"         # working-set MB (must be >> LLC to stay cold across passes)
PASSES="${PASSES:-30}"  # cold passes (build overhead ~ 1/PASSES)
# Generic events map to AMD PMU. For misaligned/split loads on AMD add e.g.
# ls_misal_loads.ma64 (check `perf list | grep -i misal`); left out by default so
# an unsupported name can't suppress the rest.
EVENTS="${EVENTS:-instructions,cycles,cache-misses,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-load-misses}"

cpu_raw="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- || echo cpu)"
cpu="$(printf '%s' "$cpu_raw" | sed -E 's/\((R|TM)\)//g; s/[0-9]+-Core//; s/ Processor//; s/ CPU//; s/@.*//' | tr -s ' ' | sed 's/^ *//; s/ *$//')"
cores="$(nproc 2>/dev/null || echo '?')"
slug="$(printf '%s_%sc' "$cpu" "$cores" | tr ' /' '__' | tr -cd '[:alnum:]._-')"
out="perf-${slug}.csv"

cargo bench --bench cold_perf --no-run >/tmp/pb.$$ 2>&1 || { cat /tmp/pb.$$; exit 1; }
bin="$(sed -n 's/.*(\(target[^)]*cold_perf[^)]*\)).*/\1/p' /tmp/pb.$$ | tail -1)"
[ -x "$bin" ] || bin="$(find target -type f -name 'cold_perf-*' ! -name '*.d' -perm -u+x 2>/dev/null | head -1)"
[ -x "$bin" ] || { echo "could not locate cold_perf binary"; exit 1; }

echo "# cpu: ${cpu_raw}"  | tee "$out"
{ echo "# cores: ${cores}"; echo "# config: value_words=${VW} ws_mb=${WS} passes=${PASSES}";
  echo "# events: ${EVENTS}"; echo "# date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } | tee -a "$out"
echo "impl,op,value_words,ws_mb,ops,instr_per_op,cyc_per_op,ipc,l1miss_per_op,llcmiss_per_op,dtlbmiss_per_op,cachemiss_per_op" >> "$out"

printf "\n%-10s %-9s %11s %10s %10s %10s %6s\n" impl op LLCmiss/op L1miss/op dTLB/op instr/op IPC
pin=""; command -v taskset >/dev/null 2>&1 && pin="taskset -c 2"
for imp in pomap hashbrown std; do
  for op in get_hit get_miss; do
    perf stat -x, -e "$EVENTS" -o /tmp/perf.$$.csv $pin "$bin" "$imp" "$op" "$VW" "$WS" "$PASSES" \
      >/tmp/run.$$.out 2>/dev/null || true
    ops="$(sed -n 's/^ops=//p' /tmp/run.$$.out)"
    [ -n "${ops:-}" ] || { printf "%-10s %-9s   (no output)\n" "$imp" "$op"; continue; }
    awk -F, -v ops="$ops" -v imp="$imp" -v op="$op" -v vw="$VW" -v ws="$WS" -v out="$out" '
      $1 ~ /^[0-9]+(\.[0-9]+)?$/ { v[$3]=$1 }
      END {
        ins=v["instructions"]; cyc=v["cycles"];
        ipc = (cyc>0)? ins/cyc : 0;
        l1=v["L1-dcache-load-misses"]/ops; llc=v["LLC-load-misses"]/ops;
        dt=v["dTLB-load-misses"]/ops; cm=v["cache-misses"]/ops;
        printf "%-10s %-9s %11.3f %10.3f %10.3f %10.1f %6.2f\n", imp, op, llc, l1, dt, ins/ops, ipc;
        printf "%s,%s,%s,%s,%s,%.1f,%.1f,%.3f,%.4f,%.4f,%.4f,%.4f\n",
               imp,op,vw,ws,ops, ins/ops, cyc/ops, ipc, l1, llc, dt, cm >> out;
      }' /tmp/perf.$$.csv
  done
done
rm -f /tmp/perf.$$.csv /tmp/run.$$.out /tmp/pb.$$
echo
echo "wrote ${out}"
