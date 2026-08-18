#!/usr/bin/env python3
"""Summarize loop_bench runs: median cand/base and cand/hashbrown per group.

Usage: summarize_loop.py loop-r1.txt [loop-r2.txt ...]

cand = the live engine (with the kept optimizations); base = the frozen
pre-optimization snapshot. cand/base medians > ~1.03 flag an architecture
where a kept optimization regresses (see docs/findings.md §7).
"""
import re
import statistics
import sys

UNIT = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}

def parse(path):
    res, cur = {}, None
    for line in open(path):
        m = re.match(r"^([a-z_0-9]+)/([a-z_0-9]+)", line)
        if m and "Benchmarking" not in line:
            cur = (m.group(1), m.group(2))
        m = re.search(r"time:\s+\[[\d.]+\s+\S+\s+([\d.]+)\s+(\S+)\s+", line)
        if m and cur:
            res[cur] = float(m.group(1)) * UNIT[m.group(2)]
            cur = None
    return res

def main():
    runs = [parse(p) for p in sys.argv[1:]]
    if not runs:
        print("usage: summarize_loop.py <loop-run.txt>...", file=sys.stderr)
        return 1
    groups = sorted({g for r in runs for (g, _) in r})
    print(f"    {'group':<22}{'cand/base':>10}{'cand/hb':>10}  (medians over {len(runs)} run(s))")
    worst = None
    for g in groups:
        cb = [r[(g, "cand")] / r[(g, "base")] for r in runs
              if (g, "cand") in r and (g, "base") in r]
        ch = [r[(g, "cand")] / r[(g, "hashbrown")] for r in runs
              if (g, "cand") in r and (g, "hashbrown") in r]
        if not cb:
            continue
        mcb = statistics.median(cb)
        mch = statistics.median(ch) if ch else float("nan")
        flag = "  <-- REGRESSION?" if mcb > 1.03 else ""
        print(f"    {g:<22}{mcb:>10.3f}{mch:>10.3f}{flag}")
        if worst is None or mcb > worst[1]:
            worst = (g, mcb)
    if worst and worst[1] > 1.03:
        print(f"    WARNING: {worst[0]} cand/base median {worst[1]:.3f} — an M5-kept"
              f" optimization may regress on this architecture.")
    else:
        print("    OK: no optimization regression on this architecture.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
