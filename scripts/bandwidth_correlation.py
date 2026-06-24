#!/usr/bin/env python3
"""Correlate per-platform cold/large-write ratios against single-thread DRAM
bandwidth and latency — the empirical spine of findings.md C2.

Thesis (§5.3): cold/large writes win where single-thread bandwidth is SCARCE
(PoMap streams full cache lines; hashbrown scatter-rehashes partial ones). The
prediction is a NEGATIVE correlation between bandwidth and PoMap's edge: higher
per-core bandwidth -> write ratio (pomap/hb) closer to or above 1.0 (less edge).
Counter-hypothesis to kill: raw DRAM latency. If latency drove it, EPYC (lower
latency than the M3) should show LESS edge, not more.

Inputs (auto-discovered, globbed so a later Zen 5c run is picked up for free):
  bench-<slug>-g4.txt   -> DRAM bandwidth (384 MiB triad row) + latency (128 MiB row)
  cold-<slug>.csv       -> cold-write ratios at the 128 MB working set

Usage:  python3 scripts/bandwidth_correlation.py [--value-bytes 8] [--ws 128]
Pure stdlib; no numpy/scipy.
"""
import argparse, csv, glob, os, re, sys
from math import sqrt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slug_from_bench(path):
    b = os.path.basename(path)
    m = re.match(r"bench-(.+)-g4\.txt$", b)
    return m.group(1) if m else None


def slug_from_cold(path):
    b = os.path.basename(path)
    m = re.match(r"cold-(.+)\.csv$", b)
    return m.group(1) if m else None


def parse_mountains(path):
    """Return (dram_bw_gbps, dram_lat_ns) from a bench g4 file; None if absent."""
    bw = lat = None
    txt = open(path, encoding="utf-8", errors="replace").read().splitlines()
    mode = None
    for line in txt:
        if "triad GB/s" in line:
            mode = "bw"; continue
        if "latency ns" in line:
            mode = "lat"; continue
        if mode == "bw":
            # last numeric column of the "384 MiB (DRAM)" row is the DRAM plateau
            if line.startswith("384 MiB"):
                bw = float(line.split()[-1]); mode = None
        elif mode == "lat":
            if line.startswith("128 MiB"):
                lat = float(line.split()[-1]); mode = None
    return bw, lat


def cold_ratios(path, value_bytes, ws):
    """pomap/hb ratios at (value_bytes, ws_mb=ws) for get_hit/insert/remove."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        rdr = csv.reader(r for r in f if not r.startswith("#"))
        header = next(rdr)
        idx = {name: i for i, name in enumerate(header)}
        for row in rdr:
            if not row or len(row) < len(header):
                continue
            try:
                vb = int(float(row[idx["value_bytes"]]))
                wsv = float(row[idx["ws_mb"]])
            except (ValueError, KeyError):
                continue
            if vb != value_bytes or abs(wsv - ws) > 1e-6:
                continue
            op = row[idx["op"]]
            pm = float(row[idx["pomap_ns"]]); hb = float(row[idx["hashbrown_ns"]])
            if hb > 0:
                out[op] = pm / hb
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sqrt(sum((x - mx) ** 2 for x in xs))
    dy = sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-bytes", type=int, default=8,
                    help="payload size to read ratios at (default 8 = u64->u64)")
    ap.add_argument("--ws", type=float, default=128.0,
                    help="cold working-set MB to read ratios at (default 128)")
    ap.add_argument("--dir", default=HERE)
    args = ap.parse_args()

    mountains = {}
    for p in glob.glob(os.path.join(args.dir, "bench-*-g4.txt")):
        s = slug_from_bench(p)
        if s:
            mountains[s] = parse_mountains(p)
    colds = {}
    for p in glob.glob(os.path.join(args.dir, "cold-*.csv")):
        s = slug_from_cold(p)
        if s:
            colds[s] = cold_ratios(p, args.value_bytes, args.ws)

    slugs = sorted(set(mountains) | set(colds))
    rows = []
    for s in slugs:
        bw, lat = mountains.get(s, (None, None))
        cr = colds.get(s, {})
        rows.append({
            "slug": s, "bw": bw, "lat": lat,
            "get_hit": cr.get("get_hit"), "insert": cr.get("insert"),
            "remove": cr.get("remove"),
        })

    def fmt(v, w=8, p=2):
        return ("{:>%d.%df}" % (w, p)).format(v) if isinstance(v, float) else ("—".rjust(w))

    print(f"\nCold/large-write correlation  (value={args.value_bytes} B, WS={args.ws:g} MB)")
    print("pomap/hb < 1.0 = PoMap faster.  bw=single-thread DRAM GB/s, lat=DRAM ns\n")
    print(f"{'platform':32} {'bw':>8} {'lat':>8} {'get_hit':>8} {'insert':>8} {'remove':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['slug']:32} {fmt(r['bw'])} {fmt(r['lat'])} "
              f"{fmt(r['get_hit'])} {fmt(r['insert'])} {fmt(r['remove'])}")

    # correlations over platforms that have bandwidth AND the write ratio
    print()
    for op in ("insert", "remove"):
        for drv, key in (("bandwidth", "bw"), ("latency", "lat")):
            pts = [(r[key], r[op]) for r in rows
                   if isinstance(r[key], float) and isinstance(r[op], float)]
            if len(pts) < 3:
                print(f"{op:>6} vs {drv:9}: n={len(pts)} (need >=3; rerun after Zen 5c)")
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            print(f"{op:>6} vs {drv:9}: n={len(pts)}  "
                  f"Pearson r={pearson(xs, ys):+.3f}  Spearman={spearman(xs, ys):+.3f}")
    print("\nPrediction: insert/remove vs bandwidth -> strong POSITIVE r "
          "(more bandwidth, ratio rises toward/above 1 = less PoMap edge);")
    print("vs latency -> weak/no monotonic relation (EPYC has lower latency than "
          "the M3 yet more write edge).")
    print("With 4 platforms this is illustrative, not significant — read the "
          "monotonic ordering, not the p-value.")


if __name__ == "__main__":
    sys.exit(main())
