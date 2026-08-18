# Benchmark results archive

Spec-named raw outputs, one set per machine. Naming:

- `bench-<spec>-g{4,2}[-date].txt` — main suite (`pomap_bench`), both growth factors
- `cold-<spec>[-date].csv` — cold-access matrix (`cold.rs`)
- `perf-<spec>.csv` — hardware counters (`cold_perf.rs`, Linux)
- `bench-{family,string,iter,union,dial,audit}-<spec>-<date>.txt` — supplementary suites
- `loop-<spec>-r{1..3}.txt` — optimizer A/B runs (cross-arch regression guard)

Undated files are the June 2026 canonical cross-machine set; dated files are
later re-runs (see docs/findings.md for which numbers are quoted where, and
for the caveats attached to each run). Collected via `scripts/collect_paper.sh`;
file new runs here.
