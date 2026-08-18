# Benchmark results archive

Spec-named raw outputs, one set per machine. Naming:

- `bench-<spec>-g{4,2}[-date].txt` — main suite (`pomap_bench`), both growth factors
- `cold-<spec>[-date].csv` — cold-access matrix (`cold.rs`)
- `perf-<spec>.csv` — hardware counters (`cold_perf.rs`, Linux)
- `bench-{family,string,iter,union,dial,audit}-<spec>-<date>.txt` — supplementary suites
- `loop-<spec>-r{1..3}.txt` — optimizer A/B runs (cross-arch regression guard)

Layout:

- `archive-2026-06/` — the June 2026 canonical cross-machine set (M5, M3,
  EPYC 9354P bare-metal, EPYC 9845 VM). These are the numbers the current
  docs/findings.md tables quote, measured on the pre-audit engine. Kept as
  the provenance record; superseded table-by-table as the 2026-08 pristine
  collection lands.
- top level — post-2026-08-18 runs of the current engine (dated). The
  `2026-08-18` M5 files were collected under background load / battery and
  are labeled soft in findings; they retire when a pristine M5 run replaces
  them.

Collected via `scripts/collect_paper.sh`; file new runs at the top level.
