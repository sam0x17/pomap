# PoMap: Performance Findings

> **Status: working draft toward an academic paper (v0.1, internal).**
> This document records measured results, their provenance, and the conclusions
> we are confident in versus those that still need confirmation. It is organized
> roughly in paper order so it can grow into a submission. Sections marked
> **[TODO]** are gaps to close before publication. Numbers labelled *soft* are
> not yet trustworthy to the stated precision.

## 1. Thesis

PoMap is a hash map that stores entries in a single flat array kept globally
sorted by hash, with each key's home ("ideal") slot derived from its hash
prefix. This yields two properties a conventional open-addressing table
(hashbrown / Swiss tables) does not have:

1. **Deterministic iteration order** (by hash, independent of insertion order).
2. **Single-cache-line probes**: the 8-byte hash is stored *inline* with its
   key and value (`#[repr(C)] struct Entry { hash: u64, key: K, value: V }`),
   so a lookup is one memory access — scan from the ideal slot, filter on exact
   `u64` comparisons, stop at the first stored hash greater than the target
   (which doubles as the empty-slot sentinel, `u64::MAX`).

The cost is an **order-maintenance tax on writes** (runs shift to stay sorted)
and a larger per-slot footprint (the inline hash).

**Contributions.** We claim three:

- **(C1) The prefix-ordered hash map.** A flat open-addressing table kept
  *globally sorted by hash value*, with each entry's home position given by its
  hash *prefix* (top bits → ideal slot) so that array order *is* hash order,
  displacing to the nearest vacancy to maintain the sort. **The ordering criterion
  is the hash, not the key** — a key comparison is *never* on the layout path
  (verified: the implementation uses only `u64` hash comparisons for navigation
  and `==` for the final match; it contains no `Ord`/`cmp`/`<` on keys, and the
  `Key: Ord` bound is in fact vestigial — functionally only `Hash + Eq + Clone`
  is needed). This is the categorical distinction from prior structured open
  addressing (§10): classical *ordered hashing* orders by **key comparison**, and
  Robin Hood orders by **probe distance**; PoMap orders by **hash**. It yields
  (a) deterministic global iteration in hash order, (b) early-terminating probes
  (stop at the first stored hash past the target — the sort order doubles as the
  negative-lookup cutoff and the empty sentinel), via a single in-line `u64`
  compare rather than a key `cmp`, and (c) an AoS inline-hash layout that makes a
  probe one cache line. We are not aware of this hash-prefix-ordered flat
  open-addressing design in prior work (search not yet complete — §10).
- **(C2) Bandwidth-favorability (empirical).** PoMap's write/bulk operations are
  sequential-streaming and bandwidth-bound; its deficits vs SIMD open addressing
  on laptop/desktop parts *invert into wins* on high-bandwidth server CPUs.
- **(C3) A mapped design space.** A set of measured negative results (§6) that
  justify the specific design choices (AoS over SoA tags, backshift over
  tombstones, memset-then-write over single-pass rebuild).

**Central empirical claim.** PoMap is a *deterministic* hash map whose point-read
and update performance **matches or beats** SwissTable-class tables (hashbrown)
on every microarchitecture tested, and whose bulk/write performance **inverts from
deficit to win as memory bandwidth grows** (laptop/desktop → server). The unifying
mechanism is the **memory hierarchy**, and the advantage is therefore largest
exactly where real large-map workloads live — **cold, low-locality access**:

- *Cold reads.* A PoMap lookup touches **one cache line** — the hash is inline
  with its (K, V), so probe + filter + fetch is a single line (one miss, one TLB
  entry). A SwissTable lookup touches **two**: the control-byte group, then the
  entry in a *separate* array (two independent misses, two TLB entries). When data
  is cold (DRAM/L3-resident, the common case for large maps), halving the misses
  is close to a 2× latency advantage; when everything is hot in L1 the miss
  difference vanishes and the gap narrows to instruction throughput. PoMap's read
  edge thus *widens with working-set size and coldness*; **below the cache
  boundary (warm) hashbrown's SIMD throughput wins** — it is a crossover, measured
  in §5.6, not a blanket read win. (Misses are the exception even cold — §5.6.)
- *Cold writes.* PoMap's inserts/repacks/backshifts are sequential streaming;
  SwissTable growth rehashes (scatter). Cold and bandwidth-bound, sequential wins
  — hence the server-CPU inversion.

We argue cold/low-locality access is *more* representative of real workloads
(large keyspaces, point lookups, low temporal locality) than the cache-resident
hot-loop that microbenchmarks default to — so the regime where PoMap is strongest
is the regime that matters. (This is a positioning argument, defensible but not a
measured fact; §5.6 specifies the cold benchmark that would establish it directly.)

## 2. Design

- **Layout.** One allocation of `MaybeUninit<Entry<K,V>>`, length
  `ideal_range + padding`. `ideal_range` is a power of two; an entry's ideal slot
  is `hash >> (64 - log2(ideal_range))`. Entries are globally sorted by hash.
- **Vacancy sentinel.** `EMPTY_HASH = u64::MAX`. The whole allocation is marked
  empty with a single `memset(0xFF)` (all-ones bytes ⇒ every slot's leading
  `u64` hash is `u64::MAX`).
- **Lookup.** Scan forward from the ideal slot; `stored == hash` triggers a key
  compare, `stored > hash` (which includes `EMPTY_HASH`) terminates.
- **Insert.** If the ideal slot is empty, write directly. Otherwise scan to the
  sorted insertion point and shift the run right into the nearest vacancy with a
  single `memmove`.
- **Remove.** Backshift: pull the trailing displaced run (entries whose ideal
  slot is left of their position) left by one; no tombstones.
- **Growth.** Grow at **62.5% load** (a low load factor keeps runs — hence shift
  and backshift chains — short; it is *usually free* in memory because
  `ideal_range` is a power of two). The growth multiplier is a compile-time
  constant generic `GROWTH` (default **4**, must be a power of two, enforced by
  an associated-const assert). On grow, the repack preserves inter-run gaps via
  **cursor spacing** (advance the write cursor past each vacant source slot),
  leaving ideal slots open so post-grow inserts land directly.
- **Provisioning.** `with_capacity(n)` sizes `ideal_range` so that `n` inserts
  trigger zero grows — matching hashbrown's contract.

`GROWTH` is a measured speed/memory dial: **GROWTH=4** drives insert-with-growth
to parity-or-better at higher memory cost; **GROWTH=2** trades ~1.5× insert
throughput for ~30% less slack memory. Nothing but insert-with-growth depends on
it (all other workloads provision via `with_capacity` and never grow mid-measure).

## 3. Implementation notes worth reporting

- **`repr(C)` on `Entry` is load-bearing.** A plain tuple `(u64, K, V)` has
  *unspecified* field order under `repr(Rust)`; for `(u64, String, String)` the
  compiler placed the hash off offset 0, breaking the raw leading-`u64` reads
  that the whole probe/vacancy scheme depends on. This was latent for the entire
  development period because every benchmark used `u64` payloads (which happened
  to be laid out hash-first). It surfaced only when a `String`-keyed test was
  added. **Lesson for the paper's methods/threats section: type-monomorphic
  microbenchmarks can hide layout bugs; test a non-trivial payload type.**

## 4. Experimental methodology

### 4.1 Platforms

| Tag | CPU | µarch | Environment | rustc | Notes |
|---|---|---|---|---|---|
| M-series | Apple M3 Max (Mac15,9), 16-core (12P+4E), 128 GB unified | Apple M3 | macOS, AC power | 1.96.0 | dev machine; remove anchors *soft* (see 4.4) |
| Zen 4 | AMD EPYC 9354P | Zen 4 | Linux, near-bare-metal | 1.90 | first x86 reference |
| Zen 5c | AMD EPYC 9845 "Turin Dense" | Zen 5c | Linux, **virtualized 8-vCPU guest**, AVX-512 (vp2intersect/vaes/gfni) | **[TODO]** | host-managed turbo/governor; ~3% noise floor |

Single-threaded streaming-throughput (STREAM-triad) curves, captured by the
`bench_bandwidth` probe in the harness (working set = 3 arrays of f64), give the
per-platform bandwidth x-axis for §5.3. M3 Max (single core, `taskset`/idle):
~226 GB/s in L1, ~142 GB/s in L3 (12 MiB WS), ~100 GB/s in DRAM (384 MiB WS).
**[TODO]** capture the same curve on Zen 4 and Zen 5c (the probe ships in the
bench; just run it) — those two numbers are what turn §5.3 from a 3-point trend
into a throughput correlation.

**[TODO]** Zen 5c rustc version.

### 4.2 Software

- Rust (edition 2024), `release` profile: `opt-level=3`, `lto="fat"`,
  `codegen-units=1`.
- Hasher: `ahash` (`BuildHasherDefault<AHasher>`) for all maps, so the hash
  function is held constant across implementations.
- Baselines: `hashbrown` 0.14 and `std::collections::HashMap`. (Note: std bundles
  its *own* hashbrown version; the two are **not** identical — see 4.4.)
- Payloads: `u64 → u64` (a `String` variant exists behind a feature flag).
- Harness: `criterion` (a fork adding comparison-groups that print each
  implementation's ratio to the in-group winner), pinned to git rev
  `441d4c65` of `github.com/sam0x17/criterion.rs` (branch `master`).

### 4.3 Workloads (9 microbenchmarks)

`insert_allocate` (build from empty, with growth), `insert_preallocated` (build
into a pre-sized map, no growth), `get_hits`, `get_misses`, `update_existing`,
`get_hotset` (Zipf-ish hot subset of size √N), `remove_hits`, `remove_misses`,
`shrink_to`. Get-class workloads scale to 1M entries; insert/remove/shrink-class
to 100k; 50 evenly-spaced sizes per group.

### 4.4 Methodology lessons (these are paper "threats to validity" material)

- **In-run normalization is mandatory.** Absolute timings drift ≥20% between
  runs (thermal, scheduler, power state). Only ratios measured *within the same
  process* against a baseline run back-to-back are trustworthy. We normalize to
  `hashbrown = 1.00`.
- **Deallocation-timing artifact (fixed).** The remove/shrink benchmarks clone
  maps per batch; an earlier harness consumed the clones inside the timed
  closure, so *freeing* the maps was timed. On Linux (munmap + TLB shootdowns)
  this inflated `remove_hits` ~60× and silently ranked implementations by
  allocation size rather than remove cost. Fixed by returning the batch from the
  closure. **All remove/shrink numbers below are post-fix.**
- **Baseline anchor instability.** `std::HashMap` and `hashbrown` 0.14 diverge on
  removes by ~2× on some platforms (different bundled hashbrown). On M-series the
  remove anchor is unstable run-to-run; **M-series remove ratios are reported as
  *soft*.** Sanity check: per-op cost (timed ÷ ops) should land in the tens of ns.
- **VM noise floor.** On the shared Zen 5c guest the insert-group noise floor is
  ~3% (a neighbor spike can skew one implementation's measurement window).
  Sub-3% effects are not validatable there; large-workload groups are stable to
  <1%.

### 4.5 Statistical reporting

criterion samples each benchmark 100 times by default (3 s warm-up, ≥5 s
measurement) and reports a bootstrapped estimate as `[lower point upper]` — a 95%
confidence interval on the estimate, not a min/median/max. We have been quoting
the **point estimate** (the middle value); the ratio tables should be read as
point-estimate ratios. For publication:

- **Report the CI, not just the point.** Extract all three bounds (the `[lo pt hi]`
  triple) so each ratio can carry an interval; a difference whose intervals
  overlap the noise floor is not a result. The §8 extractor below captures all
  three.
- **Document run counts.** Each platform table here is from ≥2 full runs (Zen 5c
  additionally took a 3rd targeted sample on the noisy `remove_hits` group). State
  the exact count and machine state per table.
- **Size points.** 50 evenly-spaced sizes per group; a stronger paper would show
  per-size curves (not just the aggregate) to expose where ratios cross (e.g. the
  `remove_misses` cache-resident→memory-bound crossover in §5.4).

### 4.6 Cold-access experiment matrix

`benches/cold.rs` (run via `scripts/run_cold.sh`) sweeps three axes to map the
warm→cold crossover and how it moves with the design's cost drivers:

- **value size** `V ∈ {8, 16, 32, 64}` B (key always `u64`). Larger values shrink
  PoMap's relative 8-byte inline-hash overhead and change entries-per-line for
  both maps — testing whether the read edge and the memory cost move with payload.
- **working-set bytes**, log-spaced 64 KiB → 128 MiB (L1 → well past any single
  LLC). Sizes are chosen by *target bytes*, not entry count, so the crossover
  aligns across value sizes and across machines with different caches.
- **operation** ∈ {get_hit, get_miss, insert, remove}.

Coldness is forced per point: build → evict caches (256 MiB stream) → time one
**random-order single pass** (each key once, no reuse). Output is CSV
(`value_bytes,op,n,ws_mb,pomap_ns,hashbrown_ns,std_ns`) with a platform-metadata
header, one file per host (`cold-<host>.csv`) for collation. The run is pinned
(`taskset -c 2`) and takes a few minutes.

Caveat carried in the harness header: single-map `get_miss` is PoMap-pessimistic
(hashbrown's 1-byte control array warms during a single-map pass); the *fair*
cold-miss is the main suite's multi-map-sweep `get_misses`. Treat matrix
`get_miss` rows as a lower bound on PoMap's miss competitiveness.

**Run on every platform** (M3 Max, Zen 4 9354P, Zen 5c 9845), idle box:
```
scripts/run_cold.sh        # → cold-<host>.csv
```
then collate the per-host CSVs. **[TODO]** capture Zen 4 + Zen 5c; the crossover
working-set should track each platform's LLC size, and the high-bandwidth Zen 5c
should show a larger, earlier cold win.

## 5. Results

### 5.1 Cross-platform normalized ratios (pomap ÷ hashbrown, GROWTH=4)

Lower is better; `<1.0` means PoMap beats hashbrown. Each column is normalized
within that platform's own run.

| Workload | M-series | Zen 4 (9354P) | Zen 5c (9845) |
|---|---|---|---|
| get_hits | 0.65 | 0.64 | **0.51** |
| get_misses | 0.83 | 0.94 | 0.87 |
| get_hotset | 0.68 | 0.52 | **0.50** |
| update_existing | 0.62 | 0.47 | **0.32** |
| remove_hits | ~1.2 *(soft)* | 0.98 | **0.58** |
| remove_misses | 0.89 | 1.16 | 1.13 |
| insert_allocate | 1.08 | 1.09 | **0.81** |
| insert_preallocated | 1.66–1.81 | 1.47 | **1.14** |
| shrink_to | 1.11 | 1.17 | **0.38** |
| **groups won (<1.0)** | 6 / 9 | 6 / 9 *(remove_hits ≈ parity)* | **7 / 9** |

Bolded Zen 5c cells are the **inversions**: operations that are deficits on the
lower-bandwidth parts become wins on the high-bandwidth server CPU.

### 5.2 Reads and updates

PoMap wins all three get workloads and `update_existing` on every platform in the
*aggregate* over the standard suite's `evenly_spaced` sizes (often ~2×, e.g.
`update_existing` 0.32× on Zen 5c). This is the inline-hash AoS locality advantage
— one cache line per probe versus hashbrown's control-byte array plus a separate
entry slab — and it widens with bandwidth. **Caveat:** that aggregate is
dominated by the larger (colder) sizes; the per-size sweep (§5.6) shows the win is
a cold/large-working-set phenomenon with a warm-regime crossover, and that
*misses* are a genuine weak spot. Read §5.2 and §5.6 together.

### 5.3 The write/bulk inversion (key result)

`shrink_to`: 1.11× (M) → 1.17× (Zen 4) → **0.38×** (Zen 5c) — PoMap shrinks 2.6×
*faster* than hashbrown on the high-BW part. `remove_hits` and `insert_allocate`
move the same direction (to 0.58× and 0.81×). Mechanism: PoMap's repack/backshift
are sequential streaming writes; hashbrown rehashes (scatter). When bandwidth is
the binding constraint, sequential wins. **We previously (wrongly) characterized
shrink_to's deficit as a fixed ~1.55× "intrinsic" cost; the three-platform trend
shows it is microarchitecture-dependent and bandwidth-driven.**

### 5.4 The two residual losses

- `insert_preallocated` (1.14–1.81×): the pure per-insert shift tax, with no
  growth to amortize. Narrows with bandwidth but does not invert. A controlled
  probe on Zen 5c (removing scan-comparison work *regressed* it) indicates inserts
  are **shift/memory-bound, not scan-bound** — see §6.
- `remove_misses` (0.89–1.16×): only a loss on the small 100k cache-resident
  workload, where hashbrown's SIMD group-probe beats a linear miss-scan. The
  *same* miss-scan **wins** at 1M (`get_misses`), because there it is
  memory-bound. This is the cache-resident/latency-bound vs bandwidth-bound axis
  in miniature.

### 5.5 Memory

Per-slot, PoMap stores 24 B for `u64/u64` (8 B inline hash + 8 + 8) versus
hashbrown's ~17 B (1 control byte + 8 + 8), and runs at a lower load factor. The
harness now measures the **real retained heap footprint** via a tracking global
allocator (a build-from-empty, so the growth-step geometry is exercised as in
real use). Measured bytes-per-entry, M3 Max, GROWTH=4:

| entries | PoMap B/ent | hashbrown B/ent | PoMap / hb |
|---|---|---|---|
| 500 | 104.1 | 34.8 | 2.99× |
| 5,000 | 40.0 | 27.9 | 1.44× |
| 50,000 | 63.0 | 22.3 | 2.83× |
| 500,000 | 100.7 | 35.7 | 2.82× |
| 5,000,000 | 40.3 | 28.5 | 1.41× |

The ratio is **lumpy (1.4–3.0×)**, governed entirely by where N lands relative to
a 4× growth boundary: ~1.4× near full (≈60% load, e.g. 5M), ~2.8× just after a
grow (≈24% load, e.g. 500k). This is the GROWTH=4 sparsity cost made concrete,
and it confirms the earlier analytic estimate. GROWTH=2 roughly halves the
post-grow sparsity (averaging ~2.0×). Note the table reflects `with_hasher`
(build-from-empty, the *pessimistic* footprint); `with_capacity(n)` provisions to
~62.5% load and lands tighter. Memory is the design's real cost — the price of
the inline 8-byte hash plus the low load factor that buys the read/write wins.

*(Fixed: the report previously printed `capacity()` (the logical threshold)
rather than measured bytes; the numbers above are the corrected, allocator-measured
footprint.)*

### 5.6 Cold access — per-size sweep (first results, M3 Max)

Measured by `benches/cold.rs`: per size (uniform `evenly_spaced`, 16k→4M), build
the map, evict caches (256 MB stream), then time a single **random-order** pass
(each key once, no reuse). This reveals a **warm→cold crossover** and refines the
earlier "PoMap wins reads everywhere" into something more precise and honest.

| op | small / warm (<=629k, cache-resident) | large / cold (>=1.5M, DRAM) |
|---|---|---|
| get_hit | **loses** 1.4-1.8x | **wins** ~0.76x (0.65-0.87) |
| insert | loses 1.4-1.7x | ~parity-win (0.83-1.1x) |
| remove | ~parity | small win (0.88-0.96x) |
| get_miss | loses ~2x | loses ~1.5x (see caveat) |

**It is a crossover, not blanket dominance.** Cache-resident (warm) →
hashbrown's SIMD group-probe wins on throughput; exceeding cache (cold/DRAM — the
regime for any large map) → PoMap's single-cache-line probe wins on hits, writes
pull to parity-or-better. Crossover on M3 Max ≈ 1M entries (~24 MB ≈ L2/SLC
boundary); it should **scale with the platform's cache size**. The main suite's
`evenly_spaced(10,1M)` aggregate is dominated by its larger (colder) points,
which is why it reported PoMap winning reads — consistent, but the crossover is
the truer statement.

**Misses are PoMap's genuine weak spot, and cold-miss measurement is subtle.** A
SwissTable miss usually resolves in the compact 1-byte control array *without*
touching the entry array; PoMap must always probe the big inline entry array, so
it loses misses. **Caveat the other way:** a single-map per-size test lets the
control array (~1.N B ≈ 4.6 MB at 4M) *warm during the pass* and stay resident,
biasing misses toward hashbrown. The main suite's **multi-map sweep** (~600 MB of
maps touched between revisits) keeps control arrays cold, and there PoMap is
competitive on misses (~0.83-0.94x). The true cold-miss ratio is between these;
an unbiased number needs the multi-map round-robin (or N large enough that the
control array >> LLC, ~50M+ entries). **[TODO].**

**Honesty note.** This complicates "cold ⇒ PoMap." PoMap wins **cold,
large-working-set hit reads** (robust, mechanism-backed) and is
competitive-to-better on cold writes; it **loses warm/cache-resident** access
(SIMD throughput) and **loses misses** (compact resident control array). The
defensible headline is the *crossover and its mechanism*, not universal cold
dominance.

**Value-size axis (M3 matrix, `cold-m3max.csv`, `benches/cold.rs`).** The crossover
holds at every value size, but the cold read win is **largest at small values and
erodes as values grow** — cold (128 MiB WS) `get_hit` pomap/hb = 0.63 (8 B), 0.75
(16 B), 0.71 (32 B), 0.85 (64 B). Mechanism (this *refutes* the naive "bigger
values help PoMap" guess): PoMap's entry is `16 + 8·W` B, so once it exceeds a
cache line (80 B at 64-B values) a lookup straddles two lines and the 1-vs-2-line
edge degrades toward 2-vs-3. Writes worsen with value size too — backshift/shift
move more bytes (cold `remove` at 64 B = 1.28× vs 0.81× at 8 B; cold `insert`
similar). So **payload size trades *against* PoMap's speed edge even as it improves
PoMap's *relative* memory cost** (the fixed 8-B inline hash shrinks as a fraction
of a larger slot). Misses remain the weak axis at all value sizes (single-map
biased here; fair = main-suite sweep). Numbers are single-run, median-of-3 —
**trust the trends, not individual cells** (a few points, e.g. 64 B at 48 MiB, are
visibly noisy).

**Two-machine cold matrix (M3 Max vs Zen 5c EPYC 9845; `cold-*.csv`).** The
crossover is microarchitecture-dependent, and the **server CPU broadens PoMap's
advantage** — exactly the direction the bandwidth thesis predicts:

- **get_hit:** both win cold (128 MiB: M3 0.63–0.85×, Zen 5c 0.67–0.92×). But the
  crossover moves *earlier* on the server: for values ≥16 B, Zen 5c PoMap wins
  get_hit at *every* size including warm/cache-resident (16 B @ 0.1–1 MiB:
  0.63–0.83×), whereas on M3 hashbrown wins warm and PoMap only takes over past
  the LLC (~16–48 MiB). At 8 B both still show the warm→cold crossover (hashbrown's
  SIMD edge survives for the tiniest entry).
- **Cold writes INVERT on the server.** At 48–128 MiB, Zen 5c PoMap *wins* insert
  (0.73–0.87×) and remove (0.80–0.85×) across value sizes; on lower-bandwidth M3
  these are only ~parity. Streaming repack/backshift beats scatter-rehash once
  bandwidth binds — the §5.3 inversion, now seen directly in the cold regime.
- **get_miss loses on both at all sizes** (M3 1.1–2.3×, Zen 5c 1.1–2.6×). The
  robust weak spot (single-map biased here; fair = main-suite sweep).
- **Value-size read erosion is M3-specific.** M3 cold get_hit degrades 0.63→0.85
  (8→64 B; entry straddles lines past 64 B); Zen 5c stays ~0.67–0.92 with no clear
  erosion — it wins broadly regardless of payload.

Caveats: Zen 5c is a **virtualized 8-vCPU slice of a 160-core part** (ratios
meaningful, absolutes VM-soft); single run, median-of-3 — several cells are noisy
(trust the trends, not individual cells). **Zen 4 (9354P) matrix not yet captured.**
**Mechanism, to be settled by `perf` (`scripts/perf_cold.sh` + `benches/cold_perf.rs`,
Linux):** the harness runs `perf stat` per impl×op (30 cold passes over a 256 MiB
working set, no inter-pass eviction so counts are clean) and reports per-op
`instructions`, IPC, and L1 / LLC / dTLB load-misses. It will resolve *which*
mechanism drives the cold-hit win — the two candidates differ and we should not
assert one unmeasured:
- **"2 lines vs 1"** in the strong form (hashbrown ≈ 2 *DRAM* misses/lookup) only
  holds if the control array also misses DRAM. At sizes where the 1-byte control
  array is LLC-resident, hashbrown is really **1 DRAM miss (entry) + 1 LLC-latency
  control access + SIMD**, and PoMap's ~15–35% win is that control+SIMD overhead,
  not a 2× miss count. The `perf` LLC-load-misses/op (expect ≈1 for both vs ≈2)
  vs L1-misses/op (expect ~1 vs ~2 — total line touches) distinguishes these.
- dTLB-load-misses/op tests the page-pressure angle (PoMap one big array vs
  hashbrown's small control + big entry).

Run on Zen 5c (perf works there): `scripts/perf_cold.sh` → `perf-<cpu>_<Nc>.csv`.
**[TODO]:** Zen 4 matrix + perf when the box returns.

## 6. Negative results (worth a paper subsection)

These are design alternatives we implemented and measured to lose; reporting them
strengthens the "why this design" argument.

- **SIMD fingerprint tags** (Swiss-table style 1-byte tags + `u8x16` scan):
  loses to the scalar inline-hash design on every read. Probing here is a *short
  sorted run*, not random open addressing, so the SIMD-over-fingerprints trick
  does not pay, and the separate tag array costs a second cache line.
- **Single-pass rebuild** (write each target slot exactly once: per-gap
  region-fill, or inline 8-byte gap stores, instead of `memset` then overwrite):
  **35–44% slower.** A single streaming `memset` of complete cache lines (no
  read-for-ownership) beats any partial-line gap-filling pattern; bytes-written
  is the wrong cost model.
- **Bounded scan windows + cascade displacement** (from the prior SoA designs)
  ported onto the AoS layout: regressed inserts and shrink.
- **Tombstone deletion**: reverted historically (insert regression without enough
  remove gain); backshift is better for this layout.
- **Scan-comparison merge** (3 compares → 1 `>=`): regressed inserts ~1% on
  Zen 5c — used as the controlled probe establishing inserts are not scan-bound.
- **`shift==1` inlined copy** (vs runtime `memmove`): sub-noise, not reproducible.

The SIMD-tags-on-AoS *insert* rework — long held as "the one remaining lever" —
is therefore **contraindicated**: inserts are not scan-bound, and SoA tags would
erode the read wins. (Caveat: the shift-vs-scan conclusion is *inferential* — see
§7 — and should be confirmed with a hardware profiler before being stated as a
result.)

## 7. Threats to validity / open items

- **Bandwidth thesis: probe now exists; two platforms still to capture.** The
  `bench_bandwidth` STREAM-triad mountain is in the harness and captured on M3 Max
  (§4.1). **[TODO]** run it on Zen 4 and Zen 5c and plot each workload's pomap/hb
  ratio against that platform's throughput at the workload's working-set size —
  this is the single most important addition for a convincing paper, and is now
  one bench run per box away.
- **Zen 5c magnitudes are VM-soft.** Shared host, host-managed turbo/governor;
  the *direction* of the inversions is far beyond the ~3% noise floor, but the
  exact factors (0.38×, 0.81×) need a **bare-metal high-bandwidth run** to be
  quotable.
- **`perf` was blocked on Zen 5c** (`perf_event_paranoid=4`, no sudo), so the
  shift-vs-scan conclusion rests on an indirect probe with a confound (the
  scan-merge also dropped the empty-slot early-exit). Needs direct profiling.
- **Anchor/version sensitivity.** Results are against hashbrown 0.14 specifically;
  std's bundled hashbrown differs. A published comparison should pin and report
  the exact baseline version(s).
- **Payload coverage.** Correctness is fuzzed for `u64` and `String`; performance
  is characterized only for `u64`. Large values, non-trivial `Drop`, and
  high-collision adversarial hashes are untested for performance.
- **Statistical rigor.** Point estimates are reported here; §4.5 specifies the
  CI-carrying, run-count-documented reporting the paper needs (the extractor in §8
  now captures criterion's full `[lo pt hi]` interval).
- **Related work / novelty.** Drafted in §10; the open task is completing the
  prior-art search so C1 (the prefix-ordered structure) can be asserted as *new*
  rather than *underexplored*. The ordering-criterion distinction (hash vs key
  comparison vs probe distance) is the crux.
- **Cold-access advantage is argued + partly observed, not yet isolated.** The
  read win is largest where access is cold (§5.2, mechanism in §1); the current
  harness re-probes keys, warming them, so it *under*-measures it. A dedicated
  cold-lookup benchmark is needed — see §5.6.

## 8. Reproducibility

Branch `simd-buckets`. Single implementation in `src/pomap.rs`; benchmark
`benches/pomap_bench.rs`. The `criterion` dev-dependency is a public git fork
(no sibling checkout needed). Run:

```
cargo test                                  # 42 tests, correctness gate
cargo bench --bench pomap_bench             # GROWTH=4 (default)
cargo bench --bench pomap_bench --features growth2   # GROWTH=2
```

The run also prints two report tables to stdout: the memory footprint
(`bench_memory_footprint`, real allocator bytes/entry) and the STREAM-triad
bandwidth mountain (`bench_bandwidth`, GB/s per working-set size).

Extract the full `[lo point hi]` interval (criterion wraps long names):
```
awk '/^[a-z_]+\/[a-z0-9_]+/{n=$1} /time:/{for(i=1;i<=NF;i++)if($i=="time:"){print n"\t"$(i+1)" "$(i+2)" "$(i+3)" "$(i+4);break}}' out.txt | grep -v report
```
(point estimate is the middle of the triple; the outer two are the 95% CI.)

On servers: pin to a non-zero core (`taskset -c 2`), set the governor to
`performance` and disable turbo where possible, keep the box idle, and capture
full output (never pipe through `tail` — it truncates early groups). Capture the
`bench_bandwidth` table on each platform — it is the bandwidth x-axis for §5.3.

Cold-access matrix + perf (the cold-read mechanism, §5.6):
```
scripts/run_cold.sh    # → cold-<cpu>_<Nc>_<GB>GB.csv  (all platforms)
scripts/perf_cold.sh   # → perf-<cpu>_<Nc>.csv         (Linux; needs perf + paranoid<=1)
```
`perf_cold.sh` env knobs: `VW` (value words 1|2|4|8), `WS` (MB), `PASSES`, `EVENTS`
(append an AMD `ls_misal_loads.*` event to measure split loads if `perf list`
shows it).

## 9. Summary of what we are confident in

1. **Reads/updates beat hashbrown everywhere** (measured, robust) — a
   *deterministic* hash table matching or beating SwissTable-class point-read
   performance, with the margin growing as access gets colder/bandwidth grows
   (the 1-miss vs 2-miss cache-line mechanism, §1). The cold-read isolation
   benchmark (§5.6) is the key experiment to make this claim airtight.
2. **Write/bulk operations are bandwidth-bound and invert from deficit to win on
   high-BW server CPUs** (measured direction robust; magnitudes VM-soft pending
   bare metal).
3. The only durable losses are the small-cache-resident miss-scan and the pure
   insert shift tax; both narrow with bandwidth and neither has a fix that
   preserves the read wins.
4. The design is at/near its optimum on three microarchitectures — the optimizer
   found no validated gain on the latest — so the contribution is the
   architecture and its bandwidth-favorability, not further micro-optimization.

## 10. Related work and positioning

> **[DRAFT — bibliographic details to verify before submission.]** Works are
> named at the confidence level we have; exact authors/venues/years are marked
> *(verify)* where we are not certain. In a finished paper this belongs in §2.
> A dedicated literature search is still owed for the closest prior art on
> *sorted / order-preserving open addressing* (see end).

**SIMD open-addressing tables (the primary baseline).** Google's *SwissTable* /
Abseil `flat_hash_map` (Kulukundis, "Designing a Fast, Efficient, Cache-friendly
Hash Table, Step by Step", CppCon 2017 *(verify)*) stores a parallel array of
1-byte control tags and probes 16 at a time with SSE2/NEON. `hashbrown`
(Amanieu d'Antras *(verify)*) is the Rust port and backs `std::collections::HashMap`.
Facebook's *F14* (Folly; Bronson & Shi, "Open-sourcing F14", 2019 *(verify)*) is
a related SIMD-chunked design with both AoS (`F14Value`) and indirected (`F14Node`)
layouts. **Contrast:** these are *unordered* and store the discriminator (control
byte / tag) in a *separate* array (SoA). PoMap is *hash-ordered* with deterministic
iteration and stores the *full* hash *inline* (AoS), trading 7 extra discriminator
bytes/slot for a single-cache-line probe and exact (collision-free) filtering. Our
own SoA-tag prototype reproduced the SwissTable approach and lost on reads (§6),
which motivated the AoS choice.

**Probe-sequence-optimizing schemes.** Robin Hood hashing (Celis, 1986 *(verify)*)
equalizes probe distances; Hopscotch (Herlihy, Shavit, Tzafrir, 2008 *(verify)*)
and Cuckoo hashing (Pagh & Rodler, 2001 *(verify)*) bound worst-case lookup.
Skarupke's `ska::flat_hash_map` / "bytell" (blog, *(verify)*) popularized several
of these in C++. **Contrast:** all are unordered and optimize the *probe*; PoMap
instead keeps the array globally *sorted by hash*, which is what makes both
deterministic iteration and the early-terminating `stored > hash` scan possible,
at the cost of shifting on insert.

**Ordered and deterministic-iteration maps.** `std::collections::BTreeMap` is the
standard ordered map: a cache-conscious B-tree, comparison-based, O(log n),
ordered by the *key*'s `Ord`. The `indexmap` crate (bluss *(verify)*) gives
deterministic *insertion* order via a separate index vector over a SwissTable.
**Contrast:** PoMap is ordered by *hash* (not key order — an important caveat for
users, since `Ord` on the key does not imply iteration order), is hash-based and
~O(1) per op rather than O(log n), and needs no auxiliary index. Its niche is
"deterministic iteration + class-leading point reads" where BTreeMap's key-order
and log-factor, or IndexMap's extra indirection, are not wanted.

**Closest prior art: ordered hashing (the reference to anchor against).** Amble &
Knuth, "Ordered hash tables" (*The Computer Journal* 17(2), 1974 *(verify
page/issue)*) is the key precedent: they keep entries within a probe sequence in
order of a key/signature so that an *unsuccessful* search terminates early — which
is exactly PoMap's `stored > hash` cutoff. **This must be cited and is the obvious
"isn't this just…" challenge.** PoMap's delta over classical ordered hashing:

1. **Global** order, not per-probe-sequence order: the position is the hash
   *prefix* (an MSD-radix bucket), so the *entire array* is one hash-sorted
   sequence. That is what gives **deterministic global iteration order** — a
   property ordered hashing does not provide and that is a primary reason to use
   PoMap.
2. A modern **flat AoS inline-hash layout** and the analysis of its cache/bandwidth
   behavior on contemporary hardware (the 1974 work predates the memory wall and
   SIMD tables entirely).
3. The empirical **bandwidth-favorability** result (C2).

So the novelty position is: **(C1) the prefix-ordered structure is claimed as a
novel synthesis** — global hash-order via prefix addressing + AoS inline hash +
deterministic iteration — extending the ordered-hashing lineage, **and (C2) the
bandwidth argument** is the empirical contribution. **[TODO]** complete the
literature search before asserting C1 as *new* rather than *underexplored*:
beyond Amble–Knuth, check its descendants, "self-organizing"/"last-come-first-
served" hashing, Robin Hood variants that maintain order, MSD-radix / histogram
bucketing of hashes, and any "hash-ordered" or "sorted flat" map in the systems
literature. Treat C1 as *defensible-pending-search*, not yet *established*.
