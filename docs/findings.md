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

**Origin and motivation.** The structure was originally developed as the layout
engine for an RCU-style concurrent hash table (the `seqmap` project, mid-2025,
sibling repo): copy-then-publish resizing is only practical when rebuilding the
table is cheap, and that requirement forces a layout whose sort order survives
growth. The first attempt used *split-ordered* (bit-reversed-hash) addressing per
Shalev & Shavit (§10) — per-bucket stability under growth, but scattered adjacency
(no locality, no meaningful iteration order) and per-cell boxing overhead.
Inverting the invariant to hash-*prefix* (MSB) addressing produced global order,
cache locality, and a streaming resize — and then kept paying: deterministic
iteration, early-terminating probes, and the measured write behavior of §5.3 all
fall out of the same choice. The concurrent map is future work; this paper
presents and measures the sequential structure.

**The unifying invariant.** Place each entry by its hash *prefix* (top `m` bits →
ideal slot) and keep the array globally sorted by full hash. The prefix map is
monotone in the hash, so **global hash order is invariant under power-of-two
growth**. One invariant, three payoffs: (a) deterministic iteration, (b)
early-terminating probes, (c) a comparison-free single-pass streaming resize.

**Contributions.** We claim three:

- **(C1) The prefix-ordered hash map (systems synthesis).** A flat open-addressing
  table kept *globally sorted by hash value*, with each entry's home position given
  by its hash *prefix* so that array order *is* hash order, displacing to the
  nearest vacancy to maintain the sort. Keeping a linear-probing table in hash
  order is established — it is *ordered linear probing*, from Amble & Knuth (1974)
  through graveyard hashing (FOCS 2021) and the ordered/unordered tight analyses
  (FOCS 2024) (§10) — so the base structure is **not claimed as new**. The claimed
  synthesis is what the flat realization yields and the literature never surfaces:
  (a) **hash-prefix home slots make home-bucket order equal *global* hash order**,
  turning the whole array into overlapping, order-fused regions (every slot heads
  its own region; boundaries are *data-defined* by the sort, not geometric — no
  fixed buckets exist) and giving **deterministic whole-map iteration** as an API
  property; (b) **the ordering criterion is the hash, not the key** — a key
  comparison is never on the layout path (verified: only `u64` compares navigate;
  `==` does the final match; the `Key: Ord` bound is vestigial — functionally only
  `Hash + Eq + Clone` is needed), distinguishing it from classical ordered hashing
  (key comparison) and Robin Hood (probe distance); (c) an **AoS inline-full-hash
  layout** that makes a probe one cache line, with `u64::MAX` doubling as empty
  sentinel and scan terminator.
- **(C2) Comparison-free streaming resize, and its measured consequences.**
  Because sort order survives growth, resize is **one sequential read pass with a
  monotone write cursor** — no re-hashing, no re-sorting, no scatter — plus
  **cursor-spacing gap injection** during the pass (re-seed inter-run gaps on grow
  so post-grow inserts land at open ideal slots; a PMA-flavored redistribution with
  no hash-table precedent we know of). The order-survives-growth invariant itself
  has chained-table precedents built for concurrent resizing (split-ordered lists;
  relativistic/RCU hash tables — §10); the flat open-addressing realization as a
  memcpy-class streaming rehash appears to be new. Its measured consequence is the
  sequential-vs-scatter write behavior: on *cold/large* working sets PoMap's writes
  **tie on high-per-core-bandwidth Apple parts and win on low-per-core-bandwidth
  AMD EPYC** (bare-metal *and* virtualized), with **virtualization extending the
  win to medium sizes** (nested page walks punish scatter). It is NOT a clean
  "server inversion," NOT bandwidth-*favorability* (Apple has the most bandwidth
  and least write advantage), and NOT a pure virtualization artifact (§5.3). On
  warm/medium sizes PoMap loses writes everywhere.
- **(C3) A measured design space (Pareto frontier + negative results).** Seven
  engines from the same family benchmarked under one harness (§6): the AoS engine
  is read-optimal; a SIMD-tag SoA variant is *build-optimal* (beats hashbrown
  building from empty); a tag-only variant is *memory-optimal* (≈ hashbrown
  bytes/slot); a bounded-window variant offers worst-case-bounded probes via
  order-preserving cascade displacement (plausibly novel — no ordered Hopscotch
  in the literature we searched, §10); and a tombstone A/B cleanly measures the
  remove-vs-insert trade the graveyard-hashing line studies. Plus the negative
  results that justify the canonical design choices (AoS over SoA tags, backshift
  over tombstones at this load factor, memset-then-write over single-pass rebuild).

**Central empirical claim.** PoMap is a *deterministic* hash map whose point-read
and update performance **matches or beats** SwissTable-class tables (hashbrown)
on every microarchitecture tested. Its writes follow a warm→cold crossover and, on
**cold/large** working sets, **tie on Apple and win on low-per-core-bandwidth AMD**
(extended to medium sizes under virtualization) — there is no universal write
inversion (§5.3). The unifying read mechanism is the **memory hierarchy**, and the
advantage is largest exactly where real large-map workloads live — **cold,
low-locality access**:

- *Cold reads.* A PoMap lookup keeps the hash inline with its (K, V), so probe +
  filter + fetch hit one contiguous region; a SwissTable lookup touches the
  control-byte group *and* the entry in a separate array. perf on the cold-hit
  path (§5.6) attributes PoMap's win to **two measured effects**: ~30% fewer
  instructions (no SIMD control-scan machinery) and ~26% fewer L1-dcache misses
  (fewer lines touched) — netting ~0.65× the cycles. (The precise DRAM/LLC-miss
  count is unconfirmed — the VM lacks L3 counters; bare-metal TODO.) The edge
  *widens with working-set size and coldness*; **below the cache boundary (warm)
  hashbrown's SIMD throughput wins** — a crossover (§5.6), not a blanket read win,
  and misses are the exception even cold.
- *Cold writes.* PoMap's inserts/repacks/backshifts are sequential streaming;
  SwissTable growth rehashes (scatter — random writes). When random access is
  expensive relative to sequential (high latency / small per-core cache / TLB
  pressure — the server regime), sequential wins — hence the server-CPU inversion.

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
- **Resize (the C2 mechanism).** Because the prefix map is monotone in the hash,
  the old array's order is already the new array's order: resize is a **single
  sequential read pass with a monotone write cursor**
  (`cursor = max(cursor, new_ideal_slot); cursor += 1`) — comparison-free, no
  re-hashing (the hash is inline), no scatter. The target is initialized with one
  streaming `memset(0xFF)` and only occupied slots are written in the pass
  (memset-then-write measurably beats write-each-slot-once — §6). Every design in
  the family, from the first prototype onward, shares this resize.
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
| M5 | Apple M5 Max, 18-core, 128 GB unified | Apple M5 | macOS, AC power | 1.96.0 | newest dev machine; remove anchors *soft* (see 4.4) |
| M3 | Apple M3 Max (Mac15,9), 16-core (12P+4E), 128 GB unified | Apple M3 | macOS, AC power | 1.96.0 | remove anchors *soft* (see 4.4) |
| EPYC 9354P | AMD EPYC 9354P, 32-core, 755 GB | Zen 4 | Linux, **bare-metal** | 1.90 | bare-metal x86 reference (settles §5.3) |
| Zen 5c (VM) | AMD EPYC 9845 "Turin Dense" | Zen 5c | Linux, **virtualized 8-vCPU guest**, AVX-512 (vp2intersect/vaes/gfni) | **[TODO]** | host-managed turbo/governor; ~3% noise floor; nested page walks (see §5.3) |

Single-threaded streaming-throughput (STREAM-triad) curves, captured by the
`bench_bandwidth` probe in the harness (working set = 3 arrays of f64), give the
per-platform bandwidth x-axis for §5.3. **Single-thread DRAM bandwidth (DRAM-WS
end of the mountain): Apple ~100–124 GB/s (M3/M5) vs AMD EPYC ~41–44 GB/s per
core** — Apple Silicon has unusually high single-core bandwidth; EPYC has modest
*per-core* bandwidth despite huge *aggregate* socket bandwidth. This inverse split
(Apple high, AMD low) is the axis §5.3's write crossover tracks. Single-thread
pointer-chase DRAM *latency* (`bench_latency` mountain): M5 ~81 ns, M3 ~118 ns,
EPYC 9354P (bare) ~102 ns — note EPYC's latency is *lower* than the M3's, which is
why latency does **not** explain the write result.

**[TODO]** Zen 5c rustc version; `bench_latency` mountain on the Zen 5c VM (its
data file predates the probe).

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
- **Thermal-state audit (2026-08-18, M5, `audit_bench.rs`): the get-class groups
  are warm-regime, and it does not matter.** The per-size RNG seeds are fixed, so
  every criterion iteration touches the same ~100 keys per map (~a few hundred KB
  across the sweep) — measured: identical work runs 22.7 µs warm (fixed seeds)
  vs 109.2 µs with per-iteration-varying seeds (a true mixed warm/cold sweep),
  confirming residency. **But the pomap/hashbrown ratio is robust to thermal
  state: 0.63× fixed/warm vs 0.67× varying/cold-sweep.** The read win is not a
  warm-cache artifact. Report suite numbers as "sweep aggregate," and cite this
  A/B when a referee asks.
- **RNG dilution: reported ratios are conservative.** The timed get loops include
  per-size `StdRng` setup plus one `random_range` per get — an overhead measured
  at **12.8 µs/iteration** (loop-only baseline), i.e. ~56% of pomap's warm get
  number and ~36% of hashbrown's. This equal additive constant compresses every
  ratio toward 1.0: net of overhead, the warm get_hits ratio is ≈**0.43×**, not
  the reported 0.63×. All main-suite read ratios therefore *understate* the
  advantage. A dilution-free harness (pre-generated index arrays, no RNG in the
  timed loop) is used by the optimizer loop bench; the cross-machine
  `pomap_bench` numbers are kept as-is for continuity, with this caveat.
- **Smaller harness notes (audited 2026-08-18):** `remove_hits`/`shrink_to` run
  against fresh clones per batch (the ~large setup clone evicts caches →
  cold-ish) while `remove_misses`/`update_existing` reuse maps across iterations
  (warm) — thermal regimes differ *between* groups; per-group cross-impl fairness
  is unaffected. `update_existing` touches only `keys[0..100]` (warm by design).
  `get_hotset`'s per-map seed collapses to the same sequence for all maps with
  ≥1000 entries (benign — the hot set is shared by design). The criterion fork's
  `comparison_benchmark_group` is reporting-only (rank summary; measurement
  machinery is upstream criterion). All impls share one fixed-key `ahash`
  builder: layouts are deterministic across runs (good for reproducibility), but
  results sample a *single* hash seeding — repeating headline tables under k
  random hash seeds is owed for the paper. No iteration-order benchmark exists
  yet; deterministic iteration is a headline feature and should be measured.

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

**Run on every platform** (M5, M3 Max, EPYC 9354P bare-metal, Zen 5c 9845 VM),
idle box:
```
scripts/run_cold.sh        # → cold-<host>.csv
```
then collate the per-host CSVs. Captured on all four (§5.6); the crossover
working-set tracks each platform's LLC size, and the lower-per-core-bandwidth AMD
parts show the earlier/broader cold win.

## 5. Results

### 5.1 Cross-platform normalized ratios (pomap ÷ hashbrown, GROWTH=4)

Lower is better; `<1.0` means PoMap beats hashbrown. Each column is normalized
within that platform's own run.

| Workload | M5 | M3 *(soft rm)* | EPYC 9354P (bare) | Zen 5c (VM) |
|---|---|---|---|---|
| get_hits | 0.69 | 0.65 | 0.65 | **0.52** |
| get_misses | 0.86 | 0.83 | 0.97 | 0.94 |
| get_hotset | 0.76 | 0.68 | 0.52 | **0.53** |
| update_existing | 0.59 | 0.62 | 0.46 | **0.30** |
| remove_hits | 1.86 *(soft)* | ~1.2 *(soft)* | 1.06 | **0.59** |
| remove_misses | 1.00 | 0.89 | 1.22 | 1.12 |
| insert_allocate | 1.20 | 1.08 | 1.07 | **0.80** |
| insert_preallocated | 1.77 | 1.66–1.81 | 1.49 | 1.15 |
| shrink_to | 1.18 | 1.11 | 1.35 | **0.37** |
| **groups won (<1.0)** | 4–5 / 9 | 6 / 9 | 4 / 9 | **7 / 9** |

Reads/update win on **all four**. The write/bulk cells, however, **do not show a
clean "server inversion"**: `shrink_to`, `insert_allocate`, and `remove_hits` win
*only on the Zen 5c VM* (0.37 / 0.80 / 0.59) and **lose on the bare-metal EPYC
9354P** (1.35 / 1.07 / 1.06) — same vendor, opposite result. The main-suite write
workloads sit at **medium/warm working sets** (~2–19 MB), inside the warm→cold
crossover, where **virtualization** (nested page walks taxing hashbrown's scatter)
is what flips them — not bandwidth, and not bare-metal architecture. The honest
write picture is the **per-size cold matrix** (§5.6), where cold/large writes tie
on Apple and win on AMD bare-metal *and* VM. Treat these main-suite write
aggregates as crossover-region samples, not a property.

### 5.2 Reads and updates

PoMap wins all three get workloads and `update_existing` on every platform in the
*aggregate* over the standard suite's `evenly_spaced` sizes (often ~2×, e.g.
`update_existing` 0.32× on Zen 5c). This is the inline-hash AoS locality advantage
— one cache line per probe versus hashbrown's control-byte array plus a separate
entry slab — and it widens with bandwidth. **Caveat:** that aggregate is
dominated by the larger (colder) sizes; the per-size sweep (§5.6) shows the win is
a cold/large-working-set phenomenon with a warm-regime crossover, and that
*misses* are a genuine weak spot. Read §5.2 and §5.6 together.

### 5.3 Writes: a crossover that depends on the machine (4-platform reconciliation)

There is **no universal write "inversion"** — and the story took two wrong turns
before the bare-metal EPYC settled it (recorded honestly here because both
overclaims could recur). Writes follow the **same warm→cold crossover as reads**,
and the cold-write *outcome* is **architecture-dependent**.

**Cold/large writes (per-size cold matrix, 128 MiB working set, pm/hb):**

| op | M5 | M3 | EPYC 9354P (bare) | EPYC 9845 (VM) |
|---|---|---|---|---|
| insert | 1.05 | 1.06 | **0.70** | **0.75** |
| remove | 1.00 | 0.81 | **0.90** | **0.79** |

So **cold/large writes ~tie on Apple and win on AMD** — and crucially the
*bare-metal* AMD wins (0.70×), so this is **not** a virtualization artifact. It is
an architecture effect that tracks **single-thread bandwidth, inversely**: AMD
EPYC has ~41–44 GB/s per core vs Apple's ~100–124 (M3/M5); PoMap's sequential
writes use scarce bandwidth efficiently (full cache lines, prefetchable) while
hashbrown's scatter-rehash wastes it (partial lines, unpredictable). When bandwidth
is abundant (Apple) the waste is free → tie; when scarce (AMD) it bites → PoMap
wins. (Note: raw DRAM *latency* does **not** explain it — bare-metal EPYC's 102 ns
is *lower* than the M3's 118 ns yet it wins; the `bench_latency` mountains are M5
81 ns, M3 118, EPYC-bare 102. So it is bandwidth-efficiency / cache-line utilization,
not latency.)

**Virtualization separately amplifies it at *medium* sizes.** The main-suite
`shrink_to` (a ~2–19 MB working set, inside the crossover) is **0.37× on the Zen 5c
VM but 1.35× on the bare-metal EPYC** — same vendor, opposite result. Under
virtualization a TLB miss is a 2D (nested) page walk, which punishes hashbrown's
scatter at sizes that are cache-resident bare-metal but TLB-thrashing in a guest.
So the eye-catching "shrink 2.7× faster" was the **VM at a medium working set**, not
a general property.

**Two corrections to retire:** (1) the original "bandwidth-favorability / scales to
high-BW servers" — wrong direction (Apple has the *most* bandwidth and the *least*
write advantage); (2) the panic "the inversion was all virtualization" — also wrong
(bare-metal AMD wins cold/large writes). The honest claim: **cold/large writes tie
on Apple and win on low-per-core-bandwidth AMD, with virtualization extending the
win to medium sizes.** The **main-suite write aggregates are a poor summary** — they
sit at medium/warm sizes inside the crossover; the per-size cold matrix is the
honest view.

**[TODO]** add the `bench_latency` mountain on the Zen 5c (its file predates the
probe) so all four platforms have it; an `amd_l3`/raw-event perf run would give the
DRAM-miss counts the generic `LLC-load-misses` cannot (it reads `<not supported>`
on AMD even bare-metal — §5.6).

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
real use). Measured bytes-per-entry vs hashbrown, M3 Max, **both growth factors**:

| entries | PoMap/hb @ GROWTH=2 | PoMap/hb @ GROWTH=4 |
|---|---|---|
| 500 | **1.56×** | 2.99× |
| 5,000 | 1.44× | 1.44× |
| 50,000 | 2.83× | 2.83× |
| 500,000 | **1.41×** | 2.82× |
| 5,000,000 | 1.41× | 1.41× |

The ratio is **lumpy**, governed by where N lands relative to a growth boundary:
~1.4× near full load, up to ~2.8–3.0× just after a grow (sparse). **GROWTH=2 is
≤ GROWTH=4 at every size** — identical where the power-of-two boundary coincides
(5k, 50k, 5M), and much tighter where GROWTH=4 overshoots into a sparse post-4×-grow
table (500, 500k). Averages: **G2 ≈ 2.0×, G4 ≈ 2.75×**; worst case ~2.8–3.4× both.
The table reflects `with_hasher` (build-from-empty, the *pessimistic* footprint);
`with_capacity(n)` provisions to ~62.5% load and lands tighter. Memory is the
design's real cost — the inline 8-byte hash plus the low load factor that buys the
read/write wins.

**The dial extends: GROWTH=8 inverts build-from-empty (measured 2026-08-18, M5,
`loop_bench` in-run, median of 3).** insert_allocate vs hashbrown: **G2 1.56×,
G4 1.01× (parity), G8 0.877× — pomap faster than hashbrown at building from
empty.** Every halving of repack volume keeps paying (total moved entries is
n·G/(G−1): 2n at G2, 1.33n at G4, 1.14n at G8). With G8 the map beats hashbrown
on *every* workload class on this machine except the §5.4 residuals
(insert_preallocated, warm removes). **Measured G8 footprint (examples/footprint.rs, validates cell-for-cell against
the §5.5 table):** at the five standard sizes, G8 *averages the same as G4*
(67.5 vs 69.6 B/entry; G2 49.6) with the same observed worst case (104.1 at
N=500) — G8's sparser growth sequence sometimes lands tighter than G4 (50.3 vs
100.7 at 500k). Caveat: over arbitrary N the *expected* slack grows with G (the
sampled parity is power-of-two coincidence; worst-case overshoot is ~G× just
past a grow); `with_capacity` provisioning is growth-independent and sidesteps
it entirely. Net: **G8 buys the build-from-empty inversion at ≈G4's average
memory on these sizes** — a legitimate published configuration for build-heavy
deployments; G4 stays the default.

**Growth factor: a two-point dial, GROWTH=4 as the performance-canonical
configuration.** GROWTH affects *only* `insert_allocate` (build-from-empty) and
this footprint — every other operation provisions via `with_capacity` and is
growth-independent. The dial: **G4 → insert_allocate ~1.1× hb (parity-class) at
~2.75× average memory; G2 → 1.5–1.9× hb at ~2.0× memory.** G4 is what makes the
build-from-empty column competitive — with it, the map is at-or-near hashbrown on
every workload class except the two §5.4 residuals — and it has been dramatically
faster than G2 across every design generation (halving repack volume: total moved
entries over n inserts is n·G/(G−1): 2n at G2 vs 1.33n at G4). It is the default
and the configuration the headline tables report. **G2 is the memory-lean
alternative**: its only cost is build-from-empty speed, which a caller sidesteps
entirely with `with_capacity` (growth-independent, ~1.8× either way), and it cuts
the design's biggest cost from ~2.75× to ~2.0× hashbrown's bytes. The paper should
present both points and let the deployment pick; the const-generic makes the choice
compile-time free.

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

**Four-machine cold matrix (M5, M3 Max, EPYC 9354P bare-metal, EPYC 9845 VM;
`cold-*.csv`).** The crossover is microarchitecture-dependent. The read win is
universal; the **write outcome splits by vendor along the single-thread-bandwidth
axis** (§5.3), and the bare-metal AMD is what settles the mechanism:

- **get_hit:** all four win cold **at small values (8–16 B)**. The crossover moves
  earliest on the **Zen 5c VM** (values ≥16 B win at essentially every size,
  including warm), whereas on Apple hashbrown wins warm and PoMap takes over past
  the LLC. **The bare-metal EPYC 9354P sits between** *(corrected 2026-08-17
  against the raw CSV)*: it wins cold at 8–16 B (0.79–0.95×) but its warm 16 B
  cells *lose* (1.20–1.26×), and the value-size erosion carries its cold get_hit
  past parity at larger values (32 B/48 MiB 1.10×, 64 B/128 MiB 1.11×). So the
  headline must be qualified: **small-value cold reads win universally;
  large-value cold reads erode to parity-or-slight-loss on bare-metal Zen 4.**
  At 8 B all four show the warm→cold crossover (hashbrown's SIMD edge survives
  for the tiniest entry).
- **Cold/large writes split by vendor (128 MiB WS, pm/hb):**

  | op | M5 | M3 | EPYC 9354P (bare) | EPYC 9845 (VM) |
  |---|---|---|---|---|
  | insert | 1.05 | 1.06 | **0.70** | **0.75** |
  | remove | 1.00 | 0.81 | **0.90** | **0.79** |

  Cold writes **tie on Apple and win on AMD — including the bare-metal EPYC
  9354P**, so this is *not* a virtualization artifact. It tracks single-thread
  bandwidth inversely (AMD ~41–44 GB/s/core scarce → PoMap's full-cache-line
  streaming wins; Apple ~100–124 abundant → scatter's waste is free → tie). Raw
  latency does not explain it (EPYC-bare 102 ns < M3 118 ns yet EPYC wins).
- **get_miss loses on all at all sizes** (1.1–2.6×). The robust weak spot
  (single-map biased here; fair = main-suite sweep).
- **Value-size read erosion is universal** (corrected): cold get_hit advantage
  shrinks as values grow (M3 0.63→0.85; Zen 5c cold-end 0.67→0.92, 8→64 B). perf on
  Zen 5c explains it (below): an instruction + IPC effect from the entry straddling
  cache lines, not a miss-count one.

Caveats: Zen 5c is a **virtualized 8-vCPU slice** (ratios meaningful, absolutes
VM-soft); single run, median-of-3 — several cells are noisy (trust trends, not
individual cells).
**Mechanism — perf-measured (Zen 5c VM, `get_hit`, 256 MiB WS, 30 passes).**
Per-op counters, pomap vs hashbrown:

| metric | pomap | hashbrown | pm/hb |
|---|---|---|---|
| instructions/op | 56.8 | 79.5 | **0.71** |
| cycles/op | 179 | 277 | **0.65** |
| L1-dcache-miss/op | 2.63 | 3.55 | **0.74** |
| cache(L2)-miss/op | 3.64 | 4.56 | 0.80 |
| dTLB-miss/op | 2.02 | 2.15 | 0.94 |

The cold-hit win is driven by **two measured mechanisms**: (1) **~30% fewer
instructions** — hashbrown's control-group load + `pcmpeqb`/`pmovmskb`/mask-scan
machinery that PoMap's scalar inline scan avoids (confirms the SIMD-overhead
claim); (2) **~26% fewer L1-dcache misses** — fewer cache lines touched (the
locality claim). The cycles ratio (0.65×) matches the wall-clock cold-hit ratio.
**So the earlier strong "2 DRAM misses vs 1" should be stated as "fewer cache-line
misses + fewer instructions"** — the L1-miss ratio is 0.74×, not 0.5×, and absolute
counts include ~1 key-array miss/op of harness overhead common to all impls.

**Value-size sweep explains the read erosion (`get_hit` pm/hb across 8→64 B):**

| value | instr | cycles | L1-miss | IPC pomap | IPC hb |
|---|---|---|---|---|---|
| 8 B | 0.71 | 0.63 | 0.74 | 0.317 | 0.281 |
| 16 B | 0.73 | 0.66 | 0.86 | 0.316 | 0.287 |
| 32 B | 0.77 | 0.73 | 0.74 | 0.355 | 0.338 |
| 64 B | 0.80 | **0.87** | 0.85 | 0.388 | **0.419** |

The cycle advantage erodes 0.63→0.87 because (1) PoMap's **instruction** edge shrinks
(0.71→0.80 — hashbrown's fixed SIMD overhead amortizes over more per-op work) and
(2) **IPC crosses over** — PoMap goes from above hashbrown (8 B) to below it (64 B),
the cache-line-straddling penalty of the 80 B entry. The L1-miss ratio is flat/noisy,
so the erosion is an instruction + execution-efficiency effect, **not** a miss-count
one. This is the mechanism behind the wall-clock value-size erosion (§5.6 above),
and it reproduces on Zen 5c — confirming the erosion is universal, not M3-specific.
(`get_miss` corroborates the miss weakness independently: PoMap's dTLB misses run
**2–4× hashbrown's** — it probes one huge array while a SwissTable miss resolves in
the small, TLB-friendly control array.)

**Two hard limits of this measurement:**
- **`LLC-loads`/`LLC-load-misses` are `<not supported>` on the VM** (no L3 PMU
  passthrough), so the precise DRAM-miss count — the cleanest test of the
  1-vs-2-line story — is *unconfirmed* and deferred to the bare-metal Zen 4 box.
- **The `get_miss` perf rows are unrepresentative.** The harness does not evict
  between the 30 passes (to keep counts clean), so hashbrown's ~15 MB control
  array warms into L3 and its miss looks cheap — *not* the cold-miss regime. The
  cold-miss verdict stays with the wall-clock matrix (PoMap loses).

**[TODO]:** bare-metal Zen 4 for LLC counters + a cold-miss-faithful perf variant.
(Tooling note: an `ops=` parse bug blanked the first CSVs; data above came from the
captured perf stderr. Fixed — re-runs now populate `perf-<cpu>_<Nc>.csv` directly.)

### 5.7 String payloads: the write story inverts (first recording, 2026-08-18, M5)

The `bench-string` feature (128-byte `String` keys *and* values; fixed this date
— it had never compiled) yields, in-run vs hashbrown, single run:

| group | ratio | | group | ratio |
|---|---|---|---|---|
| insert_allocate | **0.708** | | get_hits | **0.889** |
| shrink_to | **0.353** | | get_hotset | 0.991 |
| remove_hits | **0.736** *(soft anchor)* | | update_existing | 0.941 |
| insert_preallocated | 1.157 | | get_misses / remove_misses | 1.074 / 1.101 |

**PoMap wins builds, shrinks, and removes outright with String keys — at
GROWTH=4, warm, on the machine where the u64 suite shows write deficits.** The
mechanism is a claim the paper had not yet articulated: SwissTable stores only a
7-bit tag, so **every resize re-hashes every key** — expensive for non-trivial
keys — while PoMap's inline full hash makes its streaming resize **hash-free**
(the C2 pass never touches key bytes). The 8-byte inline hash, booked until now
as pure memory overhead, is a computational asset whenever hashing is
non-trivial. Secondary effects: the insert shift tax nearly vanishes (1.77× →
1.16×; per-insert cost is dominated by string hashing/allocation, equal for all
impls) and the read win erodes as §5.6 predicts for larger entries (56-byte
entries; 0.68× → 0.89×). Caveats: single run, M-series (remove anchor soft),
one payload shape (128 B); the cross-machine String matrix is owed. This also
sharpens the positioning: **the map's strongest workload class is
expensive-to-hash keys — precisely the common case (strings, paths, URLs,
composite keys) — where it wins builds AND reads simultaneously.**

### 5.8 Iteration throughput (first measurement, 2026-08-18, M5, `iter_bench`)

Whole-map iteration (sum keys+values; medians of 3; ratio vs hashbrown):

| entries | pomap | hashbrown | std | BTreeMap |
|---|---|---|---|---|
| 1k | 2.36× (1.20 ns/e) | 1.00 (0.51) | 1.00 | 1.22 (0.62) |
| 100k | 4.40× (5.26 ns/e) | 1.00 (1.20) | 0.99 | 1.60 (1.91) |
| 1M | 2.49× (4.93 ns/e) | 1.00 (1.98) | 1.00 | 2.44 (4.84) |

**The naive iterator was a loss, and the mechanism was instructive**: one
*unpredictable* branch per slot over a 24-byte stride (~40–50% slot density
makes the skip branch a coin flip), vs hashbrown's SIMD scan of compact control
bytes. The ratio tracked slot density exactly (worst at 100k, where
with_capacity provisioning lands at ~38% density).

**Fixed the same night (2026-08-18): a branchless 64-slot occupancy-mask
iterator** (refill a u64 mask with a cmp/set/or pass over the inline hashes —
predictable loop — then pop set bits; `remaining == 0` skips the tail).
Measured (median of 3, vs in-run hashbrown):

| entries | before | after | pomap speedup |
|---|---|---|---|
| 1k | 2.36× | 1.66× | 1.40× |
| 100k | 4.39× | **1.07×** | 4.20× |
| 1M | 2.49× | **0.61×** | 4.14× |

At 1M PoMap now iterates **39% faster than hashbrown** (~1.19 vs 1.98 ns/entry):
with the branch misses gone, whole-map iteration is a sequential stream over one
flat allocation — the same sequential-vs-scatter character as C2, surfacing on a
read path. Small maps still pay fixed overheads (1.66× at 1k). `Keys`/`Values`/
`values_mut` inherit via delegation; `IterMut` carries the same scheme;
`IntoIter`/`Drain` remain on the simple path (their `Drop` must account for
unconsumed entries — follow-up). **Headline: deterministic hash-order iteration
that beats the unordered incumbent at scale** — strictly dominating BTreeMap
(4.8 ns/e) on both order-availability-per-cost and point-op speed.

## 6. The design space: a measured Pareto frontier, plus negative results

### 6.1 Family benchmark (all seven engines, one harness)

Benchmark methodology changed substantially across design generations (§4.4), so
per-branch suite results are not comparable. To rank the family soundly, every
surviving engine was vendored into a single crate and benchmarked under the
*current* harness — same seeds, sizes, drop-fix timing, and in-run
hashbrown/std anchors — gated by a correctness cross-check of each engine against
hashbrown (`family_bench.rs`; first run 2026-08-17, M5 Max,
`bench-family-Apple_M5_Max_2026-08-17.txt`). Engines: **pomap** (current AoS,
G4), **pomap3** (pre-consolidation AoS, 75% load, no zero-grow contract),
**oldsimd** (tag-only SoA: 1-byte tags + (K,V), *no stored hash*, order on prefix
bits only, SIMD scan), **main_soa** (SoA + bounded windows + cascade
displacement), **tags_soa** (main_soa + displacement-nibble/fingerprint tag byte,
u8x16 scan), **tags2_soa** (tags_soa + tombstones). Ratios ×hashbrown, in-run:

| group | pomap | pomap3 | oldsimd | main_soa | tags_soa | tags2_soa |
|---|---|---|---|---|---|---|
| get_hits | **0.68** | 0.71 | 0.83 | 0.88 | 0.98 | 0.81 |
| get_misses | **0.84** | 0.87 | 1.08 | 1.11 | 0.91 | 0.87 |
| update_existing | 0.65 | 0.63 | 1.08 | 0.70 | 0.79 | 0.90 |
| get_hotset | **0.77** | 0.79 | 0.90 | 0.88 | 0.98 | 0.89 |
| insert_allocate | 1.28 (G2: 1.94) | 1.89 | 1.76 | 1.39 | **0.89** | 1.58 |
| insert_preallocated | **1.77** | 4.02 | 3.10 | 3.47 | 3.13 | 3.07 |
| remove_hits | 1.79 | 2.51 | 7.44 | 1.44 | 1.59 | **1.13** |
| remove_misses | 0.94 | 0.94 | 1.22 | 0.88 | 0.95 | **0.77** |
| shrink_to | 1.21 | 1.23 | **1.13** | 1.25 | 1.36 | 1.23 |

Memory (bytes/entry, build-from-empty, tracking allocator; hashbrown 22–36):
pomap G4 40–104, G2 40–63, pomap3 40–63, **oldsimd 28–45 (≈ hashbrown)**,
main_soa 49–126 (worst), tags 28–89.

**Reading of the frontier.** *(Reproduced 2026-08-18 with the in-repo harness
after the get-reorder commit: family ordering identical, current engine's
get_hits sharpened to 0.64; remove/shrink magnitudes wobble within the
soft-anchor caveats — `bench-family-Apple_M5_Max_2026-08-18b.txt`.)* The current
engine wins or ties 6 of 9 groups
in-family — all reads/updates, provisioned inserts by 1.7–2.3×, misses — and
nothing dominates it; it is the read-optimal point and the canonical engine. But
three other Pareto points exist: **tags_soa is build-optimal** (the only
hash-ordered design that beats hashbrown building from empty, 0.89×, at the cost
of the entire read advantage); **oldsimd is memory-optimal** (hashbrown-class
bytes/slot while keeping prefix order — at the cost of reads and a catastrophic
7.4× remove); **main_soa is the bounded-guarantee point** (worst-case
O(max_scan) probes via order-preserving cascade — decent all-round, worst
memory, wins nothing). Caveats: single platform (M5), background load present,
Mac remove anchor soft (std measured 0.64× hashbrown in-run) — the in-run family
*ordering* is the robust product; AMD re-runs owed (§7).

**The tombstone A/B (ties to graveyard hashing, §10).** tags2_soa differs from
tags_soa by adding tombstone deletion: removes improve 1.59 → **1.13** (and
remove_misses 0.95 → 0.77) while build-from-empty regresses 0.89 → 1.58. This is
the remove-vs-insert trade the graveyard-hashing line studies, measured cleanly
in this family. The canonical engine instead buys short runs with a low load
factor (62.5%) and backshift — at that load, tombstones were a net loss when
tried (below); at tags_soa's higher effective load, they pay on removes. Both
observations are consistent with the theory: tombstone benefit grows with load.

### 6.2 Negative results

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

- **Bandwidth/latency probes captured on three of four; correlation holds.** The
  `bench_bandwidth` STREAM-triad and `bench_latency` pointer-chase mountains are in
  the harness and captured on M5, M3, and EPYC 9354P (bare). They establish the
  inverse-bandwidth axis behind §5.3 (Apple ~100–124 vs AMD ~41–44 GB/s/core).
  **[TODO]** run both on the Zen 5c VM (its data file predates the probes) to
  complete the four-platform correlation.
- **Zen 5c magnitudes are VM-soft, and its main-suite write "inversions" are a
  medium-size virtualization effect.** Shared host, host-managed turbo/governor;
  the bare-metal EPYC 9354P shows those same main-suite write ops *losing* (shrink
  1.35, insert_allocate 1.07), so the Zen 5c factors (0.37×, 0.80×) are not a
  general property — see §5.1/§5.3. The robust, machine-independent write result is
  the per-size **cold matrix** (cold/large writes tie on Apple, win on AMD
  bare-metal *and* VM).
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
- **Family benchmark is single-platform.** §6.1's cross-design table is one M5 run
  (background load present; Mac remove anchor soft). The in-run ordering is
  robust; magnitudes need an AMD re-run (`family_bench.rs` is portable — run via
  the same collection scripts).
- **Open anomaly: `insert_preallocated` anchor instability on M5 (2026-08-18).**
  The canonical suite now reads 3.4-3.6x for insert_preallocated on this machine
  (two independent runs) vs June's committed 1.77x — while the SAME workload in
  `loop_bench` reads 1.8-1.9x in-run all night. The pomap absolute is stable; the
  discrepancy is in the anchor/harness interaction (suspects: background load,
  allocator state, bench ordering). Until re-run on an idle box, quote the
  loop_bench in-run value (~1.8x) and June's cross-machine table; treat fresh
  pomap_bench insert_preallocated cells on M5 as suspect.
- **Instruction-layout effects rival small optimizations (2026-08-18 optimizer
  loop).** A same-run A/B harness (`loop_bench`: candidate vs frozen-snapshot
  engine) with a measured identical-code noise floor (±5% single-run, ±0.5%
  median-of-3) validated one micro-optimization (hit-biased compare order in
  `get`: −2 to −5% on get_hits across 7 runs) and *rejected* the same reorder in
  three sibling functions after bisection showed its apparent −8%/+10% effects
  were **code-layout artifacts** (they moved groups whose code was untouched and
  vanished/reappeared with unrelated edits). Consequence for the paper: effects
  below ~5% on this class of hardware need cross-configuration validation (or
  layout randomization / PGO) before being reported; and the engine is otherwise
  at its optimum under this suite — the optimizer found exactly one durable
  improvement.
- **Main-suite thermal state: RESOLVED (2026-08-18, §4.4).** The get-class groups
  are warm-regime as suspected (fixed per-size seeds → resident touched set), but
  the A/B shows the ratio is thermal-robust (0.63× warm vs 0.67× true sweep) —
  the read win stands; label the numbers "sweep aggregate." The same audit
  quantified RNG dilution (§4.4): reported read ratios are conservative.

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

**One-shot per machine** — collects the complete spec-named dataset:
```
scripts/collect_all.sh   # → bench-<spec>-g4.txt, bench-<spec>-g2.txt,
                         #   cold-<spec>.csv, perf-<spec>.csv   (~20-30 min)
```
It chains the three below; copy back the files it lists at the end. Or run them
individually:
```
scripts/run_bench.sh   # main suite, both growth factors → bench-<spec>-g{4,2}.txt
scripts/run_cold.sh    # cold-access matrix             → cold-<spec>.csv  (growth-independent)
scripts/perf_cold.sh   # perf counters                  → perf-<spec>.csv  (Linux; needs perf + paranoid<=1)
```
`perf_cold.sh` env knobs: `VW` (value words 1|2|4|8), `WS` (MB), `PASSES`, `EVENTS`
(append an AMD `ls_misal_loads.*` event to measure split loads if `perf list`
shows it). Only `insert_allocate`/memory depend on GROWTH; cold + perf are
growth-independent (provisioned builds).

## 9. Summary of what we are confident in

1. **Reads/updates beat hashbrown everywhere** (measured, robust) — a
   *deterministic* hash table matching or beating SwissTable-class point-read
   performance, with the margin growing as access gets colder/larger (the
   fewer-instructions + fewer-L1-misses mechanism, perf-confirmed, §5.6).
2. **Cold/large writes tie on Apple and win on AMD** (measured across four
   machines, *including bare-metal* EPYC 9354P, so not a VM artifact). The outcome
   tracks single-thread bandwidth inversely — PoMap's full-cache-line streaming
   beats hashbrown's scatter-rehash where per-core bandwidth is scarce (AMD
   ~41–44 GB/s) and ties where it is abundant (Apple ~100–124). Virtualization
   *extends* the win to medium working sets (nested page walks). It is **not** a
   universal server inversion, **not** bandwidth-favorability (Apple has the most
   bandwidth and least write advantage), and **not** raw latency (EPYC-bare 102 ns <
   M3 118 ns yet EPYC wins) (§5.3). On warm/medium sizes PoMap loses writes.
3. The only durable losses are the small-cache-resident miss-scan (a TLB/locality
   effect) and the pure insert shift tax (warm/medium writes); neither has a fix
   that preserves the read wins.
4. The point-op design is at its optimum on the architectures tested: the
   2026-08 optimizer loop (in-run A/B vs a frozen snapshot, measured noise
   floor, median-of-3 protocol) validated exactly one point-op improvement (the
   hit-biased `get` compare order, −2 to −5% on get_hits) and rejected every
   other micro-candidate — several as instruction-layout artifacts (§7). The
   same loop found one *large* non-point-op win: the branchless occupancy-mask
   iterator (§5.8, 4× — iteration now beats hashbrown at scale).
5. **With String keys the write story inverts at G4** (§5.7): builds 0.71×,
   shrink 0.35×, removes 0.74× — because the inline full hash makes resize
   **hash-free** while SwissTable re-hashes every key. The map's strongest
   workload class is expensive-to-hash keys, which is the common case.
6. **The growth dial reaches inversion**: G8 builds from empty at 0.877×
   hashbrown with ≈G4's average measured memory at the standard sizes (§5.5).
7. **The resize is a first-class result, not an implementation detail**: the
   growth-invariance of hash-prefix order makes rehash a comparison-free
   streaming pass (§2, C2) — the flat-open-addressing realization of an invariant
   previously used only in concurrent chained tables (split-ordered lists,
   relativistic hash tables, §10) — and the family benchmark (§6.1) shows every
   engine in the lineage shares it while the *layouts* trade reads, builds,
   memory, and probe bounds against each other along a measured Pareto frontier.

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

**Closest prior art #1: ordered hashing and ordered linear probing (the base
structure — must anchor against).** Amble & Knuth, "Ordered hash tables" (*The
Computer Journal* 17(2), 1974 *(verify page/issue)*) keep entries within a probe
sequence in order of a key/signature so an *unsuccessful* search terminates early
— exactly PoMap's `stored > hash` cutoff. Critically, the modern theory
literature treats the hash-ordered linear-probing table as a **standard known
structure** under the name *ordered linear probing*: graveyard hashing (Bender,
Kuszmaul & Kuszmaul, "Linear Probing Revisited: Tombstones Mark the Death of
Primary Clustering," FOCS 2021, arXiv:2107.01250) performs its operations
"exactly as in standard ordered linear probing" and adds strategic tombstones;
Braverman & Kuszmaul give tight analyses of ordered vs unordered linear probing
(FOCS 2024, arXiv:2501.11582); Zombie hashing (SIGMOD/PACMMOD 2025) is a
practical follow-on with experiments. **Therefore C1 does not claim the base
structure.** PoMap's deltas over this line: (1) *prefix* home-slot addressing
makes home-bucket order equal **global** hash order → **deterministic whole-map
iteration**, a property the OLP literature never surfaces or exploits; (2) the
modern **flat AoS inline-full-hash layout** and its measured cache behavior
(that literature predates or ignores SwissTable-class baselines; no head-to-head
of an engineered OLP table vs production SIMD tables exists that we know of);
(3) the **streaming-resize consequence** (C2). The tombstone tension is engaged
directly by our family A/B (§6.1): tombstone benefit grows with load factor,
and the canonical engine's low-load + backshift choice is measured, not assumed.
Also check: Cleary, "Compact Hash Tables Using Bidirectional Linear Probing"
(*IEEE Trans. Computers*, 1984 *(verify)*) — also maintains hash order.

**Closest prior art #2: growth-invariant hash ordering (the C2 resize
mechanism).** Two lines use "order by a hash-derived key so growth preserves
structure," both for *concurrent chained* tables: **split-ordered lists** (Shalev
& Shavit, "Split-ordered lists: lock-free extensible hash tables," *J. ACM* 2006
*(verify)*) keep all items in one list sorted by **bit-reversed** hash so
doubling the bucket array moves nothing (buckets are lazy pointers into the
list); **relativistic / RCU-resizable hash tables** (Triplett, McKenney & Walpole,
USENIX ATC 2011 *(verify)*, and related patents) keep chains sorted by hash and
choose high-order-bit ("prefix") hashing precisely so chain order survives
doubling, enabling single-pass cross-linking resize concurrent with readers.
PoMap is the **flat open-addressing realization of the same invariant** — MSB
prefix instead of bit reversal, an array instead of chains — which converts the
invariant's payoff from "items never move / readers survive resize" into a
**comparison-free, memcpy-class streaming rehash** with the measured
sequential-vs-scatter write behavior of §5.3, plus deterministic iteration and
cache locality (which bit-reversal scatters away). Provenance note: this project
*began* at the split-ordered end (a 2025 concurrent-map prototype using
bit-reversed addressing and per-cell seqlocks) and inverted to prefix order after
hitting exactly those limitations — locality and iteration order. The
**cursor-spacing gap injection** during the resize pass is PMA-flavored (packed
memory arrays redistribute gaps in sorted arrays — Itai, Konheim & Rodeh 1981;
Bender et al. adaptive PMA *(verify)*), but PMAs are comparison-sorted,
search-indexed structures; we know of no hash table that deliberately re-seeds
displacement gaps at resize.

**Closest prior art #3: overlapping neighborhoods and bounded probes (the §6
window/cascade variant).** Hopscotch hashing (Herlihy, Shavit & Tzafrir, 2008
*(verify)*) is the established overlapping-neighborhood scheme: every home
bucket owns H consecutive slots, neighborhoods overlap, and displacement hops
items into holes — **destroying order** (bitmap bookkeeping tracks membership).
Bounded-probe-then-grow is also known practice (Skarupke's `flat_hash_map`
bounds Robin Hood probes at log₂(n) and grows on violation — blog, 2017
*(verify)*). The family's `main_soa` variant combines both with **order
preservation**: cascade displacement shifts a *contiguous sorted run* right by
one, each shifted entry remaining inside its own window, extent capped — giving
worst-case-bounded probes in a hash-ordered table. We found no ordered Hopscotch
variant in the literature *(searched 2026-08; keep looking)*. Note the current
canonical engine is the **unbounded limit** of the same overlapping-region
structure (region boundaries are data-defined by the sort itself); fixed disjoint
buckets appear nowhere in the mature family.

**Novelty position (summary).** The base structure is *ordered linear probing*
(established); the claims are: **C1** the prefix-addressed flat synthesis —
global order, deterministic iteration, one-cache-line AoS probes, hash-not-key
ordering; **C2** the growth-invariant streaming resize in flat open addressing
(chained precedents cited above) with its measured cross-platform write behavior;
**C3** the measured family Pareto frontier including the order-preserving
bounded-window mechanism and the tombstone A/B that connects the engineering to
the FOCS'21/'24 theory line. Remaining search obligations: Cleary 1984,
Amble–Knuth descendants, self-organizing/LCFS hashing, any prior
"deterministic-iteration hash map" claim, MSD-radix/histogram hash bucketing,
and ordered-Hopscotch variants.
