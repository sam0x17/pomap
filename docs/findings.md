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

**Central empirical claim.** PoMap's reads and updates beat hashbrown across
every microarchitecture tested. Its write/bulk operations (insert-with-growth,
remove, shrink) are *bandwidth-bound* sequential streaming operations, whereas
hashbrown's equivalents are scatter/rehash; consequently PoMap's standing on
those operations is a function of the machine's memory bandwidth rather than a
fixed property of the design. On bandwidth-rich server CPUs the write deficits
seen on laptop/desktop parts **invert into wins**. We therefore argue the design
*scales toward* modern server hardware.

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

| Tag | CPU | µarch | Environment | Notes |
|---|---|---|---|---|
| M-series | Apple Silicon (**[TODO] exact model**) | Apple M | macOS, laptop, AC power | dev machine; remove anchors *soft* (see 4.4) |
| Zen 4 | AMD EPYC 9354P | Zen 4 | Linux, rustc 1.90 | first x86 reference |
| Zen 5c | AMD EPYC 9845 "Turin Dense" | Zen 5c | Linux, **virtualized 8-vCPU guest**, AVX-512 (vp2intersect/vaes/gfni) | host-managed turbo/governor; ~3% noise floor |

**[TODO]** Add exact Apple model, all rustc versions, criterion fork revision,
and per-platform memory bandwidth (e.g. STREAM) to anchor the bandwidth argument
with a direct measurement rather than inference.

### 4.2 Software

- Rust (edition 2024), `release` profile: `opt-level=3`, `lto="fat"`,
  `codegen-units=1`.
- Hasher: `ahash` (`BuildHasherDefault<AHasher>`) for all maps, so the hash
  function is held constant across implementations.
- Baselines: `hashbrown` 0.14 and `std::collections::HashMap`. (Note: std bundles
  its *own* hashbrown version; the two are **not** identical — see 4.4.)
- Payloads: `u64 → u64` (a `String` variant exists behind a feature flag).
- Harness: `criterion` (a fork adding comparison-groups that print each
  implementation's ratio to the in-group winner).

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

PoMap wins all three get workloads and `update_existing` on every platform,
often by ~2× (`update_existing` 0.32× on Zen 5c). This is the inline-hash AoS
locality advantage — one cache line per probe versus hashbrown's control-byte
array plus a separate entry slab. The advantage *widens* with bandwidth.

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
hashbrown's ~17 B (1 control byte + 8 + 8), and runs at a lower load factor.
Analytically (GROWTH=4), bytes-per-entry versus hashbrown is **lumpy and ranges
~1.4–2.8×** depending on where N falls relative to a 4× growth boundary (≈1.41×
at ~60% load, ≈2.83× just after a grow at ~38% load). GROWTH=2 averages ~2.0×.

> **[TODO / known issue]** The in-benchmark memory report is unreliable: it prints
> `capacity()` (the logical 62.5% threshold) rather than the true allocated
> `total_slots`, and its figures do not match the current design's geometry. A
> correct memory-measurement harness is needed before any memory table is
> published. The numbers above are analytic.

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

- **Bandwidth thesis is supported by trend, not yet by direct measurement.**
  Three platforms in increasing-bandwidth order show the predicted inversion, but
  we have not measured each machine's memory bandwidth (e.g. STREAM) to plot the
  ratios against it. **[TODO]** — this is the single most important addition for a
  convincing paper.
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
- **Statistical rigor.** We report medians; a paper needs confidence intervals,
  documented run counts, and ideally more than 50 size points.
- **Related work.** **[TODO]** — situate against Swiss tables / Abseil flat_hash,
  Robin Hood hashing, F14, `BTreeMap` (the other ordered-map option), and prior
  "sorted/ordered open addressing" work. Citations to be added.

## 8. Reproducibility

Branch `simd-buckets`. Single implementation in `src/pomap.rs`; benchmark
`benches/pomap_bench.rs`. The `criterion` dev-dependency is a public git fork
(no sibling checkout needed). Run:

```
cargo test                                  # 42 tests, correctness gate
cargo bench --bench pomap_bench             # GROWTH=4 (default)
cargo bench --bench pomap_bench --features growth2   # GROWTH=2
```

Extract medians (criterion wraps long names):
```
awk '/^[a-z_]+\/[a-z0-9_]+/{n=$1} /time:/{for(i=1;i<=NF;i++)if($i=="time:"){print n"\t"$(i+2)" "$(i+3);break}}' out.txt | grep -v report
```

On servers: pin to a non-zero core (`taskset -c 2`), set the governor to
`performance` and disable turbo where possible, keep the box idle, and capture
full output (never pipe through `tail` — it truncates early groups).

## 9. Summary of what we are confident in

1. **Reads/updates beat hashbrown everywhere**, by a margin that grows with
   bandwidth (measured, robust).
2. **Write/bulk operations are bandwidth-bound and invert from deficit to win on
   high-BW server CPUs** (measured direction robust; magnitudes VM-soft pending
   bare metal).
3. The only durable losses are the small-cache-resident miss-scan and the pure
   insert shift tax; both narrow with bandwidth and neither has a fix that
   preserves the read wins.
4. The design is at/near its optimum on three microarchitectures — the optimizer
   found no validated gain on the latest — so the contribution is the
   architecture and its bandwidth-favorability, not further micro-optimization.
