# pomap

A **prefix-ordered hash map** for Rust: a flat open-addressing table kept
globally sorted by hash, with each entry's home slot derived from its hash
*prefix* (top bits). `no_std` + `alloc` compatible.

**The determinism contract:** (1) iteration order is deterministic regardless
of insertion order, always; (2) after `compact()`, content-equal maps have
identical bytes. No unordered hash map offers either.

One invariant — *global hash order survives growth* (the prefix map is monotone
in the hash) — pays repeatedly:

- **Deterministic iteration order** (by hash), independent of insertion order.
- **One-cache-line probes**: the full 64-bit hash lives inline with its (K, V),
  so a lookup touches one contiguous region and terminates at the first stored
  hash greater than the target (`u64::MAX` doubles as the empty sentinel).
- **Comparison-free streaming resize**: rehash is a single sequential pass with
  a monotone write cursor — no re-hashing (the hash is stored), no re-sorting,
  no scatter. With expensive-to-hash keys (strings, paths, composites) this
  inverts resize-heavy workloads outright.
- **Canonical representation on demand**: `compact()` repacks into the minimal
  table with layout a pure function of contents — content-equal maps are
  byte-identical after compaction, whatever their histories (memcmp-tested).
- **The map itself is `Eq`, `Ord`, `PartialOrd`, and `Hash`**: canonical
  iteration order makes map-level comparison and hashing lawful — maps as keys
  in maps, sets of maps, `sort()`able collections of maps. Key comparisons
  execute only to canonicalize full 64-bit hash collisions (≈ n²/2⁶⁵ — never on
  any probe, lookup, or resize path).

## Measured highlights (vs `hashbrown`, in-run, Apple M5 Max; see `docs/findings.md` for the full multi-platform story and every caveat)

| workload | ratio (lower = pomap faster) |
|---|---|
| point reads (`get`, hot-set, update) | **0.56–0.60×** |
| whole-map iteration @ 1M | **0.61×** |
| build from empty (`u64` keys, GROWTH=8 / 4) | **0.88× / ~1.0×** |
| build / shrink / remove (`String` keys, GROWTH=4) | **0.71× / 0.35× / 0.74×** |
| misses, warm removes, pre-provisioned inserts | 0.9–1.8× (the order-maintenance tax) |
| memory (bytes/entry, avg) | ~2.0× (G2) – 2.8× (G4) |

Cold-access behavior, four-platform results (Apple M3/M5, AMD EPYC Zen 4
bare-metal, Zen 5c VM), perf-counter mechanisms, the design-space Pareto
frontier across seven sibling engines, and all negative results are documented
in [`docs/findings.md`](docs/findings.md) — the working draft toward a paper.

## Usage

```rust
use pomap::PoMap;

let mut map: PoMap<u64, u64> = PoMap::new();
map.insert(1, 10);
assert_eq!(map.get(&1), Some(&10));

// Deterministic iteration order (hash order), regardless of insertion order.
for (k, v) in map.iter() { /* ... */ }

// Provision so n inserts trigger zero grows (hashbrown-compatible contract):
let mut m: PoMap<u64, u64> = PoMap::with_capacity(1_000_000);
```

The growth factor is a const generic dial (power of two, default 4):

```rust
use core::hash::BuildHasherDefault;
type Fast = pomap::PoMap<u64, u64, BuildHasherDefault<ahash::AHasher>, 8>; // fastest builds
type Lean = pomap::PoMap<u64, u64, BuildHasherDefault<ahash::AHasher>, 2>; // leanest memory
```

Feature flags: `std` (default; disable for `no_std` + `alloc`), `tui` (the
`bench_graph` explorer), `bench-string` (String-payload benchmarks),
`growth2` (bench the GROWTH=2 configuration).

## Benchmarks

| target | what it measures |
|---|---|
| `pomap_bench` | the 9-group main suite vs `hashbrown`/`std` (canonical, cross-machine) |
| `cold.rs` | per-size cold-access matrix (value size × working set × op) |
| `family_bench` | all seven historical engine designs under one harness |
| `loop_bench` | optimizer A/B: live engine vs frozen pre-optimization snapshot |
| `iter_bench` | whole-map iteration vs hashbrown/std/BTreeMap |
| `audit_bench` | methodology probes (thermal state, RNG dilution) |
| `cold_perf` | hardware-counter runs (Linux `perf`) |

`scripts/collect_paper.sh` runs the full per-machine collection with
spec-named outputs; `scripts/collect_all.sh` is the shorter classic set.
Numbers are only comparable **within a run** (normalize to the in-run
`hashbrown` anchor); see `docs/findings.md` §4 for methodology.

## Status

Research-grade: the engine is stable, fuzzed against `std::HashMap` (50k-op
randomized runs, both growth factors, `u64` and `String` payloads), and the
design space around it has been mapped and measured. The API mirrors
`std::collections::HashMap` (entry-less: `get`/`get_mut`/`insert`/`remove`/
iterators/`retain`/`drain`/`shrink_to`/`reserve`/`try_reserve`).
