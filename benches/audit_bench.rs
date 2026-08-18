//! Methodology audit for the main-suite get benchmarks.
//!
//! Questions this answers empirically:
//! 1. **Thermal state**: the suite's per-size RNG seeds are fixed, so every
//!    criterion iteration touches the SAME ~100 keys per map (~5000 entries
//!    total, a few hundred KB) — after warmup those lines are cache-resident.
//!    Does the pomap/hashbrown ratio change when seeds vary per iteration
//!    (fresh keys every pass → true sweep, mixed warm-small/cold-large)?
//! 2. **RNG dilution**: the timed loop includes StdRng setup per size plus one
//!    `random_range` per get. That cost is identical for all impls, so it
//!    compresses ratios toward 1.0. How large is it relative to a map get?
//!
//! Groups: get_hits_fixed (current semantics), get_hits_vary (per-iteration
//! seeds), rng_overhead (identical loop, no map access).

use std::{cell::Cell, collections::HashMap, hash::BuildHasherDefault, hint::black_box};

use ahash::AHasher;
use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type BenchKey = u64;
type BenchValue = u64;
type BenchHasherBuilder = BuildHasherDefault<AHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type BenchHashbrownMap = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;

const MAX_GET_INPUT_SIZE: usize = 1_000_000_usize;
const GETS_PER_ROUND: usize = 100_usize;
const NUM_INTERMEDIATE_ROUNDS: usize = 50_usize;

fn random_items(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

fn evenly_spaced_sizes(min: usize, max: usize, num_points: usize) -> Vec<usize> {
    let n = num_points.max(2);
    let mut sizes = Vec::with_capacity(n);
    for i in 0..n {
        let size = min + (max - min) * i / (n - 1);
        if sizes.last().copied() != Some(size) {
            sizes.push(size);
        }
    }
    sizes
}

macro_rules! build_maps {
    ($ty:ty, $target_sizes:expr, $keys:expr, $values:expr) => {{
        $target_sizes
            .iter()
            .map(|&size| {
                let mut map: $ty =
                    <$ty>::with_capacity_and_hasher(size, BenchHasherBuilder::default());
                for idx in 0..size {
                    map.insert($keys[idx], $values[idx]);
                }
                (size, map)
            })
            .collect::<Vec<(usize, $ty)>>()
    }};
}

/// Current main-suite semantics: per-size seed fixed → identical keys each iteration.
fn bench_get_hits_fixed(c: &mut Criterion) {
    let target_sizes = evenly_spaced_sizes(10, MAX_GET_INPUT_SIZE, NUM_INTERMEDIATE_ROUNDS);
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("get_hits_fixed");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &(size, ref map) in &maps {
                        let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                        for _ in 0..GETS_PER_ROUND {
                            let idx = rng.random_range(0..size);
                            let key = &keys[idx];
                            black_box(map.get(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    variant!(BenchPoMap, "pomap");
    variant!(BenchHashbrownMap, "hashbrown");
    variant!(BenchHashMap, "std_hashmap");
    group.finish();
}

/// Audit variant: seed varies per iteration → fresh random keys every pass.
/// Defeats cross-iteration cache residency of the touched set; large maps'
/// accesses become genuinely cold, small maps stay resident (a true sweep).
fn bench_get_hits_vary(c: &mut Criterion) {
    let target_sizes = evenly_spaced_sizes(10, MAX_GET_INPUT_SIZE, NUM_INTERMEDIATE_ROUNDS);
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("get_hits_vary");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values);
            let iter_counter = Cell::new(0u64);
            group.bench_function($label, |b| {
                b.iter(|| {
                    let it = iter_counter.get();
                    iter_counter.set(it.wrapping_add(1));
                    for &(size, ref map) in &maps {
                        let mut rng =
                            StdRng::seed_from_u64(0xC01DBEEF ^ size as u64 ^ (it << 32));
                        for _ in 0..GETS_PER_ROUND {
                            let idx = rng.random_range(0..size);
                            let key = &keys[idx];
                            black_box(map.get(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    variant!(BenchPoMap, "pomap");
    variant!(BenchHashbrownMap, "hashbrown");
    variant!(BenchHashMap, "std_hashmap");
    group.finish();
}

/// The timed loop with the map access replaced by black_box of the index:
/// measures the RNG + loop overhead that is ADDED equally to every impl's
/// number in the get groups (ratio-compression floor).
fn bench_rng_overhead(c: &mut Criterion) {
    let target_sizes = evenly_spaced_sizes(10, MAX_GET_INPUT_SIZE, NUM_INTERMEDIATE_ROUNDS);
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("rng_overhead");

    group.bench_function("loop_only", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    black_box(&keys[idx]);
                }
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_get_hits_fixed,
    bench_get_hits_vary,
    bench_rng_overhead
);
criterion_main!(benches);
