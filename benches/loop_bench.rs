//! Optimizer A/B harness: the LIVE engine (`src/pomap.rs`, the candidate being
//! edited) vs a FROZEN snapshot of the last-known-good engine
//! (`benches/support/base_snapshot.rs`) vs hashbrown — all in one process, so
//! candidate/base ratios are immune to run-to-run drift (the only trustworthy
//! comparison on this codebase; see findings §4.4).
//!
//! Differences from `pomap_bench.rs` (deliberate; see the 2026-08-18 audit):
//! - No RNG inside timed loops: index sequences are pre-generated, removing the
//!   ~12.8 µs/iteration additive overhead that compresses ratios toward 1.0.
//! - Only three impls (cand / base / hashbrown) to keep iterations fast.
//! - Workload shapes, sizes, and seeds otherwise match `pomap_bench.rs`.
//!
//! Iteration 0 runs with cand == base: every cand/base ratio should be 1.00,
//! and the observed spread IS the per-group noise floor for keep/revert calls.

// Benchmark harness: style lints are silenced wholesale -- timed workload
// code must stay byte-comparable with the recorded methodology.
#![allow(clippy::all, unused_imports, dead_code)]

extern crate alloc;

use std::{collections::HashSet, hash::BuildHasherDefault, hint::black_box};

use ahash::AHasher;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[path = "support/base_snapshot.rs"]
#[allow(dead_code, unused_imports, clippy::all)]
mod base_snapshot;

type BenchKey = u64;
type BenchValue = u64;
type BenchHasherBuilder = BuildHasherDefault<AHasher>;

type Cand = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type CandG2 = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 2>;
type CandG8 = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 8>;
type Base = base_snapshot::PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type Hb = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;

const MAX_GET_INPUT_SIZE: usize = 1_000_000_usize;
const MAX_INSERT_INPUT_SIZE: usize = 100_000_usize;
const GETS_PER_ROUND: usize = 100_usize;
const NUM_INTERMEDIATE_ROUNDS: usize = 50_usize;
const HOT_SET: usize = MAX_GET_INPUT_SIZE.isqrt();

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

fn get_target_sizes() -> Vec<usize> {
    evenly_spaced_sizes(10, MAX_GET_INPUT_SIZE, NUM_INTERMEDIATE_ROUNDS)
}

fn insert_target_sizes() -> Vec<usize> {
    evenly_spaced_sizes(10, MAX_INSERT_INPUT_SIZE, NUM_INTERMEDIATE_ROUNDS)
}

/// Pre-generated per-size probe index sequences (same seeds as pomap_bench's
/// in-loop RNG so the visited keys match the established workload).
fn pregen_indices(target_sizes: &[usize], seed_base: u64) -> Vec<Vec<u32>> {
    target_sizes
        .iter()
        .map(|&size| {
            let mut rng = StdRng::seed_from_u64(seed_base ^ size as u64);
            (0..GETS_PER_ROUND)
                .map(|_| rng.random_range(0..size) as u32)
                .collect()
        })
        .collect()
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
    ($ty:ty, $target_sizes:expr, $keys:expr, $values:expr, cap_mult = $mult:expr) => {{
        $target_sizes
            .iter()
            .map(|&size| {
                let capacity = size.saturating_mul($mult).max(size);
                let mut map: $ty =
                    <$ty>::with_capacity_and_hasher(capacity, BenchHasherBuilder::default());
                for idx in 0..size {
                    map.insert($keys[idx], $values[idx]);
                }
                (size, map)
            })
            .collect::<Vec<(usize, $ty)>>()
    }};
}

fn bench_insert_allocate(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max = *target_sizes.iter().max().unwrap();
    let keys = random_items(0xA11CE, max);
    let values = random_items(0xFACE, max);
    let mut group = c.comparison_benchmark_group("insert_allocate");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &size in &target_sizes {
                        let mut map: $ty = <$ty>::with_hasher(BenchHasherBuilder::default());
                        for (key, val) in keys.iter().zip(values.iter()).take(size) {
                            black_box(map.insert(*key, *val));
                        }
                        black_box(&map);
                    }
                });
            });
        };
    }
    variant!(Cand, "cand");
    // Growth-factor dial points (G4 is the default `cand`): only this group
    // depends on GROWTH — all other groups provision via with_capacity.
    variant!(CandG2, "cand_g2");
    variant!(CandG8, "cand_g8");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_insert_preallocated(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max = *target_sizes.iter().max().unwrap();
    let keys = random_items(0xA11CE, max);
    let values = random_items(0xFACE, max);
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &size in &target_sizes {
                        let mut map: $ty =
                            <$ty>::with_capacity_and_hasher(size, BenchHasherBuilder::default());
                        for (key, val) in keys.iter().zip(values.iter()).take(size) {
                            black_box(map.insert(*key, *val));
                        }
                        black_box(&map);
                    }
                });
            });
        };
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let values = random_items(0x1CEBEEF, MAX_GET_INPUT_SIZE);
    let idx = pregen_indices(&target_sizes, 0xC01DBEEF);
    let mut group = c.comparison_benchmark_group("get_hits");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for ((_, map), ids) in maps.iter().zip(&idx) {
                        for &i in ids {
                            black_box(map.get(&keys[i as usize]));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_get_misses(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let present_keys = random_items(0xABA1, MAX_GET_INPUT_SIZE);
    let present_values = random_items(0xCAB, MAX_GET_INPUT_SIZE);

    let mut present: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(present_keys.len(), BenchHasherBuilder::default());
    for k in &present_keys {
        present.insert(*k);
    }
    let mut miss_rng = StdRng::seed_from_u64(0xBA5E);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(MAX_GET_INPUT_SIZE);
    while miss_keys.len() < MAX_GET_INPUT_SIZE {
        let cand: BenchKey = miss_rng.random();
        if !present.contains(&cand) {
            miss_keys.push(cand);
        }
    }
    let idx = pregen_indices(&target_sizes, 0xC0FFEE42);
    let mut group = c.comparison_benchmark_group("get_misses");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &present_keys, &present_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for ((_, map), ids) in maps.iter().zip(&idx) {
                        for &i in ids {
                            black_box(map.get(&miss_keys[i as usize]));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys = random_items(0xC0FFEE, MAX_GET_INPUT_SIZE);
    let initial = random_items(0xABC, MAX_GET_INPUT_SIZE);
    let update_values = random_items(0xDEF, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("update_existing");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let mut maps = build_maps!($ty, &target_sizes, &keys, &initial);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for (size, map) in maps.iter_mut() {
                        let size = *size;
                        for idx in 0..GETS_PER_ROUND {
                            let key = &keys[idx % size];
                            let val = update_values[idx % size];
                            if let Some(v) = map.get_mut(key) {
                                black_box(*v = val);
                            }
                        }
                    }
                });
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_hot_gets(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET).map(|_| rng.random()).collect();
    let mut map_keys = hot_keys.clone();
    map_keys.extend(random_items(0xD00D, MAX_GET_INPUT_SIZE - HOT_SET));
    let map_values = random_items(0xBADD, MAX_GET_INPUT_SIZE);
    let hot_counts: Vec<usize> = target_sizes
        .iter()
        .map(|&size| HOT_SET.min(size).max(1))
        .collect();
    // Pre-generate hot index sequences per map (seeded as pomap_bench).
    let idx: Vec<Vec<u32>> = hot_counts
        .iter()
        .map(|&hc| {
            let mut rng = StdRng::seed_from_u64(0xDEC0DE42 ^ hc as u64);
            (0..GETS_PER_ROUND)
                .map(|_| rng.random_range(0..hc) as u32)
                .collect()
        })
        .collect();
    let mut group = c.comparison_benchmark_group("get_hotset");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &map_keys, &map_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for ((_, map), ids) in maps.iter().zip(&idx) {
                        for &i in ids {
                            black_box(map.get(&hot_keys[i as usize]));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_remove_hits(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max = *target_sizes.iter().max().unwrap();
    let keys = random_items(0xD15EA5E, max);
    let values = random_items(0xBEEFC0DE, max);
    let mut group = c.comparison_benchmark_group("remove_hits");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values);
            group.bench_function($label, |b| {
                b.iter_batched(
                    || {
                        maps.iter()
                            .map(|(size, map)| (*size, map.clone()))
                            .collect::<Vec<(usize, $ty)>>()
                    },
                    |mut batch| {
                        for (size, map) in batch.iter_mut() {
                            let removes = GETS_PER_ROUND.min(*size);
                            for idx in 0..removes {
                                black_box(map.remove(&keys[idx]));
                            }
                        }
                        batch
                    },
                    BatchSize::LargeInput,
                );
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_remove_misses(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max = *target_sizes.iter().max().unwrap();
    let present_keys = random_items(0xC0FFEE, max);
    let present_values = random_items(0xBADF00D, max);

    let mut present: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(present_keys.len(), BenchHasherBuilder::default());
    for k in &present_keys {
        present.insert(*k);
    }
    let mut miss_rng = StdRng::seed_from_u64(0xF00DFACE);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(max);
    while miss_keys.len() < max {
        let cand: BenchKey = miss_rng.random();
        if !present.contains(&cand) {
            miss_keys.push(cand);
        }
    }
    let mut group = c.comparison_benchmark_group("remove_misses");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let mut maps = build_maps!($ty, &target_sizes, &present_keys, &present_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for (size, map) in maps.iter_mut() {
                        let removes = GETS_PER_ROUND.min(*size);
                        for idx in 0..removes {
                            black_box(map.remove(&miss_keys[idx]));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

fn bench_shrink_to(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max = *target_sizes.iter().max().unwrap();
    let keys = random_items(0x5A11CE, max);
    let values = random_items(0xC0FFEE55, max);
    let mut group = c.comparison_benchmark_group("shrink_to");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values, cap_mult = 8);
            group.bench_function($label, |b| {
                b.iter_batched(
                    || {
                        maps.iter()
                            .map(|(size, map)| (*size, map.clone()))
                            .collect::<Vec<(usize, $ty)>>()
                    },
                    |mut batch| {
                        for (size, map) in batch.iter_mut() {
                            black_box(map.shrink_to(*size));
                        }
                        batch
                    },
                    BatchSize::LargeInput,
                );
            });
            drop(maps);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    variant!(Hb, "hashbrown");
    group.finish();
}

/// Whole-map iteration at 1M — A/Bs the masked iterator (cand) against the
/// pre-optimization per-slot iterator (base) in-run, per architecture.
fn bench_iterate_1m(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let keys = random_items(0xFEED, N);
    let values = random_items(0x1CEBEEF, N);
    let mut group = c.comparison_benchmark_group("iterate_1m");

    macro_rules! variant {
        ($ty:ty, $label:expr) => {{
            let mut m: $ty = <$ty>::with_capacity_and_hasher(N, BenchHasherBuilder::default());
            for i in 0..N {
                m.insert(keys[i], values[i]);
            }
            group.bench_function($label, |b| {
                b.iter(|| {
                    let mut acc = 0u64;
                    for (k, v) in m.iter() {
                        acc = acc.wrapping_add(*k).wrapping_add(*v);
                    }
                    black_box(acc)
                });
            });
            drop(m);
        }};
    }
    variant!(Cand, "cand");
    variant!(Base, "base");
    {
        let mut m: Hb = Hb::with_capacity_and_hasher(N, BenchHasherBuilder::default());
        for i in 0..N {
            m.insert(keys[i], values[i]);
        }
        group.bench_function("hashbrown", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in m.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_allocate,
    bench_insert_preallocated,
    bench_get_hits,
    bench_get_misses,
    bench_update,
    bench_hot_gets,
    bench_remove_hits,
    bench_remove_misses,
    bench_shrink_to,
    bench_iterate_1m
);
criterion_main!(benches);
