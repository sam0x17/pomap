//! Full-map iteration benchmark — the missing coverage for PoMap's headline
//! feature (deterministic iteration order).
//!
//! Compares whole-map iteration throughput: PoMap (hash order, flat array at
//! ~62.5% density), hashbrown and std (unordered, control-byte guided), and
//! BTreeMap (the standard *ordered* baseline — key order, pointer-chasing).
//! Each timed pass sums keys and values so the iteration cannot be elided.
//!
//! Maps are built identically (same keys/values/seeds as the main suite's
//! generators); PoMap and the hash maps provision via with_capacity.

// Benchmark harness: style lints are silenced wholesale -- timed workload
// code must stay byte-comparable with the recorded methodology.
#![allow(clippy::all, unused_imports, dead_code)]

use std::{collections::BTreeMap, collections::HashMap, hash::BuildHasherDefault, hint::black_box};

use ahash::AHasher;
use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type BenchKey = u64;
type BenchValue = u64;
type BenchHasherBuilder = BuildHasherDefault<AHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type BenchHb = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchStd = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;

const SIZES: [usize; 3] = [1_000, 100_000, 1_000_000];

fn random_items(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

fn bench_iterate(c: &mut Criterion) {
    let max = *SIZES.iter().max().unwrap();
    let keys = random_items(0xFEED, max);
    let values = random_items(0x1CEBEEF, max);

    for &size in &SIZES {
        let mut group = c.comparison_benchmark_group(format!("iterate_{size}"));

        let mut pm: BenchPoMap =
            BenchPoMap::with_capacity_and_hasher(size, BenchHasherBuilder::default());
        for i in 0..size {
            pm.insert(keys[i], values[i]);
        }
        group.bench_function("pomap", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in pm.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            });
        });
        drop(pm);

        let mut hb: BenchHb =
            BenchHb::with_capacity_and_hasher(size, BenchHasherBuilder::default());
        for i in 0..size {
            hb.insert(keys[i], values[i]);
        }
        group.bench_function("hashbrown", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in hb.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            });
        });
        drop(hb);

        let mut sm: BenchStd =
            BenchStd::with_capacity_and_hasher(size, BenchHasherBuilder::default());
        for i in 0..size {
            sm.insert(keys[i], values[i]);
        }
        group.bench_function("std_hashmap", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in sm.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            });
        });
        drop(sm);

        let mut bt: BTreeMap<BenchKey, BenchValue> = BTreeMap::new();
        for i in 0..size {
            bt.insert(keys[i], values[i]);
        }
        group.bench_function("btreemap", |b| {
            b.iter(|| {
                let mut acc = 0u64;
                for (k, v) in bt.iter() {
                    acc = acc.wrapping_add(*k).wrapping_add(*v);
                }
                black_box(acc)
            });
        });
        drop(bt);

        group.finish();
    }
}

criterion_group!(benches, bench_iterate);
criterion_main!(benches);
