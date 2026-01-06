use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    hint::black_box,
};

use ahash::AHasher;
use criterion::{Criterion, criterion_group, criterion_main};
use pomap::pomap::PoMap;
#[cfg(feature = "bench-string")]
use rand::distr::Alphanumeric;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Swap these type aliases to switch benchmark payloads.
#[cfg(feature = "bench-string")]
type BenchType = String;
#[cfg(not(feature = "bench-string"))]
type BenchType = u64;

type BenchKey = BenchType;
type BenchValue = BenchType;

/// Hasher configuration shared by PoMap and std::collections benchmarks.
type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasher>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;

#[inline]
fn new_std_hashmap() -> BenchHashMap {
    BenchHashMap::with_hasher(BenchHasherBuilder::default())
}

#[inline]
fn std_hashmap_with_capacity(capacity: usize) -> BenchHashMap {
    BenchHashMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

const INSERT_INPUT_SIZE: usize = 50_000_usize;
#[cfg(feature = "bench-string")]
const MAX_INPUT_SIZE: usize = 250_000_usize;
#[cfg(not(feature = "bench-string"))]
const MAX_INPUT_SIZE: usize = 10_000_000_usize;
const GETS_PER_ROUND: usize = 1_000_usize;
const NUM_INTERMEDIATE_ROUNDS: usize = 10_usize;
const HOT_SET: usize = MAX_INPUT_SIZE.isqrt();

#[cfg(feature = "bench-string")]
const STR_LEN: usize = 128;

#[cfg(feature = "bench-string")]
fn random_string(rng: &mut StdRng) -> String {
    (0..STR_LEN)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect()
}

fn random_bench_item(rng: &mut StdRng) -> BenchType {
    #[cfg(feature = "bench-string")]
    {
        random_string(rng)
    }
    #[cfg(not(feature = "bench-string"))]
    {
        rng.random()
    }
}

fn random_items(seed: u64, count: usize) -> Vec<BenchType> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| random_bench_item(&mut rng)).collect()
}

fn target_sizes() -> Vec<usize> {
    let mut power_targets = Vec::new();
    let mut current = 10usize;

    while current < MAX_INPUT_SIZE {
        power_targets.push(current);
        current *= 10;
    }
    power_targets.push(MAX_INPUT_SIZE);

    let rounds = NUM_INTERMEDIATE_ROUNDS.max(2);
    let mut targets = Vec::new();
    for window in power_targets.windows(2) {
        let start = window[0];
        let end = window[1];
        for step in 0..rounds {
            let size = start + (end - start) * step / (rounds - 1);
            if targets.last().copied() != Some(size) {
                targets.push(size);
            }
        }
    }

    if targets.is_empty() {
        targets.push(MAX_INPUT_SIZE);
    }

    targets
}

fn build_pomap_maps_from_data(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchPoMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let mut map: BenchPoMap = BenchPoMap::with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_std_maps_from_data(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchHashMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn bench_insert_allocate(c: &mut Criterion) {
    let keys: Vec<BenchKey> = random_items(0xA11CE, INSERT_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0xFACE, INSERT_INPUT_SIZE);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(BenchKey, BenchValue)>>();
    let mut group = c.comparison_benchmark_group("insert_allocate");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map: BenchPoMap = BenchPoMap::new();
            for (key, val) in &combined {
                black_box(map.insert(key.clone(), val.clone()));
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map: BenchHashMap = new_std_hashmap();
            for (key, val) in &combined {
                black_box(map.insert(key.clone(), val.clone()));
            }
        });
    });

    group.finish();
}

fn bench_insert_preallocated(c: &mut Criterion) {
    let keys: Vec<BenchKey> = random_items(0xBEEF, INSERT_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0xF00D, INSERT_INPUT_SIZE);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(BenchKey, BenchValue)>>();
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    let mut map: BenchPoMap = BenchPoMap::new();
    for (key, val) in &combined {
        black_box(map.insert(key.clone(), val.clone()));
    }
    let initial_cap = map.capacity();
    let initial_scan = map.max_scan();
    let initial_requested = initial_cap - initial_scan;
    drop(map);

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map: BenchPoMap = BenchPoMap::with_capacity(initial_requested);
            // assert_eq!(map.capacity(), initial_cap);
            for (key, val) in &combined {
                black_box(map.insert(key.clone(), val.clone()));
            }
            // assert_eq!(map.max_scan(), initial_scan);
            // assert_eq!(map.len(), INPUT_SIZE);
            // assert_eq!(map.capacity(), initial_cap);
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map: BenchHashMap = std_hashmap_with_capacity(INSERT_INPUT_SIZE);
            for (key, val) in &combined {
                black_box(map.insert(key.clone(), val.clone()));
            }
        });
    });

    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let target_sizes = target_sizes();
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, MAX_INPUT_SIZE);
    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &keys, &values);
    let mut group = c.comparison_benchmark_group("get_hits");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &(size, ref map) in &pomap_maps {
                for idx in 0..GETS_PER_ROUND {
                    let key = &keys[idx % size];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(pomap_maps);
    let std_maps = build_std_maps_from_data(&target_sizes, &keys, &values);

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for &(size, ref map) in &std_maps {
                for idx in 0..GETS_PER_ROUND {
                    let key = &keys[idx % size];
                    black_box(map.get(key));
                }
            }
        });
    });

    group.finish();
}

fn bench_get_misses(c: &mut Criterion) {
    let target_sizes = target_sizes();
    let present_keys: Vec<BenchKey> = random_items(0xABA1, MAX_INPUT_SIZE);
    let present_values: Vec<BenchValue> = random_items(0xCAB, MAX_INPUT_SIZE);

    let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(present_keys.len(), BenchHasherBuilder::default());
    for key in &present_keys {
        present_set.insert(key.clone());
    }

    let mut miss_rng = StdRng::seed_from_u64(0xBA5E);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(MAX_INPUT_SIZE);
    while miss_keys.len() < MAX_INPUT_SIZE {
        let candidate = random_bench_item(&mut miss_rng);
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }

    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &present_keys, &present_values);
    let mut group = c.comparison_benchmark_group("get_misses");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &(size, ref map) in &pomap_maps {
                for idx in 0..GETS_PER_ROUND {
                    let key = &miss_keys[idx % size];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(pomap_maps);
    let std_maps = build_std_maps_from_data(&target_sizes, &present_keys, &present_values);

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for &(size, ref map) in &std_maps {
                for idx in 0..GETS_PER_ROUND {
                    let key = &miss_keys[idx % size];
                    black_box(map.get(key));
                }
            }
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let target_sizes = target_sizes();
    let keys: Vec<BenchKey> = random_items(0xC0FFEE, MAX_INPUT_SIZE);
    let initial_values: Vec<BenchValue> = random_items(0xABC, MAX_INPUT_SIZE);
    let update_values: Vec<BenchValue> = random_items(0xDEF, MAX_INPUT_SIZE);
    let mut pomap_maps = build_pomap_maps_from_data(&target_sizes, &keys, &initial_values);
    let mut group = c.comparison_benchmark_group("update_existing");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for (size, map) in pomap_maps.iter_mut() {
                let size = *size;
                for idx in 0..GETS_PER_ROUND {
                    let key = keys[idx % size].clone();
                    let val = update_values[idx % size].clone();
                    black_box(map.insert(key, val));
                }
            }
        });
    });

    drop(pomap_maps);
    let mut std_maps = build_std_maps_from_data(&target_sizes, &keys, &initial_values);

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for (size, map) in std_maps.iter_mut() {
                let size = *size;
                for idx in 0..GETS_PER_ROUND {
                    let key = keys[idx % size].clone();
                    let val = update_values[idx % size].clone();
                    black_box(map.insert(key, val));
                }
            }
        });
    });

    group.finish();
}

fn bench_hot_gets(c: &mut Criterion) {
    let target_sizes = target_sizes();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET).map(|_| random_bench_item(&mut rng)).collect();
    let mut map_keys: Vec<BenchKey> = hot_keys.clone();
    map_keys.extend(random_items(0xD00D, MAX_INPUT_SIZE - HOT_SET));
    let map_values: Vec<BenchValue> = random_items(0xBADD, MAX_INPUT_SIZE);
    let hot_counts: Vec<usize> = target_sizes
        .iter()
        .map(|&size| HOT_SET.min(size).max(1))
        .collect();
    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &map_keys, &map_values);

    let mut group = c.comparison_benchmark_group("get_hotset");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for ((_, map), &hot_count) in pomap_maps.iter().zip(&hot_counts) {
                for idx in 0..GETS_PER_ROUND {
                    let key = &hot_keys[idx % hot_count];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(pomap_maps);
    let std_maps = build_std_maps_from_data(&target_sizes, &map_keys, &map_values);

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for ((_, map), &hot_count) in std_maps.iter().zip(&hot_counts) {
                for idx in 0..GETS_PER_ROUND {
                    let key = &hot_keys[idx % hot_count];
                    black_box(map.get(key));
                }
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_allocate,
    bench_insert_preallocated,
    bench_get_hits,
    bench_get_misses,
    bench_update,
    bench_hot_gets
);
criterion_main!(benches);
