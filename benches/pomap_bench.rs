use std::{collections::HashMap, hash::BuildHasherDefault, hint::black_box};

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

#[cfg(feature = "bench-string")]
const INPUT_SIZE: usize = 250_000_usize.next_power_of_two();
#[cfg(not(feature = "bench-string"))]
const INPUT_SIZE: usize = 20_000_000_usize.next_power_of_two();
const HOT_SET: usize = (INPUT_SIZE.ilog2() * INPUT_SIZE.ilog2() * INPUT_SIZE.ilog2()) as usize;

#[cfg(feature = "bench-string")]
const STR_LEN: usize = 128;

#[cfg(feature = "bench-string")]
fn random_string(rng: &mut StdRng) -> String {
    (0..STR_LEN)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect()
}

fn random_items(seed: u64, count: usize) -> Vec<BenchType> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            #[cfg(feature = "bench-string")]
            {
                random_string(&mut rng)
            }
            #[cfg(not(feature = "bench-string"))]
            {
                rng.random()
            }
        })
        .collect()
}

fn bench_insert_allocate(c: &mut Criterion) {
    let keys: Vec<BenchKey> = random_items(0xA11CE, INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0xFACE, INPUT_SIZE);
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
    let keys: Vec<BenchKey> = random_items(0xBEEF, INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0xF00D, INPUT_SIZE);
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
            let mut map: BenchHashMap = std_hashmap_with_capacity(INPUT_SIZE);
            for (key, val) in &combined {
                black_box(map.insert(key.clone(), val.clone()));
            }
        });
    });

    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let keys: Vec<BenchKey> = random_items(0xFEED, INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("get_hits");

    let mut map: BenchPoMap = BenchPoMap::new();
    for key in &keys {
        map.insert(key.clone(), values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;
        b.iter(|| {
            let key = keys[cursor].clone();
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap = new_std_hashmap();
    for key in &keys {
        map.insert(key.clone(), values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;
        b.iter(|| {
            let key = keys[cursor].clone();
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_get_misses(c: &mut Criterion) {
    let present_keys: Vec<BenchKey> = random_items(0xABA1, INPUT_SIZE);
    let present_values: Vec<BenchValue> = random_items(0xCAB, INPUT_SIZE);
    let miss_keys: Vec<BenchKey> = random_items(0xBA5E, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("get_misses");

    let mut map = BenchPoMap::new();
    for key in &present_keys {
        map.insert(key.clone(), present_values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = miss_keys[cursor].clone();
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap = new_std_hashmap();
    for key in &present_keys {
        map.insert(key.clone(), present_values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = miss_keys[cursor].clone();
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let keys: Vec<BenchKey> = random_items(0xC0FFEE, INPUT_SIZE);
    let initial_values: Vec<BenchValue> = random_items(0xABC, INPUT_SIZE);
    let update_values: Vec<BenchValue> = random_items(0xDEF, INPUT_SIZE);
    let mut cursor = 0usize;
    let mut group = c.comparison_benchmark_group("update_existing");

    let mut map: BenchPoMap = BenchPoMap::new();
    for key in &keys {
        map.insert(key.clone(), initial_values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor].clone();
            let val = update_values[cursor].clone();
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap = new_std_hashmap();
    for key in &keys {
        map.insert(key.clone(), initial_values[cursor].clone());
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor].clone();
            let val = update_values[cursor].clone();
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_hot_gets(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET)
        .map(|_| {
            #[cfg(feature = "bench-string")]
            {
                random_string(&mut rng)
            }
            #[cfg(not(feature = "bench-string"))]
            {
                rng.random()
            }
        })
        .collect();
    let mut map_keys: Vec<BenchKey> = hot_keys.clone();
    map_keys.extend(random_items(0xD00D, INPUT_SIZE - HOT_SET));
    let map_values: Vec<BenchValue> = random_items(0xBADD, INPUT_SIZE);

    let mut group = c.comparison_benchmark_group("get_hotset");

    let mut map: BenchPoMap = BenchPoMap::new();
    for (idx, key) in map_keys.iter().enumerate() {
        map.insert(key.clone(), map_values[idx].clone());
    }

    group.bench_function("pomap", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let key = hot_keys[cursor % HOT_SET].clone();
            cursor = (cursor + 1) % INPUT_SIZE;
            black_box(map.get(&key));
        });
    });

    let mut map: BenchHashMap = new_std_hashmap();
    for (idx, key) in map_keys.iter().enumerate() {
        map.insert(key.clone(), map_values[idx].clone());
    }

    group.bench_function("std_hashmap", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let key = hot_keys[cursor % HOT_SET].clone();
            cursor = (cursor + 1) % INPUT_SIZE;
            black_box(map.get(&key));
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
