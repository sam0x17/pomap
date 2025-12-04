use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
    hint::black_box,
};

use ahash::AHasher;
use criterion::{Criterion, criterion_group, criterion_main};
use pomap::pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Hasher configuration shared by PoMap and std::collections benchmarks.
type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;
type BenchPoMap<K, V> = PoMap<K, V, BenchHasher>;
type BenchHashMap<K, V> = HashMap<K, V, BenchHasherBuilder>;

#[inline]
fn new_std_hashmap<K, V>() -> BenchHashMap<K, V> {
    BenchHashMap::with_hasher(BenchHasherBuilder::default())
}

#[inline]
fn std_hashmap_with_capacity<K, V>(capacity: usize) -> BenchHashMap<K, V> {
    BenchHashMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

const INPUT_SIZE: usize = 1_000_000_usize.next_power_of_two();
const HOT_SET: usize = (INPUT_SIZE.ilog2() as usize).next_power_of_two()
    * (INPUT_SIZE.ilog2() as usize).next_power_of_two();

fn random_data(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

fn bench_insert_allocate(c: &mut Criterion) {
    let keys = random_data(0xA11CE, INPUT_SIZE);
    let values = random_data(0xFACE, INPUT_SIZE);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(u64, u64)>>();
    let mut group = c.comparison_benchmark_group("insert_allocate");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map: BenchPoMap<u64, u64> = BenchPoMap::new();
            for &(key, val) in &combined {
                black_box(map.insert(key, val));
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map: BenchHashMap<u64, u64> = new_std_hashmap();
            for &(key, val) in &combined {
                black_box(map.insert(key, val));
            }
        });
    });

    group.finish();
}

fn bench_insert_preallocated(c: &mut Criterion) {
    let keys = random_data(0xBEEF, INPUT_SIZE);
    let values = random_data(0xF00D, INPUT_SIZE);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(u64, u64)>>();
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    let mut map: BenchPoMap<u64, u64> = BenchPoMap::new();
    for &(key, val) in &combined {
        black_box(map.insert(key, val));
    }
    let initial_cap = map.capacity();
    drop(map);

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map: BenchPoMap<u64, u64> = BenchPoMap::with_capacity(initial_cap);
            for &(key, val) in &combined {
                black_box(map.insert(key, val));
            }
            assert_eq!(map.capacity(), initial_cap);
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map: BenchHashMap<u64, u64> = std_hashmap_with_capacity(INPUT_SIZE);
            for &(key, val) in &combined {
                black_box(map.insert(key, val));
            }
        });
    });

    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let keys = random_data(0xFEED, INPUT_SIZE);
    let values = random_data(0x1CEBEEF, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("get_hits");

    let mut map: BenchPoMap<u64, u64> = BenchPoMap::new();
    for &key in &keys {
        map.insert(key, values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap<u64, u64> = new_std_hashmap();
    for &key in &keys {
        map.insert(key, values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_get_misses(c: &mut Criterion) {
    let present_keys = random_data(0xABA1, INPUT_SIZE);
    let present_values = random_data(0xCAB, INPUT_SIZE);
    let miss_keys = random_data(0xBA5E, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("get_misses");

    let mut map = BenchPoMap::<u64, u64>::new();
    for &key in &present_keys {
        map.insert(key, present_values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = miss_keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap<u64, u64> = new_std_hashmap();
    for &key in &present_keys {
        map.insert(key, present_values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = miss_keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let keys = random_data(0xC0FFEE, INPUT_SIZE);
    let initial_values = random_data(0xABC, INPUT_SIZE);
    let update_values = random_data(0xDEF, INPUT_SIZE);
    let mut cursor = 0usize;
    let mut group = c.comparison_benchmark_group("update_existing");

    let mut map: BenchPoMap<u64, u64> = BenchPoMap::new();
    for &key in &keys {
        map.insert(key, initial_values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("pomap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor];
            let val = update_values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    let mut map: BenchHashMap<u64, u64> = new_std_hashmap();
    for &key in &keys {
        map.insert(key, initial_values[cursor]);
        cursor = (cursor + 1) % INPUT_SIZE;
    }

    group.bench_function("std_hashmap", |b| {
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor];
            let val = update_values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_hot_gets(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<u64> = (0..HOT_SET).map(|_| rng.random()).collect();
    let mut map_keys = hot_keys.clone();
    map_keys.extend(random_data(0xD00D, INPUT_SIZE - HOT_SET));
    let map_values = random_data(0xBADD, INPUT_SIZE);

    let mut group = c.comparison_benchmark_group("get_hotset");

    let mut map: BenchPoMap<u64, u64> = BenchPoMap::new();
    for (idx, &key) in map_keys.iter().enumerate() {
        map.insert(key, map_values[idx]);
    }

    group.bench_function("pomap", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let key = hot_keys[cursor % HOT_SET];
            cursor = (cursor + 1) % INPUT_SIZE;
            black_box(map.get(&key));
        });
    });

    let mut map: BenchHashMap<u64, u64> = new_std_hashmap();
    for (idx, &key) in map_keys.iter().enumerate() {
        map.insert(key, map_values[idx]);
    }

    group.bench_function("std_hashmap", |b| {
        let mut cursor = 0;
        b.iter(|| {
            let key = hot_keys[cursor % HOT_SET];
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
