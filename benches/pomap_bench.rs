use std::{collections::HashMap, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
use pomap::pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

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
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("insert_allocate");

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::new();
        b.iter(|| {
            let key = keys[cursor];
            let val = values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> = HashMap::new();
        b.iter(|| {
            let key = keys[cursor];
            let val = values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_insert_preallocated(c: &mut Criterion) {
    let keys = random_data(0xBEEF, INPUT_SIZE);
    let values = random_data(0xF00D, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity((INPUT_SIZE + 1).next_power_of_two());
        b.iter(|| {
            let key = keys[cursor];
            let val = values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> =
            HashMap::with_capacity((INPUT_SIZE + 1).next_power_of_two());
        b.iter(|| {
            let key = keys[cursor];
            let val = values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let keys = random_data(0xFEED, INPUT_SIZE);
    let values = random_data(0x1CEBEEF, INPUT_SIZE);
    let mut cursor = 0;
    let mut group = c.comparison_benchmark_group("get_hits");

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(keys.len().next_power_of_two());
        for &key in &keys {
            map.insert(key, values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> = HashMap::with_capacity(keys.len().next_power_of_two());
        for &key in &keys {
            map.insert(key, values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
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

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(present_keys.len().next_power_of_two());
        for &key in &present_keys {
            map.insert(key, present_values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
        cursor = 0;

        b.iter(|| {
            let key = miss_keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> =
            HashMap::with_capacity(present_keys.len().next_power_of_two());
        for &key in &present_keys {
            map.insert(key, present_values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
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

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(keys.len().next_power_of_two());
        for &key in &keys {
            map.insert(key, initial_values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
        cursor = 0;

        b.iter(|| {
            let key = keys[cursor];
            let val = update_values[cursor];
            black_box(map.insert(key, val));
            cursor = (cursor + 1) % INPUT_SIZE;
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> = HashMap::with_capacity(keys.len().next_power_of_two());
        for &key in &keys {
            map.insert(key, initial_values[cursor]);
            cursor = (cursor + 1) % INPUT_SIZE;
        }
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

    group.bench_function("pomap", |b| {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(map_keys.len().next_power_of_two());
        for (idx, &key) in map_keys.iter().enumerate() {
            map.insert(key, map_values[idx]);
        }

        let mut cursor = 0usize;
        b.iter(|| {
            let key = hot_keys[cursor % HOT_SET];
            cursor = (cursor + 1) % INPUT_SIZE;
            black_box(map.get(&key));
        });
    });

    group.bench_function("std_hashmap", |b| {
        let mut map: HashMap<u64, u64> = HashMap::with_capacity(map_keys.len().next_power_of_two());
        for (idx, &key) in map_keys.iter().enumerate() {
            map.insert(key, map_values[idx]);
        }

        let mut cursor = 0usize;
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
