use std::collections::HashMap;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pomap::PoMap;
use rand::{SeedableRng, rngs::StdRng};

const INPUT_SIZE: usize = 100_0000;

fn random_keys() -> Vec<i32> {
    let mut rng = StdRng::from_entropy();
    (0..INPUT_SIZE)
        .map(|_| rand::Rng::r#gen::<i32>(&mut rng))
        .collect()
}

fn bench_insert_allocate(c: &mut Criterion) {
    let keys = random_keys();
    let mut cursor = 0;
    let mut group = c.benchmark_group("insert_allocate");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(0);
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(0);
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    group.finish();
}

fn bench_insert_no_allocate(c: &mut Criterion) {
    let keys = random_keys();
    let mut cursor = 0;
    let mut group = c.benchmark_group("insert_no_allocate");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(INPUT_SIZE * 4);
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(INPUT_SIZE * 4);
        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let keys = random_keys();
    let mut cursor = 0;
    let mut group = c.benchmark_group("get");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(0);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(0);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.get(&key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let keys = random_keys();
    let mut cursor = 0;
    let mut group = c.benchmark_group("update");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(INPUT_SIZE * 4);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(INPUT_SIZE * 4);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let keys = random_keys();
    let mut cursor = 0;
    let mut group = c.benchmark_group("remove");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(INPUT_SIZE * 4);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.remove(&key));
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    cursor = 0;

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(INPUT_SIZE * 4);
        for &key in &keys {
            map.insert(key, key);
        }

        b.iter(|| {
            let key = keys[cursor];
            black_box(map.remove(&key));
            black_box(map.insert(key, key));
            cursor = (cursor + 1) % INPUT_SIZE;
            1
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_allocate,
    bench_insert_no_allocate,
    bench_get,
    bench_update,
    bench_remove
);
criterion_main!(benches);
