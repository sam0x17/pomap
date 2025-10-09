use std::collections::HashMap;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pomap::PoMap;

const INPUT_SIZE: usize = 10_000;

fn bench_insert_allocate(c: &mut Criterion) {
    let keys: Vec<i32> = (0..INPUT_SIZE as i32).collect();
    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map = PoMap::with_capacity(0);
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
            black_box(map.len());
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map = HashMap::with_capacity(0);
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
            black_box(map.len());
        });
    });

    group.finish();
}

fn bench_insert_no_allocate(c: &mut Criterion) {
    let keys: Vec<i32> = (0..INPUT_SIZE as i32).collect();
    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        b.iter(|| {
            let mut map = PoMap::with_capacity(INPUT_SIZE * 2);
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
            black_box(map.len());
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            let mut map = HashMap::with_capacity(INPUT_SIZE * 2);
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
            black_box(map.len());
        });
    });

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let keys: Vec<i32> = (0..INPUT_SIZE as i32).collect();
    let mut group = c.benchmark_group("get");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(0);
        for key in &keys {
            map.insert(*key, *key);
        }

        b.iter(|| {
            for key in &keys {
                black_box(map.get(key));
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(0);
        for key in &keys {
            map.insert(*key, *key);
        }

        b.iter(|| {
            for key in &keys {
                black_box(map.get(key));
            }
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let keys: Vec<i32> = (0..INPUT_SIZE as i32).collect();
    let mut group = c.benchmark_group("update");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        let mut map = PoMap::with_capacity(INPUT_SIZE * 2);
        for key in &keys {
            map.insert(*key, *key);
        }

        b.iter(|| {
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        let mut map = HashMap::with_capacity(INPUT_SIZE * 2);
        for key in &keys {
            map.insert(*key, *key);
        }

        b.iter(|| {
            for key in &keys {
                black_box(map.insert(*key, *key));
            }
        });
    });

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let keys: Vec<i32> = (0..INPUT_SIZE as i32).collect();
    let mut group = c.benchmark_group("remove");
    group.throughput(Throughput::Elements(INPUT_SIZE as u64));

    group.bench_function("pomap", |b| {
        b.iter_batched(
            || {
                let mut map = PoMap::with_capacity(INPUT_SIZE);
                for key in &keys {
                    map.insert(*key, *key);
                }
                map
            },
            |mut map| {
                for key in &keys {
                    black_box(map.remove(key));
                }
                black_box(map.len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std_hashmap", |b| {
        b.iter_batched(
            || {
                let mut map = HashMap::with_capacity(INPUT_SIZE);
                for key in &keys {
                    map.insert(*key, *key);
                }
                map
            },
            |mut map| {
                for key in &keys {
                    black_box(map.remove(key));
                }
                black_box(map.len());
            },
            BatchSize::SmallInput,
        );
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
