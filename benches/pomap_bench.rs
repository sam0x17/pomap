use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
};

use ahash::AHasher;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// Tracking allocator — measures net heap bytes held by a map after construction
// ---------------------------------------------------------------------------

static NET_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct TrackingAlloc;

unsafe impl GlobalAlloc for TrackingAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        NET_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            NET_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size > layout.size() {
                NET_ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                NET_ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOC: TrackingAlloc = TrackingAlloc;
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
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
/// PoMap growth factor: default 4; `--features growth2` benches GROWTH=2
/// (e.g. `cargo bench --bench pomap_bench --features growth2`).
#[cfg(feature = "growth2")]
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 2>;
#[cfg(not(feature = "growth2"))]
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type BenchHashMap = HashMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchHashbrownMap = HashbrownMap<BenchKey, BenchValue, BenchHasherBuilder>;

#[inline]
fn new_std_hashmap() -> BenchHashMap {
    BenchHashMap::with_hasher(BenchHasherBuilder::default())
}

#[inline]
fn std_hashmap_with_capacity(capacity: usize) -> BenchHashMap {
    BenchHashMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

#[inline]
fn new_hashbrown_hashmap() -> BenchHashbrownMap {
    BenchHashbrownMap::with_hasher(BenchHasherBuilder::default())
}

#[inline]
fn hashbrown_with_capacity(capacity: usize) -> BenchHashbrownMap {
    BenchHashbrownMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default())
}

const MAX_GET_INPUT_SIZE: usize = 1_000_000_usize;
const MAX_INSERT_INPUT_SIZE: usize = 100_000_usize;
const GETS_PER_ROUND: usize = 100_usize;
const NUM_INTERMEDIATE_ROUNDS: usize = 50_usize;
const HOT_SET: usize = MAX_GET_INPUT_SIZE.isqrt();

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

fn build_pomap_maps_from_data(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchPoMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let mut map: BenchPoMap =
                BenchPoMap::with_capacity_and_hasher(size, BenchHasherBuilder::default());
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_pomap_maps_from_data_with_capacity(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
    capacity_multiplier: usize,
) -> Vec<(usize, BenchPoMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let capacity = size.saturating_mul(capacity_multiplier).max(size);
            let mut map: BenchPoMap =
                BenchPoMap::with_capacity_and_hasher(capacity, BenchHasherBuilder::default());
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

fn build_hashbrown_maps_from_data(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
) -> Vec<(usize, BenchHashbrownMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_std_maps_from_data_with_capacity(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
    capacity_multiplier: usize,
) -> Vec<(usize, BenchHashMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let capacity = size.saturating_mul(capacity_multiplier).max(size);
            let mut map: BenchHashMap = std_hashmap_with_capacity(capacity);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn build_hashbrown_maps_from_data_with_capacity(
    target_sizes: &[usize],
    keys: &[BenchKey],
    values: &[BenchValue],
    capacity_multiplier: usize,
) -> Vec<(usize, BenchHashbrownMap)> {
    target_sizes
        .iter()
        .map(|&size| {
            let capacity = size.saturating_mul(capacity_multiplier).max(size);
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(capacity);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            (size, map)
        })
        .collect()
}

fn bench_insert_allocate(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xA11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_target_size);
    let mut group = c.comparison_benchmark_group("insert_allocate");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchPoMap = BenchPoMap::with_hasher(BenchHasherBuilder::default());
                for (key, val) in keys.iter().zip(values.iter()).take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchHashMap = new_std_hashmap();
                for (key, val) in keys.iter().zip(values.iter()).take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchHashbrownMap = new_hashbrown_hashmap();
                for (key, val) in keys.iter().zip(values.iter()).take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.finish();
}

fn bench_insert_preallocated(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes
        .iter()
        .max()
        .expect("target sizes should not be empty");
    let keys: Vec<BenchKey> = random_items(0xA11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_target_size);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(BenchKey, BenchValue)>>();
    // Each implementation pre-allocates for `size` entries using its own strategy.
    // This is the fairest comparison: each decides its own allocation for the workload.
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchPoMap =
                    BenchPoMap::with_capacity_and_hasher(size, BenchHasherBuilder::default());
                for (key, val) in combined.iter().take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchHashMap = std_hashmap_with_capacity(size);
                for (key, val) in combined.iter().take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
                for (key, val) in combined.iter().take(size) {
                    black_box(map.insert(key.clone(), val.clone()));
                }
                black_box(&map);
            }
        });
    });

    group.finish();
}

fn bench_get_hits(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, MAX_GET_INPUT_SIZE);
    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &keys, &values);
    let mut group = c.comparison_benchmark_group("get_hits");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &(size, ref map) in &pomap_maps {
                let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &keys[idx];
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
                let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(std_maps);
    let hashbrown_maps = build_hashbrown_maps_from_data(&target_sizes, &keys, &values);

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &(size, ref map) in &hashbrown_maps {
                let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });



    group.finish();
}

fn bench_get_misses(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let present_keys: Vec<BenchKey> = random_items(0xABA1, MAX_GET_INPUT_SIZE);
    let present_values: Vec<BenchValue> = random_items(0xCAB, MAX_GET_INPUT_SIZE);

    let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(present_keys.len(), BenchHasherBuilder::default());
    for key in &present_keys {
        present_set.insert(key.clone());
    }

    let mut miss_rng = StdRng::seed_from_u64(0xBA5E);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(MAX_GET_INPUT_SIZE);
    while miss_keys.len() < MAX_GET_INPUT_SIZE {
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
                let mut rng = StdRng::seed_from_u64(0xC0FFEE42 ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &miss_keys[idx];
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
                let mut rng = StdRng::seed_from_u64(0xC0FFEE42 ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &miss_keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(std_maps);
    let hashbrown_maps =
        build_hashbrown_maps_from_data(&target_sizes, &present_keys, &present_values);

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &(size, ref map) in &hashbrown_maps {
                let mut rng = StdRng::seed_from_u64(0xC0FFEE42 ^ size as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..size);
                    let key = &miss_keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(hashbrown_maps);


    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys: Vec<BenchKey> = random_items(0xC0FFEE, MAX_GET_INPUT_SIZE);
    let initial_values: Vec<BenchValue> = random_items(0xABC, MAX_GET_INPUT_SIZE);
    let update_values: Vec<BenchValue> = random_items(0xDEF, MAX_GET_INPUT_SIZE);
    let mut pomap_maps = build_pomap_maps_from_data(&target_sizes, &keys, &initial_values);
    let mut group = c.comparison_benchmark_group("update_existing");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for (size, map) in pomap_maps.iter_mut() {
                let size = *size;
                for idx in 0..GETS_PER_ROUND {
                    let key = &keys[idx % size];
                    let val = update_values[idx % size].clone();
                    if let Some(v) = map.get_mut(key) {
                        black_box(*v = val);
                    }
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
                    let key = &keys[idx % size];
                    let val = update_values[idx % size].clone();
                    if let Some(v) = map.get_mut(key) {
                        black_box(*v = val);
                    }
                }
            }
        });
    });

    drop(std_maps);
    let mut hashbrown_maps = build_hashbrown_maps_from_data(&target_sizes, &keys, &initial_values);

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for (size, map) in hashbrown_maps.iter_mut() {
                let size = *size;
                for idx in 0..GETS_PER_ROUND {
                    let key = &keys[idx % size];
                    let val = update_values[idx % size].clone();
                    if let Some(v) = map.get_mut(key) {
                        black_box(*v = val);
                    }
                }
            }
        });
    });

    drop(hashbrown_maps);


    group.finish();
}

fn bench_hot_gets(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET).map(|_| random_bench_item(&mut rng)).collect();
    let mut map_keys: Vec<BenchKey> = hot_keys.clone();
    map_keys.extend(random_items(0xD00D, MAX_GET_INPUT_SIZE - HOT_SET));
    let map_values: Vec<BenchValue> = random_items(0xBADD, MAX_GET_INPUT_SIZE);
    let hot_counts: Vec<usize> = target_sizes
        .iter()
        .map(|&size| HOT_SET.min(size).max(1))
        .collect();
    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &map_keys, &map_values);

    let mut group = c.comparison_benchmark_group("get_hotset");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for ((_, map), &hot_count) in pomap_maps.iter().zip(&hot_counts) {
                let mut rng = StdRng::seed_from_u64(0xDEC0DE42 ^ hot_count as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..hot_count);
                    let key = &hot_keys[idx];
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
                let mut rng = StdRng::seed_from_u64(0xDEC0DE42 ^ hot_count as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..hot_count);
                    let key = &hot_keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(std_maps);
    let hashbrown_maps = build_hashbrown_maps_from_data(&target_sizes, &map_keys, &map_values);

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for ((_, map), &hot_count) in hashbrown_maps.iter().zip(&hot_counts) {
                let mut rng = StdRng::seed_from_u64(0xDEC0DE42 ^ hot_count as u64);
                for _ in 0..GETS_PER_ROUND {
                    let idx = rng.random_range(0..hot_count);
                    let key = &hot_keys[idx];
                    black_box(map.get(key));
                }
            }
        });
    });

    drop(hashbrown_maps);


    group.finish();
}

fn bench_remove_hits(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes
        .iter()
        .max()
        .expect("target sizes should not be empty");
    let keys: Vec<BenchKey> = random_items(0xD15EA5E, max_target_size);
    let values: Vec<BenchValue> = random_items(0xBEEFC0DE, max_target_size);
    let pomap_maps = build_pomap_maps_from_data(&target_sizes, &keys, &values);
    let mut group = c.comparison_benchmark_group("remove_hits");

    group.bench_function("pomap", |b| {
        b.iter_batched(
            || {
                pomap_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchPoMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    let removes = GETS_PER_ROUND.min(*size);
                    for idx in 0..removes {
                        let key = &keys[idx];
                        black_box(map.remove(key));
                    }
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(pomap_maps);
    let std_maps = build_std_maps_from_data(&target_sizes, &keys, &values);
    group.bench_function("std_hashmap", |b| {
        b.iter_batched(
            || {
                std_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    let removes = GETS_PER_ROUND.min(*size);
                    for idx in 0..removes {
                        let key = &keys[idx];
                        black_box(map.remove(key));
                    }
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(std_maps);
    let hashbrown_maps = build_hashbrown_maps_from_data(&target_sizes, &keys, &values);
    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || {
                hashbrown_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashbrownMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    let removes = GETS_PER_ROUND.min(*size);
                    for idx in 0..removes {
                        let key = &keys[idx];
                        black_box(map.remove(key));
                    }
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(hashbrown_maps);
    group.finish();
}

fn bench_remove_misses(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes
        .iter()
        .max()
        .expect("target sizes should not be empty");
    let present_keys: Vec<BenchKey> = random_items(0xC0FFEE, max_target_size);
    let present_values: Vec<BenchValue> = random_items(0xBADF00D, max_target_size);

    let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(present_keys.len(), BenchHasherBuilder::default());
    for key in &present_keys {
        present_set.insert(key.clone());
    }

    let mut miss_rng = StdRng::seed_from_u64(0xF00DFACE);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(max_target_size);
    while miss_keys.len() < max_target_size {
        let candidate = random_bench_item(&mut miss_rng);
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }

    let mut pomap_maps = build_pomap_maps_from_data(&target_sizes, &present_keys, &present_values);
    let mut group = c.comparison_benchmark_group("remove_misses");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for (size, map) in pomap_maps.iter_mut() {
                let removes = GETS_PER_ROUND.min(*size);
                for idx in 0..removes {
                    let key = &miss_keys[idx];
                    black_box(map.remove(key));
                }
            }
        });
    });

    drop(pomap_maps);
    let mut std_maps = build_std_maps_from_data(&target_sizes, &present_keys, &present_values);

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for (size, map) in std_maps.iter_mut() {
                let removes = GETS_PER_ROUND.min(*size);
                for idx in 0..removes {
                    let key = &miss_keys[idx];
                    black_box(map.remove(key));
                }
            }
        });
    });

    drop(std_maps);
    let mut hashbrown_maps =
        build_hashbrown_maps_from_data(&target_sizes, &present_keys, &present_values);

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for (size, map) in hashbrown_maps.iter_mut() {
                let removes = GETS_PER_ROUND.min(*size);
                for idx in 0..removes {
                    let key = &miss_keys[idx];
                    black_box(map.remove(key));
                }
            }
        });
    });

    drop(hashbrown_maps);


    group.finish();
}

fn bench_shrink_to(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes
        .iter()
        .max()
        .expect("target sizes should not be empty");
    let keys: Vec<BenchKey> = random_items(0x5A11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xC0FFEE55, max_target_size);
    let pomap_maps = build_pomap_maps_from_data_with_capacity(&target_sizes, &keys, &values, 8);
    let mut group = c.comparison_benchmark_group("shrink_to");

    group.bench_function("pomap", |b| {
        b.iter_batched(
            || {
                pomap_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchPoMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    black_box(map.shrink_to(*size));
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(pomap_maps);
    let std_maps = build_std_maps_from_data_with_capacity(&target_sizes, &keys, &values, 8);
    group.bench_function("std_hashmap", |b| {
        b.iter_batched(
            || {
                std_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    black_box(map.shrink_to(*size));
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(std_maps);
    let hashbrown_maps =
        build_hashbrown_maps_from_data_with_capacity(&target_sizes, &keys, &values, 8);
    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || {
                hashbrown_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashbrownMap)>>()
            },
            |mut maps| {
                for (size, map) in maps.iter_mut() {
                    black_box(map.shrink_to(*size));
                }
                maps // return the batch so its drop is not timed
            },
            BatchSize::LargeInput,
        );
    });

    drop(hashbrown_maps);
    group.finish();
}

fn measure_map_bytes<F: FnOnce() -> R, R>(build: F) -> (usize, R) {
    let before = NET_ALLOCATED.load(Ordering::Relaxed);
    let map = build();
    let bytes = NET_ALLOCATED.load(Ordering::Relaxed).saturating_sub(before);
    (bytes, map)
}

/// STREAM-triad memory-bandwidth "mountain": sustained throughput at working-set
/// sizes spanning L1 → DRAM. Gives each platform run a self-reported bandwidth
/// curve to correlate against PoMap's bandwidth-bound write/bulk results. The
/// working set is `3 * n * 8` bytes (arrays a/b/c of f64); throughput counts the
/// two reads + one write per element (24 B). Single-threaded, pinned by the
/// caller's `taskset`.
fn bench_bandwidth(_c: &mut Criterion) {
    use std::time::Instant;

    // (elements per array, human label for the 3-array working set)
    let configs: [(usize, &str); 6] = [
        (1 << 10, "24 KiB (L1)"),
        (1 << 13, "192 KiB (L2)"),
        (1 << 16, "1.5 MiB (L2/L3)"),
        (1 << 19, "12 MiB (L3)"),
        (1 << 22, "96 MiB (L3/DRAM)"),
        (1 << 24, "384 MiB (DRAM)"),
    ];
    let max_n = 1usize << 24;
    let mut a = vec![0.0f64; max_n];
    // Index-dependent data so the triad can't be constant-folded.
    let b: Vec<f64> = (0..max_n).map(|i| i as f64).collect();
    let cc: Vec<f64> = (0..max_n).map(|i| (i % 7) as f64).collect();
    let scalar = 3.0f64;

    println!("\n{:<20} {:>12} {:>14}", "working set", "per array", "triad GB/s");
    println!("{}", "-".repeat(50));
    for (n, label) in configs {
        let a = &mut a[..n];
        let b = &b[..n];
        let cc = &cc[..n];
        // ~256M element-updates of total work per size (min 5 iters for stability).
        let iters = (256_000_000usize / n).max(5);
        // Warm the working set into cache (for the levels where it fits).
        for i in 0..n {
            a[i] = b[i] + scalar * cc[i];
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            for i in 0..n {
                a[i] = b[i] + scalar * cc[i];
            }
            // Force each outer iteration to be observed so successive triads
            // aren't collapsed, without pessimizing the inner vectorized loop.
            black_box(a.as_ptr());
        }
        let secs = t0.elapsed().as_secs_f64();
        let gbps = (n as f64 * 24.0 * iters as f64) / secs / 1e9;
        println!(
            "{:<20} {:>9} KiB {:>13.1}",
            label,
            n * 8 / 1024,
            gbps
        );
    }
    black_box(&a);
}

/// Pointer-chase latency "mountain": load-to-use latency of a dependent random
/// load chain at working-set sizes spanning L1 → DRAM. Each element is the index
/// of the next, forming a single random cycle (Sattolo), so loads are serialized
/// and unpredictable — ILP and prefetch cannot hide them, isolating latency
/// (vs `bench_bandwidth`, which measures throughput). The cache "cliffs" reveal
/// each platform's cache sizes; the DRAM plateau is its random-access latency.
/// Together they separate the write/bulk inversion's candidate drivers — cache
/// residency vs raw latency (vs TLB/page-walk, visible as a high DRAM plateau).
/// Single-threaded, pinned by the caller's `taskset`.
fn bench_latency(_c: &mut Criterion) {
    use std::time::Instant;

    let configs: [(usize, &str); 6] = [
        (1 << 11, "16 KiB (L1)"),
        (1 << 14, "128 KiB (L2)"),
        (1 << 17, "1 MiB (L2/L3)"),
        (1 << 20, "8 MiB (L3)"),
        (1 << 23, "64 MiB (L3/DRAM)"),
        (1 << 24, "128 MiB (DRAM)"),
    ];
    let max_n = 1usize << 24;
    let mut next = vec![0usize; max_n];
    let mut rng = StdRng::seed_from_u64(0x1A7E11C7);

    println!("\n{:<20} {:>12} {:>14}", "working set", "size", "latency ns");
    println!("{}", "-".repeat(48));
    for (n, label) in configs {
        let next = &mut next[..n];
        // Single random cycle over [0, n) via Sattolo's algorithm: position k of a
        // cyclic order points at position k+1, so chasing visits all n with no
        // short sub-cycles. Build `order` then link it into `next`.
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.random_range(0..i)); // j in [0,i): single n-cycle
        }
        for k in 0..n {
            next[order[k]] = order[(k + 1) % n];
        }
        // ~64M dependent loads (min 1 full cycle); each can't start until the prior
        // load retires, so wall-time / count ≈ load-to-use latency.
        let chases = (64_000_000usize / n).max(1) * n;
        let mut p = order[0];
        let t0 = Instant::now();
        for _ in 0..chases {
            p = next[p];
        }
        let secs = t0.elapsed().as_secs_f64();
        black_box(p);
        println!(
            "{:<20} {:>9} KiB {:>13.2}",
            label,
            n * 8 / 1024,
            secs / chases as f64 * 1e9
        );
    }
    black_box(&next);
}

fn bench_memory_footprint(c: &mut Criterion) {
    // Reports the REAL retained heap footprint of each map (net bytes held after
    // a build-from-empty), measured by the tracking global allocator — not a
    // logical capacity() estimate. bytes/entry and the pm/hb ratio are the
    // paper-ready memory metrics. Build maps via `with_hasher` (default growth)
    // so the growth-step geometry is exercised exactly as in real use.
    let sizes = [500usize, 5_000, 50_000, 500_000, 5_000_000];
    let max_size = *sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xB17E5, max_size);
    let values: Vec<BenchValue> = random_items(0xF007, max_size);

    println!(
        "\n{:<10} {:>14} {:>14} {:>11} {:>11} {:>9}",
        "entries", "pomap bytes", "hb bytes", "pm B/ent", "hb B/ent", "pm/hb"
    );
    println!("{}", "-".repeat(74));

    for &size in &sizes {
        let (pm_bytes, pm) = measure_map_bytes(|| {
            let mut m = BenchPoMap::with_hasher(BenchHasherBuilder::default());
            for i in 0..size {
                m.insert(keys[i].clone(), values[i].clone());
            }
            m
        });
        drop(black_box(pm));

        let (hb_bytes, hb) = measure_map_bytes(|| {
            let mut m = new_hashbrown_hashmap();
            for i in 0..size {
                m.insert(keys[i].clone(), values[i].clone());
            }
            m
        });
        drop(black_box(hb));

        println!(
            "{:<10} {:>14} {:>14} {:>11.1} {:>11.1} {:>8.2}x",
            size,
            pm_bytes,
            hb_bytes,
            pm_bytes as f64 / size as f64,
            hb_bytes as f64 / size as f64,
            pm_bytes as f64 / hb_bytes.max(1) as f64,
        );
    }

    // No-op benchmark so `-- memory_footprint_bytes` filter still triggers this group.
    let mut group = c.benchmark_group("memory_footprint_bytes");
    group.bench_function("report", |b| b.iter(|| {}));
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
    bench_memory_footprint,
    bench_bandwidth,
    bench_latency
);

criterion_main!(benches);
