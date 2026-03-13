use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
};

use ahash::AHasher;
#[cfg(not(feature = "get-graph"))]
use criterion::criterion_main;
use criterion::{BatchSize, Criterion, criterion_group};

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
#[cfg(feature = "get-graph")]
use rand::seq::SliceRandom;

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
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
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
const NUM_INTERMEDIATE_ROUNDS: usize = 5_usize;
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

fn get_target_sizes() -> Vec<usize> {
    let mut power_targets = Vec::new();
    let mut current = 10usize;

    while current < MAX_GET_INPUT_SIZE {
        power_targets.push(current);
        current *= 10;
    }
    power_targets.push(MAX_GET_INPUT_SIZE);

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
        targets.push(MAX_GET_INPUT_SIZE);
    }

    targets
}

fn insert_target_sizes() -> Vec<usize> {
    let mut power_targets = Vec::new();
    let mut current = 10usize;

    while current < MAX_INSERT_INPUT_SIZE {
        power_targets.push(current);
        current *= 10;
    }
    power_targets.push(MAX_INSERT_INPUT_SIZE);
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
        targets.push(MAX_INSERT_INPUT_SIZE);
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
    let max_target_size = *target_sizes
        .iter()
        .max()
        .expect("target sizes should not be empty");
    let keys: Vec<BenchKey> = random_items(0xA11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_target_size);
    let mut group = c.comparison_benchmark_group("insert_allocate");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchPoMap =
                        BenchPoMap::with_hasher(BenchHasherBuilder::default());
                    for (key, val) in keys.iter().zip(values.iter()).take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
            }
        });
    });

    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchHashMap = new_std_hashmap();
                    for (key, val) in keys.iter().zip(values.iter()).take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
            }
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &size in &target_sizes {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchHashbrownMap = new_hashbrown_hashmap();
                    for (key, val) in keys.iter().zip(values.iter()).take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
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
    // Pre-compute the capacity pomap needs so that inserting `size` entries causes zero
    // grows. Then give all three implementations the same capacity so they compete on
    // equal memory footprints. For pomap, grows are driven by window overflow (not load
    // factor), so we iteratively insert-and-grow until the capacity stabilizes.
    let preallocated_capacities: Vec<usize> = target_sizes
        .iter()
        .map(|&size| {
            let mut requested = size;
            loop {
                let mut map: BenchPoMap =
                    BenchPoMap::with_capacity_and_hasher(requested, BenchHasherBuilder::default());
                let cap_before = map.capacity();
                for (key, val) in combined.iter().take(size) {
                    map.insert(key.clone(), val.clone());
                }
                let cap_after = map.capacity();
                if cap_after == cap_before {
                    return cap_after - map.max_scan();
                }
                requested = cap_after - map.max_scan();
            }
        })
        .collect();
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    group.bench_function("pomap", |b| {
        b.iter(|| {
            for (i, &size) in target_sizes.iter().enumerate() {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchPoMap = BenchPoMap::with_capacity_and_hasher(
                        preallocated_capacities[i],
                        BenchHasherBuilder::default(),
                    );
                    for (key, val) in combined.iter().take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
            }
        });
    });

    // std/hashbrown interpret with_capacity(N) as "hold N elements before growing",
    // internally allocating ~N*8/7 slots. To match pomap's slot count we request N*7/8,
    // but never less than `size` (so they don't grow when pomap wouldn't).
    group.bench_function("std_hashmap", |b| {
        b.iter(|| {
            for (i, &size) in target_sizes.iter().enumerate() {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchHashMap =
                        std_hashmap_with_capacity((preallocated_capacities[i] * 7 / 8).max(size));
                    for (key, val) in combined.iter().take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
            }
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for (i, &size) in target_sizes.iter().enumerate() {
                let mut iterations = 0;
                while iterations < max_target_size {
                    let mut map: BenchHashbrownMap =
                        hashbrown_with_capacity((preallocated_capacities[i] * 7 / 8).max(size));
                    for (key, val) in combined.iter().take(size) {
                        black_box(map.insert(key.clone(), val.clone()));
                    }
                    iterations += size;
                }
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
                    let val = update_values[idx % size];
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
                    let val = update_values[idx % size];
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
                    let val = update_values[idx % size];
                    if let Some(v) = map.get_mut(key) {
                        black_box(*v = val);
                    }
                }
            }
        });
    });

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
            },
            BatchSize::LargeInput,
        );
    });

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
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn measure_map_bytes<F: FnOnce() -> R, R>(build: F) -> (usize, R) {
    let before = NET_ALLOCATED.load(Ordering::Relaxed);
    let map = build();
    let bytes = NET_ALLOCATED.load(Ordering::Relaxed).saturating_sub(before);
    (bytes, map)
}

fn bench_memory_footprint(c: &mut Criterion) {
    // Slot sizes for the current BenchKey/BenchValue types.
    // PoMap:     8 (hash u64) + size_of::<K>() + size_of::<V>()
    // hashbrown: 1 (control)  + size_of::<K>() + size_of::<V>()
    // std HashMap is backed by hashbrown so its slot size is the same.
    const HB_SLOT: usize = 1 + std::mem::size_of::<BenchKey>() + std::mem::size_of::<BenchValue>();

    let sizes = [500usize, 5_000, 50_000, 500_000, 5_000_000];
    let max_size = *sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xB17E5, max_size);
    let values: Vec<BenchValue> = random_items(0xF007, max_size);

    println!(
        "\n{:<12} {:>14} {:>14} {:>12} {:>12} {:>12}",
        "entries", "pomap slots", "hb slots", "pm/hb", "pm load%", "hb load%"
    );
    println!("{}", "-".repeat(80));

    for &size in &sizes {
        let (_, pm) = measure_map_bytes(|| {
            let mut m = BenchPoMap::with_hasher(BenchHasherBuilder::default());
            for i in 0..size {
                m.insert(keys[i].clone(), values[i].clone());
            }
            m
        });
        let pm_slots = pm.capacity();
        drop(black_box(pm));

        let (hb_bytes, hb) = measure_map_bytes(|| {
            let mut m = new_hashbrown_hashmap();
            for i in 0..size {
                m.insert(keys[i].clone(), values[i].clone());
            }
            m
        });
        let hb_slots = hb_bytes / HB_SLOT;
        drop(black_box(hb));

        println!(
            "{:<12} {:>14} {:>14} {:>11.2}x {:>11.1}% {:>11.1}%",
            size,
            pm_slots,
            hb_slots,
            pm_slots as f64 / hb_slots.max(1) as f64,
            size as f64 / pm_slots as f64 * 100.0,
            size as f64 / hb_slots.max(1) as f64 * 100.0,
        );
    }

    // No-op benchmark so `-- memory_footprint_bytes` filter still triggers this group.
    let mut group = c.benchmark_group("memory_footprint_bytes");
    group.bench_function("report", |b| b.iter(|| {}));
    group.finish();
}

#[cfg(feature = "get-graph")]
fn get_graph() {
    use std::collections::HashMap as StdMap;
    use std::io::{self, Write as _};
    use std::time::{Duration, Instant};

    use crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{
        Terminal,
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        symbols,
        text::{Line, Span},
        widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph},
    };

    // 1..10, then 2x: 20, 40, 80, ..., 100_000
    let mut gpr_values: Vec<usize> = (1..=10).collect();
    let mut v = 10.0f64;
    loop {
        v = (v * 2.0).ceil();
        let n = v as usize;
        if n > 100_000 {
            break;
        }
        if n != *gpr_values.last().unwrap() {
            gpr_values.push(n);
        }
    }
    if *gpr_values.last().unwrap() != 100_000 {
        gpr_values.push(100_000);
    }

    let target_sizes: Vec<usize> = vec![10, 100, 1_000, 10_000, 100_000, 1_000_000];
    let max_size = *target_sizes.last().unwrap();

    let keys: Vec<BenchKey> = random_items(0xFEED, max_size);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, max_size);

    let mut present_set: HashSet<BenchKey, BenchHasherBuilder> =
        HashSet::with_capacity_and_hasher(keys.len(), BenchHasherBuilder::default());
    for key in &keys {
        present_set.insert(key.clone());
    }
    let mut miss_rng = StdRng::seed_from_u64(0xBA5E);
    let mut miss_keys: Vec<BenchKey> = Vec::with_capacity(max_size);
    while miss_keys.len() < max_size {
        let candidate = random_bench_item(&mut miss_rng);
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }
    drop(present_set);

    let mut hot_rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET)
        .map(|_| random_bench_item(&mut hot_rng))
        .collect();
    let mut hot_map_keys: Vec<BenchKey> = hot_keys.clone();
    hot_map_keys.extend(random_items(0xD00D, max_size - HOT_SET));
    let hot_map_values: Vec<BenchValue> = random_items(0xBADD, max_size);

    const ROUNDS: u64 = 100;

    let mut file = std::fs::File::create("get_graph.csv").expect("failed to create get_graph.csv");
    writeln!(file, "gets_per_round,bench,impl,map_size,per_get_ns").unwrap();

    // TUI setup — install panic hook to restore terminal on crash.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen);
        original_hook(info);
    }));
    enable_raw_mode().unwrap();
    execute!(io::stderr(), EnterAlternateScreen).unwrap();
    let backend = CrosstermBackend::new(io::stderr());
    let mut terminal = Terminal::new(backend).unwrap();

    // Chart data: (bench, impl) -> Vec<(log10_gpr, avg_per_get_ns)>
    let mut chart_data: StdMap<(&str, &str), Vec<(f64, f64)>> = StdMap::new();
    let _total_gpr = gpr_values.len();

    const BENCH_NAMES: [&str; 3] = ["get_hits", "get_misses", "get_hotset"];
    const IMPL_COLORS: [(&str, Color); 3] = [
        ("pomap", Color::Green),
        ("std_hashmap", Color::Yellow),
        ("hashbrown", Color::Red),
    ];

    fn draw_tui(
        terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
        chart_data: &StdMap<(&str, &str), Vec<(f64, f64)>>,
        status: &str,
    ) {
        let _ =
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Ratio(1, 4),
                        Constraint::Ratio(1, 4),
                        Constraint::Ratio(1, 4),
                        Constraint::Min(2),
                    ])
                    .split(f.area());

                let x_labels: Vec<Span> = ["1", "10", "100", "1K", "10K", "100K"]
                    .iter()
                    .map(|s| Span::raw(*s))
                    .collect();

                for (i, bench) in BENCH_NAMES.iter().enumerate() {
                    // At each x, compute average of all impls, then show each as ratio to that average
                    let all_impl_data: Vec<(&str, &Vec<(f64, f64)>)> = IMPL_COLORS
                        .iter()
                        .filter_map(|(name, _)| {
                            chart_data
                                .get(&(*bench as &str, *name as &str))
                                .map(|d| (*name, d))
                        })
                        .collect();

                    let avg_at = |x: f64| -> f64 {
                        let (sum, count) = all_impl_data
                            .iter()
                            .filter_map(|(_, d)| d.iter().find(|(dx, _)| *dx == x).map(|(_, y)| *y))
                            .fold((0.0, 0usize), |(s, c), y| (s + y, c + 1));
                        if count > 0 { sum / count as f64 } else { 1.0 }
                    };

                    let datasets: Vec<Dataset> = IMPL_COLORS
                        .iter()
                        .filter_map(|(impl_name, color)| {
                            let data = chart_data.get(&(*bench as &str, *impl_name as &str))?;
                            let ratio: Vec<(f64, f64)> =
                                data.iter().map(|&(x, y)| (x, y / avg_at(x))).collect();
                            let ratio = Vec::leak(ratio);
                            Some(
                                Dataset::default()
                                    .marker(symbols::Marker::Braille)
                                    .graph_type(ratatui::widgets::GraphType::Line)
                                    .style(Style::default().fg(*color))
                                    .data(ratio),
                            )
                        })
                        .collect();

                    let all_ratios = || {
                        IMPL_COLORS
                            .iter()
                            .filter_map(|(name, _)| {
                                chart_data.get(&(*bench as &str, *name as &str))
                            })
                            .flat_map(|v| v.iter().map(|&(x, y)| y / avg_at(x)))
                    };
                    let max_r = all_ratios().fold(1.0f64, f64::max);
                    let min_r = all_ratios().fold(f64::MAX, f64::min).min(1.0);
                    let margin = (max_r - min_r).max(0.01) * 0.15;
                    let y_lo = (min_r - margin).max(0.0);
                    let y_hi = max_r + margin;

                    let y_lo_s = format!("{:.2}x", y_lo);
                    let y_mid_s = format!("{:.2}x", (y_lo + y_hi) / 2.0);
                    let y_hi_s = format!("{:.2}x", y_hi);

                    let chart =
                        Chart::new(datasets)
                            .block(Block::default().title(*bench).borders(Borders::ALL))
                            .x_axis(
                                Axis::default()
                                    .title("gets_per_round")
                                    .bounds([0.0, 5.0])
                                    .labels(x_labels.clone()),
                            )
                            .y_axis(Axis::default().title("vs avg").bounds([y_lo, y_hi]).labels(
                                vec![Span::raw(y_lo_s), Span::raw(y_mid_s), Span::raw(y_hi_s)],
                            ));

                    f.render_widget(chart, chunks[i]);
                }

                let bottom = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(1), Constraint::Length(1)])
                    .split(chunks[3]);

                let legend = Line::from(vec![
                    Span::styled("■ pomap", Style::default().fg(Color::Green)),
                    Span::raw("  "),
                    Span::styled("■ std_hashmap", Style::default().fg(Color::Yellow)),
                    Span::raw("  "),
                    Span::styled("■ hashbrown", Style::default().fg(Color::Red)),
                ]);
                f.render_widget(Paragraph::new(legend), bottom[0]);
                f.render_widget(Paragraph::new(status), bottom[1]);
            });
    }

    // Timing macro matching criterion: benchmark each size independently, then average.
    macro_rules! time_gets {
        ($file:expr, $chart:expr, $bench:expr, $impl_name:expr, $maps:expr,
         $gpr:expr, $lookup_keys:expr, $seed_base:expr, $key_range:expr) => {{
            let mut sum_per_get = 0.0f64;
            for &(size, ref map) in $maps.iter() {
                let range = $key_range(size);
                let mut total_ns = 0u128;
                for round in 0..ROUNDS {
                    let seed = $seed_base ^ size as u64;
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(round));
                    let start = Instant::now();
                    for _ in 0..$gpr {
                        let idx = rng.random_range(0..range);
                        black_box(map.get(&$lookup_keys[idx]));
                    }
                    total_ns += start.elapsed().as_nanos();
                }
                sum_per_get += total_ns as f64 / ($gpr as f64 * ROUNDS as f64);
            }
            let per_get = sum_per_get / $maps.len() as f64;
            writeln!(
                $file,
                "{},{},{},all,{:.2}",
                $gpr, $bench, $impl_name, per_get
            )
            .unwrap();
            let x = ($gpr as f64).log10().max(0.0);
            $chart
                .entry(($bench, $impl_name))
                .or_insert_with(Vec::new)
                .push((x, per_get));
        }};
    }

    // Background thread for instant quit on 'q' or Ctrl+C.
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    let quit = Arc::new(AtomicBool::new(false));
    let quit_bg = quit.clone();
    let _input_thread = std::thread::spawn(move || {
        loop {
            if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.code == KeyCode::Char('q')
                        || (key.code == KeyCode::Char('c')
                            && key
                                .modifiers
                                .contains(crossterm::event::KeyModifiers::CONTROL))
                    {
                        quit_bg.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
            if quit_bg.load(Ordering::Relaxed) {
                break;
            }
        }
    });

    #[allow(unused_assignments)]
    let mut status = String::new();
    let bench_start = Instant::now();
    let total_steps = gpr_values.len() * 9; // 9 bench×impl combos per gpr
    let mut completed_steps = 0usize;

    macro_rules! step {
        ($gpr:expr, $bench:expr, $impl_name:expr, $($rest:tt)*) => {{
            let eta = if completed_steps > 0 {
                let elapsed = bench_start.elapsed().as_secs_f64();
                let per_step = elapsed / completed_steps as f64;
                let remaining = (total_steps - completed_steps) as f64 * per_step;
                if remaining >= 3600.0 {
                    format!("{:.1}h", remaining / 3600.0)
                } else if remaining >= 60.0 {
                    format!("{:.0}m", remaining / 60.0)
                } else {
                    format!("{:.0}s", remaining)
                }
            } else {
                "...".to_string()
            };
            status = format!(
                " gpr={:<6} | {} / {} | ETA {} | 'q' or Ctrl+C to stop",
                $gpr, $bench, $impl_name, eta
            );
            draw_tui(&mut terminal, &chart_data, &status);
            time_gets!(file, chart_data, $bench, $impl_name, $($rest)*);
            completed_steps += 1;
        }};
    }

    macro_rules! check_quit_labeled {
        ($label:lifetime) => {
            if quit.load(Ordering::Relaxed) {
                break $label;
            }
        };
    }

    // Each bench type: build all 3 impls' maps (like criterion's bench functions),
    // then for each GPR value, run impls in random order. Measure per-size independently.
    // Drop all maps between bench types to reduce memory pressure.

    let mut order_rng = StdRng::seed_from_u64(0x0DE12);

    // get_hits
    {
        let pm = build_pomap_maps_from_data(&target_sizes, &keys, &values);
        let sm = build_std_maps_from_data(&target_sizes, &keys, &values);
        let hb = build_hashbrown_maps_from_data(&target_sizes, &keys, &values);
        'hits: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('hits);
                match idx {
                    0 => step!(gpr, "get_hits", "pomap", pm, gpr, keys, 0xC01DBEEF, |s: usize| s),
                    1 => step!(gpr, "get_hits", "std_hashmap", sm, gpr, keys, 0xC01DBEEF, |s: usize| s),
                    _ => step!(gpr, "get_hits", "hashbrown", hb, gpr, keys, 0xC01DBEEF, |s: usize| s),
                }
            }
        }
    }

    // get_misses — maps built from keys/values, lookups use miss_keys
    {
        let pm = build_pomap_maps_from_data(&target_sizes, &keys, &values);
        let sm = build_std_maps_from_data(&target_sizes, &keys, &values);
        let hb = build_hashbrown_maps_from_data(&target_sizes, &keys, &values);
        'misses: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('misses);
                match idx {
                    0 => step!(gpr, "get_misses", "pomap", pm, gpr, miss_keys, 0xC0FFEE42, |s: usize| s),
                    1 => step!(gpr, "get_misses", "std_hashmap", sm, gpr, miss_keys, 0xC0FFEE42, |s: usize| s),
                    _ => step!(gpr, "get_misses", "hashbrown", hb, gpr, miss_keys, 0xC0FFEE42, |s: usize| s),
                }
            }
        }
    }

    // get_hotset
    {
        let pm = build_pomap_maps_from_data(&target_sizes, &hot_map_keys, &hot_map_values);
        let sm = build_std_maps_from_data(&target_sizes, &hot_map_keys, &hot_map_values);
        let hb = build_hashbrown_maps_from_data(&target_sizes, &hot_map_keys, &hot_map_values);
        'hotset: for (_, &gpr) in gpr_values.iter().enumerate() {
            let mut order = [0u8, 1, 2];
            order.shuffle(&mut order_rng);
            for &idx in &order {
                check_quit_labeled!('hotset);
                match idx {
                    0 => step!(gpr, "get_hotset", "pomap", pm, gpr, hot_keys, 0xDEC0DE42, |s: usize| HOT_SET.min(s).max(1)),
                    1 => step!(gpr, "get_hotset", "std_hashmap", sm, gpr, hot_keys, 0xDEC0DE42, |s: usize| HOT_SET.min(s).max(1)),
                    _ => step!(gpr, "get_hotset", "hashbrown", hb, gpr, hot_keys, 0xDEC0DE42, |s: usize| HOT_SET.min(s).max(1)),
                }
            }
        }
    }

    if !quit.load(Ordering::Relaxed) {
        draw_tui(&mut terminal, &chart_data, " Done! Press any key to exit.");
        while !quit.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
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
    bench_memory_footprint
);

#[cfg(not(feature = "get-graph"))]
criterion_main!(benches);

#[cfg(feature = "get-graph")]
fn main() {
    get_graph();
}
