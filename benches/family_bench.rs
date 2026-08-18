//! Cross-design "family" benchmark: every surviving PoMap design, one harness,
//! one methodology (the current one: post drop-fix, evenly-spaced sizes,
//! in-run normalization against the same hashbrown/std anchors).
//!
//! Variants:
//! - `pomap`     — current consolidated engine (AoS inline hash), GROWTH=4
//! - `pomap_g2`  — same engine, GROWTH=2 (insert_allocate + memory only; all
//!                 other groups provision via with_capacity and are growth-independent)
//! - `pomap3`    — pre-consolidation AoS scalar engine (75% load, no gap
//!                 preservation, no with_capacity zero-grow contract)
//! - `oldsimd`   — the deleted simd-buckets engine: SoA u8 tags + (K,V) entries,
//!                 SIMD scan, order on hash-prefix bits only
//! - `main_soa`  — branch `main`: SoA u64 hashes + entries, bounded scan windows,
//!                 cascade displacement, grow-on-window-overflow
//! - `tags_soa`  — branch `simd-tags`: main_soa + displacement-nibble/fingerprint
//!                 tag byte with u8x16 SIMD scan
//!
//! Workload constants and seeds are identical to `pomap_bench.rs`.

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
use rand::{Rng, SeedableRng, rngs::StdRng};

extern crate alloc;

#[path = "support/family/v_pomap3.rs"]
#[allow(dead_code)]
mod v_pomap3;
#[path = "support/family/v_oldsimd.rs"]
#[allow(dead_code)]
mod v_oldsimd;
#[path = "support/family/v_main_soa.rs"]
#[allow(dead_code)]
mod v_main_soa;
#[path = "support/family/v_tags_soa.rs"]
#[allow(dead_code)]
mod v_tags_soa;
#[path = "support/family/v_tags2_soa.rs"]
#[allow(dead_code)]
mod v_tags2_soa;

type BenchKey = u64;
type BenchValue = u64;

/// Hasher configuration shared by every implementation.
type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;

type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 4>;
type BenchPoMapG2 = PoMap<BenchKey, BenchValue, BenchHasherBuilder, 2>;
type BenchPoMap3 = v_pomap3::PoMap3<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchOldSimd = v_oldsimd::PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchMainSoa = v_main_soa::PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchTagsSoa = v_tags_soa::PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
type BenchTags2Soa = v_tags2_soa::PoMap<BenchKey, BenchValue, BenchHasherBuilder>;
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

// ---------------------------------------------------------------------------
// Variant machinery: every pomap-family engine exposes the same inherent API
// (with_hasher / with_capacity_and_hasher / insert / get / get_mut / remove /
// shrink_to / Clone), so macros stamp out identical per-variant blocks.
// std/hashbrown blocks are written out verbatim, unchanged from pomap_bench.rs.
// ---------------------------------------------------------------------------

/// Build one prebuilt map per target size (provisioned via with_capacity).
macro_rules! build_maps {
    ($ty:ty, $target_sizes:expr, $keys:expr, $values:expr) => {{
        $target_sizes
            .iter()
            .map(|&size| {
                let mut map: $ty =
                    <$ty>::with_capacity_and_hasher(size, BenchHasherBuilder::default());
                for idx in 0..size {
                    map.insert($keys[idx].clone(), $values[idx].clone());
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
                    map.insert($keys[idx].clone(), $values[idx].clone());
                }
                (size, map)
            })
            .collect::<Vec<(usize, $ty)>>()
    }};
}

/// Correctness gate: every variant must agree with hashbrown before we bother
/// timing it. Catches vendored-engine breakage (API drift, layout bugs) fast.
fn sanity_check() {
    const N: usize = 50_000;
    let keys = random_items(0x5A517, N);
    let values = random_items(0x5A11E, N);
    let miss_keys = random_items(0xDEAD5A51, 1000);

    let mut hb = hashbrown_with_capacity(N);
    for i in 0..N {
        hb.insert(keys[i], values[i]);
    }

    macro_rules! check {
        ($ty:ty, $label:expr) => {{
            let mut m: $ty = <$ty>::with_capacity_and_hasher(N, BenchHasherBuilder::default());
            for i in 0..N {
                m.insert(keys[i], values[i]);
            }
            for i in (0..N).step_by(7) {
                assert_eq!(m.get(&keys[i]), hb.get(&keys[i]), "{}: get mismatch", $label);
            }
            for k in &miss_keys {
                if !hb.contains_key(k) {
                    assert!(m.get(k).is_none(), "{}: phantom hit", $label);
                }
            }
            for i in (0..N).step_by(97) {
                assert_eq!(m.remove(&keys[i]), hb.get(&keys[i]).copied(), "{}: remove mismatch", $label);
            }
            for i in (0..N).step_by(97) {
                assert!(m.get(&keys[i]).is_none(), "{}: still present after remove", $label);
            }
            eprintln!("sanity ok: {}", $label);
        }};
    }

    check!(BenchPoMap, "pomap");
    check!(BenchPoMapG2, "pomap_g2");
    check!(BenchPoMap3, "pomap3");
    check!(BenchOldSimd, "oldsimd");
    check!(BenchMainSoa, "main_soa");
    check!(BenchTagsSoa, "tags_soa");
    check!(BenchTags2Soa, "tags2_soa");
}

fn bench_sanity(_c: &mut Criterion) {
    sanity_check();
}

// ---------------------------------------------------------------------------
// insert_allocate — build from empty, growth exercised
// ---------------------------------------------------------------------------

fn bench_insert_allocate(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xA11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_target_size);
    let mut group = c.comparison_benchmark_group("insert_allocate");

    macro_rules! insert_allocate_variant {
        ($ty:ty, $label:expr) => {
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &size in &target_sizes {
                        let mut map: $ty = <$ty>::with_hasher(BenchHasherBuilder::default());
                        for (key, val) in keys.iter().zip(values.iter()).take(size) {
                            black_box(map.insert(key.clone(), val.clone()));
                        }
                        black_box(&map);
                    }
                });
            });
        };
    }

    insert_allocate_variant!(BenchPoMap, "pomap");
    insert_allocate_variant!(BenchPoMapG2, "pomap_g2");
    insert_allocate_variant!(BenchPoMap3, "pomap3");
    insert_allocate_variant!(BenchOldSimd, "oldsimd");
    insert_allocate_variant!(BenchMainSoa, "main_soa");
    insert_allocate_variant!(BenchTagsSoa, "tags_soa");
    insert_allocate_variant!(BenchTags2Soa, "tags2_soa");

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

// ---------------------------------------------------------------------------
// insert_preallocated — build into a pre-sized map (each impl's own strategy)
// ---------------------------------------------------------------------------

fn bench_insert_preallocated(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xA11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xFACE, max_target_size);
    let combined = keys
        .into_iter()
        .zip(values.into_iter())
        .collect::<Vec<(BenchKey, BenchValue)>>();
    let mut group = c.comparison_benchmark_group("insert_preallocated");

    macro_rules! insert_prealloc_variant {
        ($ty:ty, $label:expr) => {
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &size in &target_sizes {
                        let mut map: $ty =
                            <$ty>::with_capacity_and_hasher(size, BenchHasherBuilder::default());
                        for (key, val) in combined.iter().take(size) {
                            black_box(map.insert(key.clone(), val.clone()));
                        }
                        black_box(&map);
                    }
                });
            });
        };
    }

    insert_prealloc_variant!(BenchPoMap, "pomap");
    insert_prealloc_variant!(BenchPoMap3, "pomap3");
    insert_prealloc_variant!(BenchOldSimd, "oldsimd");
    insert_prealloc_variant!(BenchMainSoa, "main_soa");
    insert_prealloc_variant!(BenchTagsSoa, "tags_soa");
    insert_prealloc_variant!(BenchTags2Soa, "tags2_soa");

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

// ---------------------------------------------------------------------------
// get_hits
// ---------------------------------------------------------------------------

fn bench_get_hits(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys: Vec<BenchKey> = random_items(0xFEED, MAX_GET_INPUT_SIZE);
    let values: Vec<BenchValue> = random_items(0x1CEBEEF, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("get_hits");

    macro_rules! get_hits_variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &keys, &values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &(size, ref map) in &maps {
                        let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
                        for _ in 0..GETS_PER_ROUND {
                            let idx = rng.random_range(0..size);
                            let key = &keys[idx];
                            black_box(map.get(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    get_hits_variant!(BenchPoMap, "pomap");
    get_hits_variant!(BenchPoMap3, "pomap3");
    get_hits_variant!(BenchOldSimd, "oldsimd");
    get_hits_variant!(BenchMainSoa, "main_soa");
    get_hits_variant!(BenchTagsSoa, "tags_soa");
    get_hits_variant!(BenchTags2Soa, "tags2_soa");

    let std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

    let hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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
    drop(hashbrown_maps);

    group.finish();
}

// ---------------------------------------------------------------------------
// get_misses
// ---------------------------------------------------------------------------

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
        let candidate: BenchKey = miss_rng.random();
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }

    let mut group = c.comparison_benchmark_group("get_misses");

    macro_rules! get_misses_variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &present_keys, &present_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for &(size, ref map) in &maps {
                        let mut rng = StdRng::seed_from_u64(0xC0FFEE42 ^ size as u64);
                        for _ in 0..GETS_PER_ROUND {
                            let idx = rng.random_range(0..size);
                            let key = &miss_keys[idx];
                            black_box(map.get(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    get_misses_variant!(BenchPoMap, "pomap");
    get_misses_variant!(BenchPoMap3, "pomap3");
    get_misses_variant!(BenchOldSimd, "oldsimd");
    get_misses_variant!(BenchMainSoa, "main_soa");
    get_misses_variant!(BenchTagsSoa, "tags_soa");
    get_misses_variant!(BenchTags2Soa, "tags2_soa");

    let std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(present_keys[idx].clone(), present_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

    let hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(present_keys[idx].clone(), present_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

// ---------------------------------------------------------------------------
// update_existing
// ---------------------------------------------------------------------------

fn bench_update(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let keys: Vec<BenchKey> = random_items(0xC0FFEE, MAX_GET_INPUT_SIZE);
    let initial_values: Vec<BenchValue> = random_items(0xABC, MAX_GET_INPUT_SIZE);
    let update_values: Vec<BenchValue> = random_items(0xDEF, MAX_GET_INPUT_SIZE);
    let mut group = c.comparison_benchmark_group("update_existing");

    macro_rules! update_variant {
        ($ty:ty, $label:expr) => {{
            let mut maps = build_maps!($ty, &target_sizes, &keys, &initial_values);
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

    update_variant!(BenchPoMap, "pomap");
    update_variant!(BenchPoMap3, "pomap3");
    update_variant!(BenchOldSimd, "oldsimd");
    update_variant!(BenchMainSoa, "main_soa");
    update_variant!(BenchTagsSoa, "tags_soa");
    update_variant!(BenchTags2Soa, "tags2_soa");

    let mut std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), initial_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

    let mut hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), initial_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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
    drop(hashbrown_maps);

    group.finish();
}

// ---------------------------------------------------------------------------
// get_hotset
// ---------------------------------------------------------------------------

fn bench_hot_gets(c: &mut Criterion) {
    let target_sizes = get_target_sizes();
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    let hot_keys: Vec<BenchKey> = (0..HOT_SET).map(|_| rng.random()).collect();
    let mut map_keys: Vec<BenchKey> = hot_keys.clone();
    map_keys.extend(random_items(0xD00D, MAX_GET_INPUT_SIZE - HOT_SET));
    let map_values: Vec<BenchValue> = random_items(0xBADD, MAX_GET_INPUT_SIZE);
    let hot_counts: Vec<usize> = target_sizes
        .iter()
        .map(|&size| HOT_SET.min(size).max(1))
        .collect();

    let mut group = c.comparison_benchmark_group("get_hotset");

    macro_rules! hot_gets_variant {
        ($ty:ty, $label:expr) => {{
            let maps = build_maps!($ty, &target_sizes, &map_keys, &map_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for ((_, map), &hot_count) in maps.iter().zip(&hot_counts) {
                        let mut rng = StdRng::seed_from_u64(0xDEC0DE42 ^ hot_count as u64);
                        for _ in 0..GETS_PER_ROUND {
                            let idx = rng.random_range(0..hot_count);
                            let key = &hot_keys[idx];
                            black_box(map.get(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    hot_gets_variant!(BenchPoMap, "pomap");
    hot_gets_variant!(BenchPoMap3, "pomap3");
    hot_gets_variant!(BenchOldSimd, "oldsimd");
    hot_gets_variant!(BenchMainSoa, "main_soa");
    hot_gets_variant!(BenchTagsSoa, "tags_soa");
    hot_gets_variant!(BenchTags2Soa, "tags2_soa");

    let std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(map_keys[idx].clone(), map_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

    let hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(map_keys[idx].clone(), map_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

// ---------------------------------------------------------------------------
// remove_hits — iter_batched, batch returned so its drop is not timed
// ---------------------------------------------------------------------------

fn bench_remove_hits(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xD15EA5E, max_target_size);
    let values: Vec<BenchValue> = random_items(0xBEEFC0DE, max_target_size);
    let mut group = c.comparison_benchmark_group("remove_hits");

    macro_rules! remove_hits_variant {
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
                                let key = &keys[idx];
                                black_box(map.remove(key));
                            }
                        }
                        batch // return the batch so its drop is not timed
                    },
                    BatchSize::LargeInput,
                );
            });
            drop(maps);
        }};
    }

    remove_hits_variant!(BenchPoMap, "pomap");
    remove_hits_variant!(BenchPoMap3, "pomap3");
    remove_hits_variant!(BenchOldSimd, "oldsimd");
    remove_hits_variant!(BenchMainSoa, "main_soa");
    remove_hits_variant!(BenchTagsSoa, "tags_soa");
    remove_hits_variant!(BenchTags2Soa, "tags2_soa");

    let std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
    group.bench_function("std_hashmap", |b| {
        b.iter_batched(
            || {
                std_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashMap)>>()
            },
            |mut batch| {
                for (size, map) in batch.iter_mut() {
                    let removes = GETS_PER_ROUND.min(*size);
                    for idx in 0..removes {
                        let key = &keys[idx];
                        black_box(map.remove(key));
                    }
                }
                batch
            },
            BatchSize::LargeInput,
        );
    });
    drop(std_maps);

    let hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || {
                hashbrown_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashbrownMap)>>()
            },
            |mut batch| {
                for (size, map) in batch.iter_mut() {
                    let removes = GETS_PER_ROUND.min(*size);
                    for idx in 0..removes {
                        let key = &keys[idx];
                        black_box(map.remove(key));
                    }
                }
                batch
            },
            BatchSize::LargeInput,
        );
    });
    drop(hashbrown_maps);

    group.finish();
}

// ---------------------------------------------------------------------------
// remove_misses
// ---------------------------------------------------------------------------

fn bench_remove_misses(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
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
        let candidate: BenchKey = miss_rng.random();
        if !present_set.contains(&candidate) {
            miss_keys.push(candidate);
        }
    }

    let mut group = c.comparison_benchmark_group("remove_misses");

    macro_rules! remove_misses_variant {
        ($ty:ty, $label:expr) => {{
            let mut maps = build_maps!($ty, &target_sizes, &present_keys, &present_values);
            group.bench_function($label, |b| {
                b.iter(|| {
                    for (size, map) in maps.iter_mut() {
                        let removes = GETS_PER_ROUND.min(*size);
                        for idx in 0..removes {
                            let key = &miss_keys[idx];
                            black_box(map.remove(key));
                        }
                    }
                });
            });
            drop(maps);
        }};
    }

    remove_misses_variant!(BenchPoMap, "pomap");
    remove_misses_variant!(BenchPoMap3, "pomap3");
    remove_misses_variant!(BenchOldSimd, "oldsimd");
    remove_misses_variant!(BenchMainSoa, "main_soa");
    remove_misses_variant!(BenchTagsSoa, "tags_soa");
    remove_misses_variant!(BenchTags2Soa, "tags2_soa");

    let mut std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashMap = std_hashmap_with_capacity(size);
            for idx in 0..size {
                map.insert(present_keys[idx].clone(), present_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

    let mut hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(size);
            for idx in 0..size {
                map.insert(present_keys[idx].clone(), present_values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
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

// ---------------------------------------------------------------------------
// shrink_to — 8x over-provisioned maps shrunk to fit
// ---------------------------------------------------------------------------

fn bench_shrink_to(c: &mut Criterion) {
    let target_sizes = insert_target_sizes();
    let max_target_size = *target_sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0x5A11CE, max_target_size);
    let values: Vec<BenchValue> = random_items(0xC0FFEE55, max_target_size);
    let mut group = c.comparison_benchmark_group("shrink_to");

    macro_rules! shrink_variant {
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
                        batch // return the batch so its drop is not timed
                    },
                    BatchSize::LargeInput,
                );
            });
            drop(maps);
        }};
    }

    shrink_variant!(BenchPoMap, "pomap");
    shrink_variant!(BenchPoMap3, "pomap3");
    shrink_variant!(BenchOldSimd, "oldsimd");
    shrink_variant!(BenchMainSoa, "main_soa");
    shrink_variant!(BenchTagsSoa, "tags_soa");
    shrink_variant!(BenchTags2Soa, "tags2_soa");

    let std_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let capacity = size.saturating_mul(8).max(size);
            let mut map: BenchHashMap = std_hashmap_with_capacity(capacity);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
    group.bench_function("std_hashmap", |b| {
        b.iter_batched(
            || {
                std_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashMap)>>()
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
    drop(std_maps);

    let hashbrown_maps = {
        let mut v = Vec::new();
        for &size in &target_sizes {
            let capacity = size.saturating_mul(8).max(size);
            let mut map: BenchHashbrownMap = hashbrown_with_capacity(capacity);
            for idx in 0..size {
                map.insert(keys[idx].clone(), values[idx].clone());
            }
            v.push((size, map));
        }
        v
    };
    group.bench_function("hashbrown", |b| {
        b.iter_batched(
            || {
                hashbrown_maps
                    .iter()
                    .map(|(size, map)| (*size, map.clone()))
                    .collect::<Vec<(usize, BenchHashbrownMap)>>()
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
    drop(hashbrown_maps);

    group.finish();
}

// ---------------------------------------------------------------------------
// memory footprint — real retained heap bytes per variant (tracking allocator)
// ---------------------------------------------------------------------------

fn measure_map_bytes<F: FnOnce() -> R, R>(build: F) -> (usize, R) {
    let before = NET_ALLOCATED.load(Ordering::Relaxed);
    let map = build();
    let bytes = NET_ALLOCATED.load(Ordering::Relaxed).saturating_sub(before);
    (bytes, map)
}

fn bench_memory_footprint(c: &mut Criterion) {
    let sizes = [500usize, 5_000, 50_000, 500_000, 5_000_000];
    let max_size = *sizes.iter().max().unwrap();
    let keys: Vec<BenchKey> = random_items(0xB17E5, max_size);
    let values: Vec<BenchValue> = random_items(0xF007, max_size);

    println!(
        "\n{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}  (bytes/entry)",
        "entries", "pomap_g4", "pomap_g2", "pomap3", "oldsimd", "main_soa", "tags_soa", "tags2_soa", "hashbrown"
    );
    println!("{}", "-".repeat(96));

    for &size in &sizes {
        macro_rules! footprint {
            ($ty:ty) => {{
                let (bytes, m) = measure_map_bytes(|| {
                    let mut m: $ty = <$ty>::with_hasher(BenchHasherBuilder::default());
                    for i in 0..size {
                        m.insert(keys[i].clone(), values[i].clone());
                    }
                    m
                });
                drop(black_box(m));
                bytes as f64 / size as f64
            }};
        }
        let g4 = footprint!(BenchPoMap);
        let g2 = footprint!(BenchPoMapG2);
        let p3 = footprint!(BenchPoMap3);
        let os = footprint!(BenchOldSimd);
        let ms = footprint!(BenchMainSoa);
        let ts = footprint!(BenchTagsSoa);
        let t2 = footprint!(BenchTags2Soa);
        let hb = footprint!(BenchHashbrownMap);

        println!(
            "{:<10} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            size, g4, g2, p3, os, ms, ts, t2, hb
        );
    }

    // No-op benchmark so the group appears in criterion output.
    let mut group = c.benchmark_group("memory_footprint_bytes");
    group.bench_function("report", |b| b.iter(|| {}));
    group.finish();
}

criterion_group!(
    benches,
    bench_sanity,
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
criterion_main!(benches);
