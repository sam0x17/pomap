use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
    hint::black_box,
};

use ahash::AHasher;
use hashbrown::HashMap as HashbrownMap;
use pomap::pomap2::PoMap2;
use rand::{Rng, SeedableRng, rngs::StdRng};

type BenchHasher = AHasher;
type BenchHasherBuilder = BuildHasherDefault<BenchHasher>;

type BenchPoMap2 = PoMap2<u64, u64, BenchHasherBuilder>;
type BenchHashMap = HashMap<u64, u64, BenchHasherBuilder>;
type BenchHashbrownMap = HashbrownMap<u64, u64, BenchHasherBuilder>;

fn random_items(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

const MAX_SIZE: usize = 100_000;

fn main() {
    let keys: Vec<u64> = random_items(0xA11CE, MAX_SIZE);
    let values: Vec<u64> = random_items(0xFACE, MAX_SIZE);

    let sizes: Vec<usize> = {
        let mut s = Vec::new();
        let n = 50usize;
        for i in 0..n {
            let size = 10 + (MAX_SIZE - 10) * i / (n - 1);
            if s.last().copied() != Some(size) {
                s.push(size);
            }
        }
        s
    };

    println!("=== INSERT ALLOCATE ===");
    let rounds = 3;

    // PoMap2
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for &size in &sizes {
            let mut map = BenchPoMap2::with_hasher(BenchHasherBuilder::default());
            for idx in 0..size {
                black_box(map.insert(keys[idx], values[idx]));
            }
            black_box(&map);
        }
    }
    let pomap2_time = start.elapsed();

    // Hashbrown
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for &size in &sizes {
            let mut map = BenchHashbrownMap::with_hasher(BenchHasherBuilder::default());
            for idx in 0..size {
                black_box(map.insert(keys[idx], values[idx]));
            }
            black_box(&map);
        }
    }
    let hashbrown_time = start.elapsed();

    // Std
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for &size in &sizes {
            let mut map: BenchHashMap = HashMap::with_hasher(BenchHasherBuilder::default());
            for idx in 0..size {
                black_box(map.insert(keys[idx], values[idx]));
            }
            black_box(&map);
        }
    }
    let std_time = start.elapsed();

    println!("  pomap2:    {:>8.2}ms", pomap2_time.as_secs_f64() * 1000.0);
    println!("  hashbrown: {:>8.2}ms", hashbrown_time.as_secs_f64() * 1000.0);
    println!("  std:       {:>8.2}ms", std_time.as_secs_f64() * 1000.0);
    println!("  pomap2 vs hashbrown: {:.1}x", pomap2_time.as_secs_f64() / hashbrown_time.as_secs_f64());

    // INSERT PRE-ALLOCATED (no growing)
    println!("\n=== INSERT PRE-ALLOCATED ===");
    let prealloc_size = 50_000;
    let rounds = 5;

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        // Over-allocate to prevent any bucket overflow (birthday problem).
        let mut map = BenchPoMap2::with_capacity_and_hasher(prealloc_size * 5, BenchHasherBuilder::default());
        for idx in 0..prealloc_size {
            black_box(map.insert(keys[idx], values[idx]));
        }
        black_box(&map);
    }
    let pomap2_prealloc = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let mut map = BenchHashbrownMap::with_capacity_and_hasher(prealloc_size, BenchHasherBuilder::default());
        for idx in 0..prealloc_size {
            black_box(map.insert(keys[idx], values[idx]));
        }
        black_box(&map);
    }
    let hb_prealloc = start.elapsed();

    println!("  pomap2:    {:>8.2}ms", pomap2_prealloc.as_secs_f64() * 1000.0);
    println!("  hashbrown: {:>8.2}ms", hb_prealloc.as_secs_f64() * 1000.0);
    println!("  pomap2 vs hashbrown: {:.2}x", pomap2_prealloc.as_secs_f64() / hb_prealloc.as_secs_f64());

    // Build maps for get/update benchmarks
    let mut pomap2_maps: Vec<BenchPoMap2> = Vec::new();
    let mut hb_maps: Vec<BenchHashbrownMap> = Vec::new();
    let get_sizes: Vec<usize> = (0..50).map(|i| 10 + (MAX_SIZE - 10) * i / 49).collect();

    for &size in &get_sizes {
        let mut pm = BenchPoMap2::with_hasher(BenchHasherBuilder::default());
        let mut hb = BenchHashbrownMap::with_hasher(BenchHasherBuilder::default());
        for idx in 0..size {
            pm.insert(keys[idx], values[idx]);
            hb.insert(keys[idx], values[idx]);
        }
        pomap2_maps.push(pm);
        hb_maps.push(hb);
    }

    println!("\n=== GET HITS ===");
    let rounds = 10;
    let gets_per = 100;

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for (i, map) in pomap2_maps.iter().enumerate() {
            let size = get_sizes[i];
            let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
            for _ in 0..gets_per {
                let idx = rng.random_range(0..size);
                black_box(map.get(&keys[idx]));
            }
        }
    }
    let pomap2_get = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for (i, map) in hb_maps.iter().enumerate() {
            let size = get_sizes[i];
            let mut rng = StdRng::seed_from_u64(0xC01DBEEF ^ size as u64);
            for _ in 0..gets_per {
                let idx = rng.random_range(0..size);
                black_box(map.get(&keys[idx]));
            }
        }
    }
    let hb_get = start.elapsed();

    println!("  pomap2:    {:>8.2}ms", pomap2_get.as_secs_f64() * 1000.0);
    println!("  hashbrown: {:>8.2}ms", hb_get.as_secs_f64() * 1000.0);
    println!("  pomap2 vs hashbrown: {:.3}x", pomap2_get.as_secs_f64() / hb_get.as_secs_f64());

    println!("\n=== UPDATE (get_mut) ===");
    let mut pomap2_maps_mut = pomap2_maps;
    let mut hb_maps_mut = hb_maps;

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for (i, map) in pomap2_maps_mut.iter_mut().enumerate() {
            let size = get_sizes[i];
            for idx in 0..gets_per.min(size) {
                if let Some(v) = map.get_mut(&keys[idx]) {
                    *v = black_box(*v + 1);
                }
            }
        }
    }
    let pomap2_update = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..rounds {
        for (i, map) in hb_maps_mut.iter_mut().enumerate() {
            let size = get_sizes[i];
            for idx in 0..gets_per.min(size) {
                if let Some(v) = map.get_mut(&keys[idx]) {
                    *v = black_box(*v + 1);
                }
            }
        }
    }
    let hb_update = start.elapsed();

    println!("  pomap2:    {:>8.2}ms", pomap2_update.as_secs_f64() * 1000.0);
    println!("  hashbrown: {:>8.2}ms", hb_update.as_secs_f64() * 1000.0);
    println!("  pomap2 vs hashbrown: {:.3}x", pomap2_update.as_secs_f64() / hb_update.as_secs_f64());

    // Remove benchmark
    println!("\n=== REMOVE HITS ===");
    let remove_sizes: Vec<usize> = (0..50).map(|i| 10 + (100_000 - 10) * i / 49).collect();
    let remove_keys: Vec<u64> = random_items(0xD15EA5E, 100_000);
    let remove_values: Vec<u64> = random_items(0xBEEFC0DE, 100_000);
    let removes_per = 100;

    // Build + clone for pomap2
    let pomap2_remove_maps: Vec<BenchPoMap2> = remove_sizes.iter().map(|&size| {
        let mut m = BenchPoMap2::with_hasher(BenchHasherBuilder::default());
        for idx in 0..size { m.insert(remove_keys[idx], remove_values[idx]); }
        m
    }).collect();

    let hb_remove_maps: Vec<BenchHashbrownMap> = remove_sizes.iter().map(|&size| {
        let mut m = BenchHashbrownMap::with_hasher(BenchHasherBuilder::default());
        for idx in 0..size { m.insert(remove_keys[idx], remove_values[idx]); }
        m
    }).collect();

    let start = std::time::Instant::now();
    let mut maps = pomap2_remove_maps.iter().map(|m| m.clone()).collect::<Vec<_>>();
    for (i, map) in maps.iter_mut().enumerate() {
        let size = remove_sizes[i];
        let removes = removes_per.min(size);
        for idx in 0..removes {
            black_box(map.remove(&remove_keys[idx]));
        }
    }
    let pomap2_remove = start.elapsed();

    let start = std::time::Instant::now();
    let mut maps = hb_remove_maps.iter().map(|m| m.clone()).collect::<Vec<_>>();
    for (i, map) in maps.iter_mut().enumerate() {
        let size = remove_sizes[i];
        let removes = removes_per.min(size);
        for idx in 0..removes {
            black_box(map.remove(&remove_keys[idx]));
        }
    }
    let hb_remove = start.elapsed();

    println!("  pomap2:    {:>8.2}ms", pomap2_remove.as_secs_f64() * 1000.0);
    println!("  hashbrown: {:>8.2}ms", hb_remove.as_secs_f64() * 1000.0);
    println!("  pomap2 vs hashbrown: {:.3}x", pomap2_remove.as_secs_f64() / hb_remove.as_secs_f64());

    // Memory
    println!("\n=== MEMORY (slot count) ===");
    for &size in &[500usize, 5000, 50000, 500000] {
        let mut pm = BenchPoMap2::with_hasher(BenchHasherBuilder::default());
        let mut hb = BenchHashbrownMap::with_hasher(BenchHasherBuilder::default());
        for i in 0..size {
            pm.insert(keys[i % keys.len()], values[i % values.len()]);
            hb.insert(keys[i % keys.len()], values[i % values.len()]);
        }
        let pm_cap = pm.capacity();
        let hb_cap = hb.capacity();
        println!("  {:>8} entries: pomap2={:>8} hb={:>8} ratio={:.2}x load={:.1}%",
                 size, pm_cap, hb_cap,
                 pm_cap as f64 / hb_cap as f64,
                 size as f64 / pm_cap as f64 * 100.0);
    }
}
