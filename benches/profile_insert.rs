use std::{hash::BuildHasherDefault, hint::black_box};

use ahash::AHasher;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type BenchKey = u64;
type BenchValue = u64;
type BenchHasherBuilder = BuildHasherDefault<AHasher>;
type BenchPoMap = PoMap<BenchKey, BenchValue, BenchHasherBuilder>;

const MAX_SIZE: usize = 100_000;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "allocate".to_string());
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let keys: Vec<u64> = (0..MAX_SIZE).map(|_| rng.random()).collect();
    let mut rng = StdRng::seed_from_u64(0xFACE);
    let values: Vec<u64> = (0..MAX_SIZE).map(|_| rng.random()).collect();

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

    // "warmup" mode: allocate but pre-touch pages by doing a throwaway insert cycle first.
    let rounds = if mode == "preallocated" { 10 } else { 3 };

    for _ in 0..rounds {
        for &size in &sizes {
            let mut map: BenchPoMap = if mode == "preallocated" {
                BenchPoMap::with_capacity_and_hasher(size, BenchHasherBuilder::default())
            } else {
                BenchPoMap::with_hasher(BenchHasherBuilder::default())
            };
            for idx in 0..size {
                black_box(map.insert(keys[idx], values[idx]));
            }
            black_box(&map);
        }
    }
}
