//! Minimal cold-read driver for `perf stat` (Linux). Does the *bare* work — build
//! one map, then `passes` random-order probe passes over a working set ≫ LLC so
//! every pass stays cold — with no eviction buffer (which would pollute the
//! counters). `perf stat` wraps the whole process; build is one pass-equivalent of
//! traffic, negligible vs `passes`≈30, so per-op counts ≈ counters / (passes·N).
//!
//! Prints one line to stdout: `N=<n> ops=<passes*n>` (the normalizer). All the
//! interesting numbers come from `perf stat` on stderr — drive it via
//! `scripts/perf_cold.sh`.
//!
//! Usage: cold_perf <pomap|hashbrown|std> <get_hit|get_miss> <value_words> <ws_mb> <passes>

use std::collections::HashMap as StdMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;

use ahash::AHasher;
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type HB = BuildHasherDefault<AHasher>;

fn fisher_yates(n: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
    v
}

fn run<const W: usize>(imp: &str, op: &str, ws_mb: usize, passes: usize) {
    type Val<const W: usize> = [u64; W];
    let pm_slot = 16 + 8 * W;
    let n = (ws_mb * (1 << 20) / pm_slot).max(1024);

    let mut rng = StdRng::seed_from_u64(0xC01DB00C);
    let keys: Vec<u64> = (0..n).map(|_| rng.random()).collect();
    let miss: Vec<u64> = (0..n).map(|_| rng.random()).collect();
    let vals: Vec<Val<W>> = (0..n).map(|_| std::array::from_fn(|_| rng.random())).collect();
    let order = fisher_yates(n, &mut rng);
    let probe = if op == "get_miss" { &miss } else { &keys };

    // Normalizer for the script (stdout; perf output is on stderr).
    println!("N={n} ops={}", passes * n);

    macro_rules! build_probe {
        ($MapTy:ty) => {{
            let mut m: $MapTy = <$MapTy>::with_capacity_and_hasher(n, HB::default());
            for i in 0..n {
                m.insert(keys[i], vals[i]);
            }
            for _ in 0..passes {
                for &i in &order {
                    black_box(m.get(&probe[i]));
                }
            }
            black_box(&m);
        }};
    }
    match imp {
        "pomap" => build_probe!(PoMap<u64, Val<W>, HB>),
        "hashbrown" => build_probe!(HashbrownMap<u64, Val<W>, HB>),
        "std" => build_probe!(StdMap<u64, Val<W>, HB>),
        _ => {
            eprintln!("unknown impl: {imp}");
            std::process::exit(2);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "usage: cold_perf <pomap|hashbrown|std> <get_hit|get_miss> <value_words(1|2|4|8)> <ws_mb> <passes>"
        );
        std::process::exit(2);
    }
    let (imp, op) = (a[1].as_str(), a[2].as_str());
    let vw: usize = a[3].parse().expect("value_words");
    let ws_mb: usize = a[4].parse().expect("ws_mb");
    let passes: usize = a[5].parse().expect("passes");
    match vw {
        1 => run::<1>(imp, op, ws_mb, passes),
        2 => run::<2>(imp, op, ws_mb, passes),
        4 => run::<4>(imp, op, ws_mb, passes),
        8 => run::<8>(imp, op, ws_mb, passes),
        _ => {
            eprintln!("value_words must be 1|2|4|8");
            std::process::exit(2);
        }
    }
}
