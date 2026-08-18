//! Cold-access experiment matrix — PoMap vs hashbrown vs std::HashMap.
//!
//! Sweeps three axes and emits machine-collatable CSV:
//!   * value size  V ∈ {8, 16, 32, 64} bytes   (key is always u64)
//!   * working-set bytes  (log-spaced, L1 → well past LLC) — the warm→cold axis
//!   * operation  ∈ {get_hit, get_miss, insert, remove}
//!
//! Coldness is forced per measurement: build, evict caches with a stream ≫ LLC,
//! then time a single **random-order** pass (each key touched once, no reuse).
//! Sizes are chosen by target working-set BYTES (not entry count) so the crossover
//! aligns across value sizes and across machines with different cache sizes.
//!
//! CAVEAT (printed): single-map `get_miss` is biased toward hashbrown — its 1-byte
//! control array warms during the pass. The *fair* cold-miss number is the main
//! suite's multi-map-sweep `get_misses` (which keeps control arrays cold). Treat
//! the `get_miss` rows here as a PoMap-pessimistic bound.
//!
//! Output: CSV to stdout — `value_bytes,op,n,ws_mb,pomap_ns,hashbrown_ns,std_ns`.
//! Run via `scripts/run_cold.sh` (captures platform metadata, pins a core).

// Benchmark harness: style lints are silenced wholesale -- timed workload
// code must stay byte-comparable with the recorded methodology.
#![allow(clippy::all, unused_imports, dead_code)]

use std::collections::HashMap as StdMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
use std::time::Instant;

use ahash::AHasher;
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type HB = BuildHasherDefault<AHasher>;

/// Target working-set sizes in BYTES (PoMap footprint ≈ this), log-spaced L1→DRAM.
const WS_BYTES: [usize; 7] = [
    64 << 10,   // 64 KiB  (L1/L2)
    256 << 10,  // 256 KiB (L2)
    1 << 20,    // 1 MiB   (L2/L3)
    4 << 20,    // 4 MiB   (L3)
    16 << 20,   // 16 MiB  (L3/edge)
    48 << 20,   // 48 MiB  (≳ LLC on most parts)
    128 << 20,  // 128 MiB (DRAM)
];
const EVICT_BYTES: usize = 256 << 20;
const READ_TRIALS: usize = 3;
const WRITE_TRIALS: usize = 2;

fn fisher_yates(n: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
    v
}

fn evict(scratch: &mut [u64]) {
    let mut s = 0u64;
    for v in scratch.iter_mut() {
        *v = v.wrapping_add(1);
        s = s.wrapping_add(*v);
    }
    black_box(s);
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn cold(n: usize, trials: usize, scratch: &mut [u64], mut pass: impl FnMut()) -> f64 {
    let mut s = Vec::with_capacity(trials);
    for _ in 0..trials {
        evict(scratch);
        let t = Instant::now();
        pass();
        s.push(t.elapsed().as_secs_f64() / n as f64 * 1e9);
    }
    median(s)
}

/// Run the full size×op sweep for one value type `[u64; W]` (value = 8·W bytes).
fn run<const W: usize>(rng: &mut StdRng, scratch: &mut [u64]) {
    type Val<const W: usize> = [u64; W];
    let value_bytes = 8 * W;
    // PoMap slot ≈ 8 (hash) + 8 (key) + 8·W (value), 8-aligned.
    let pm_slot = 16 + 8 * W;
    let max_n = WS_BYTES[WS_BYTES.len() - 1] / pm_slot + 1;

    let keys: Vec<u64> = (0..max_n).map(|_| rng.random()).collect();
    let miss: Vec<u64> = (0..max_n).map(|_| rng.random()).collect();
    let vals: Vec<Val<W>> = (0..max_n)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

    let build_pm = |n: usize| {
        let mut m: PoMap<u64, Val<W>, HB> = PoMap::with_capacity_and_hasher(n, HB::default());
        for i in 0..n {
            m.insert(keys[i], vals[i]);
        }
        m
    };
    let build_hb = |n: usize| {
        let mut m: HashbrownMap<u64, Val<W>, HB> =
            HashbrownMap::with_capacity_and_hasher(n, HB::default());
        for i in 0..n {
            m.insert(keys[i], vals[i]);
        }
        m
    };
    let build_sm = |n: usize| {
        let mut m: StdMap<u64, Val<W>, HB> = StdMap::with_capacity_and_hasher(n, HB::default());
        for i in 0..n {
            m.insert(keys[i], vals[i]);
        }
        m
    };

    for &ws in &WS_BYTES {
        let n = (ws / pm_slot).max(64);
        let ws_mb = (n * pm_slot) as f64 / (1 << 20) as f64;
        let order = fisher_yates(n, rng);

        // get_hit (single-map; entry array stays cold → unbiased)
        let (pm, hb, sm) = (build_pm(n), build_hb(n), build_sm(n));
        let p = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(pm.get(&keys[i])); });
        let h = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(hb.get(&keys[i])); });
        let s = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(sm.get(&keys[i])); });
        println!("{value_bytes},get_hit,{n},{ws_mb:.1},{p:.2},{h:.2},{s:.2}");

        // get_miss (single-map; PoMap-pessimistic — see header caveat)
        let pmm = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(pm.get(&miss[i])); });
        let hbm = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(hb.get(&miss[i])); });
        let smm = cold(n, READ_TRIALS, scratch, || for &i in &order { black_box(sm.get(&miss[i])); });
        println!("{value_bytes},get_miss,{n},{ws_mb:.1},{pmm:.2},{hbm:.2},{smm:.2}");
        drop((pm, hb, sm));

        // insert (fresh pre-sized map per trial)
        let pi = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m: PoMap<u64, Val<W>, HB> = PoMap::with_capacity_and_hasher(n, HB::default());
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.insert(keys[i], vals[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        let hi = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m: HashbrownMap<u64, Val<W>, HB> = HashbrownMap::with_capacity_and_hasher(n, HB::default());
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.insert(keys[i], vals[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        let si = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m: StdMap<u64, Val<W>, HB> = StdMap::with_capacity_and_hasher(n, HB::default());
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.insert(keys[i], vals[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        println!("{value_bytes},insert,{n},{ws_mb:.1},{pi:.2},{hi:.2},{si:.2}");

        // remove (rebuild full map per trial)
        let pr = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m = build_pm(n);
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.remove(&keys[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        let hr = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m = build_hb(n);
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.remove(&keys[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        let sr = {
            let mut t = Vec::new();
            for _ in 0..WRITE_TRIALS {
                let mut m = build_sm(n);
                evict(scratch);
                let s = Instant::now();
                for &i in &order { black_box(m.remove(&keys[i])); }
                t.push(s.elapsed().as_secs_f64() / n as f64 * 1e9);
                black_box(&m);
            }
            median(t)
        };
        println!("{value_bytes},remove,{n},{ws_mb:.1},{pr:.2},{hr:.2},{sr:.2}");
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xC01DB00C);
    let mut scratch = vec![0u64; EVICT_BYTES / 8];
    println!("value_bytes,op,n,ws_mb,pomap_ns,hashbrown_ns,std_ns");
    run::<1>(&mut rng, &mut scratch); // 8 B value
    run::<2>(&mut rng, &mut scratch); // 16 B
    run::<4>(&mut rng, &mut scratch); // 32 B
    run::<8>(&mut rng, &mut scratch); // 64 B
}
