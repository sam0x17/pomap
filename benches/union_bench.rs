//! Set-algebra benchmark: PoMap's O(n+m) streaming merge union vs the
//! conventional clone-and-extend union on hashbrown (which re-probes and
//! re-hashes per key). Two overlap regimes at two sizes.

// Benchmark harness: style lints are silenced wholesale -- timed workload
// code must stay byte-comparable with the recorded methodology.
#![allow(clippy::all, unused_imports, dead_code)]

use std::hash::BuildHasherDefault;
use std::hint::black_box;

use ahash::AHasher;
use criterion::{Criterion, criterion_group, criterion_main};
use hashbrown::HashMap as HashbrownMap;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type H = BuildHasherDefault<AHasher>;
type Pm = PoMap<u64, u64, H, 4>;
type Hb = HashbrownMap<u64, u64, H>;

fn random_items(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.random()).collect()
}

fn bench_union(c: &mut Criterion) {
    for (n, overlap_tag, second_seed) in
        [(100_000usize, "disjoint", 0xB0B_u64), (100_000, "half", 0xA11CE), (1_000_000, "disjoint", 0xB0B), (1_000_000, "half", 0xA11CE)]
    {
        let keys_a = random_items(0xA11CE, n);
        let keys_b: Vec<u64> = if overlap_tag == "half" {
            // Half of b's keys collide with a's (same seed prefix), half fresh.
            // Same seed as keys_a => first n/2 keys genuinely overlap.
            let mut v = random_items(second_seed, n / 2);
            v.extend(random_items(0xBEEF, n - n / 2));
            v
        } else {
            random_items(second_seed, n)
        };

        let mut pa: Pm = Pm::with_capacity_and_hasher(n, H::default());
        let mut pb: Pm = Pm::with_capacity_and_hasher(n, H::default());
        let mut ha: Hb = Hb::with_capacity_and_hasher(n, H::default());
        let mut hb: Hb = Hb::with_capacity_and_hasher(n, H::default());
        for i in 0..n {
            pa.insert(keys_a[i], 1);
            ha.insert(keys_a[i], 1);
        }
        for &k in &keys_b {
            pb.insert(k, 2);
            hb.insert(k, 2);
        }

        let mut group = c.comparison_benchmark_group(format!("union_{n}_{overlap_tag}"));
        group.bench_function("pomap", |b| {
            b.iter(|| black_box(pa.union(&pb)));
        });
        group.bench_function("hashbrown", |b| {
            b.iter(|| {
                let mut u = ha.clone();
                u.extend(hb.iter().map(|(k, v)| (*k, *v)));
                black_box(u)
            });
        });
        group.finish();
    }
}

/// The no-rehash thesis test: with expensive-to-hash keys (128-byte Strings),
/// hashbrown's extend pays hash(128B) per key; PoMap's merge never hashes.
fn bench_union_string(c: &mut Criterion) {
    type PmS = PoMap<String, u64, H, 4>;
    type HbS = HashbrownMap<String, u64, H>;
    let n = 100_000usize;
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let mk_key = |rng: &mut StdRng| -> String {
        (0..128).map(|_| (b'a' + (rng.random::<u8>() % 26)) as char).collect()
    };
    let keys_a: Vec<String> = (0..n).map(|_| mk_key(&mut rng)).collect();
    let mut rng2 = StdRng::seed_from_u64(0xBEEF);
    let keys_b: Vec<String> = (0..n).map(|_| mk_key(&mut rng2)).collect();

    let mut pa: PmS = PmS::with_capacity_and_hasher(n, H::default());
    let mut pb: PmS = PmS::with_capacity_and_hasher(n, H::default());
    let mut ha: HbS = HbS::with_capacity_and_hasher(n, H::default());
    let mut hb: HbS = HbS::with_capacity_and_hasher(n, H::default());
    for i in 0..n {
        pa.insert(keys_a[i].clone(), 1);
        ha.insert(keys_a[i].clone(), 1);
        pb.insert(keys_b[i].clone(), 2);
        hb.insert(keys_b[i].clone(), 2);
    }

    let mut group = c.comparison_benchmark_group("union_string_100000");
    group.bench_function("pomap", |b| {
        b.iter(|| black_box(pa.union(&pb)));
    });
    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            let mut u = ha.clone();
            u.extend(hb.iter().map(|(k, v)| (k.clone(), *v)));
            black_box(u)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_union, bench_union_string);
criterion_main!(benches);
