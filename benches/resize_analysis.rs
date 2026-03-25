use std::hash::BuildHasherDefault;
use std::collections::HashMap;
use ahash::AHasher;
use pomap::PoMap;
use rand::{Rng, SeedableRng, rngs::StdRng};

type BenchHasherBuilder = BuildHasherDefault<AHasher>;
type BenchPoMap = PoMap<u64, u64, BenchHasherBuilder>;

/// Extract displacement for every key by iterating slots and reading tags.
fn get_key_displacements(map: &BenchPoMap) -> HashMap<u64, usize> {
    let tags = map.occupied_slot_tags();
    let hashes = map.occupied_slot_hashes();
    let mut result: HashMap<u64, usize> = HashMap::new();
    for (&(pos, tag), &(_hpos, _hash)) in tags.iter().zip(hashes.iter()) {
        let disp = (tag >> 4) as usize;
        // We need the key, but we only have position + hash.
        // Use hash as a proxy for the key (u64 keys in this bench).
        // The key IS the value we inserted (key == value).
        // We can recover it from the hash... actually we can't directly.
        // Let's use a different approach: iterate the map and record displacement.
        let _ = (pos, disp);
    }
    // Use a simpler approach: just capture (hash, displacement) and match by hash.
    let mut by_hash = HashMap::new();
    for (&(_pos, tag), &(_hpos, hash)) in tags.iter().zip(hashes.iter()) {
        let disp = (tag >> 4) as usize;
        by_hash.insert(hash, disp);
    }
    by_hash
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let keys: Vec<u64> = (0..50_000).map(|_| rng.random()).collect();

    for &target in &[500usize, 2000, 8000, 30000] {
        let mut map = BenchPoMap::with_hasher(BenchHasherBuilder::default());
        for i in 0..target {
            map.insert(keys[i], keys[i]);
        }

        let cap_before = map.capacity();
        let disps_before = get_key_displacements(&map);

        // Force resize
        let mut extra = 0;
        loop {
            map.insert(keys[target + extra], keys[target + extra]);
            extra += 1;
            if map.capacity() != cap_before { break; }
            if extra > 50000 { break; }
        }

        let cap_after = map.capacity();
        let disps_after = get_key_displacements(&map);

        // Match entries by hash and compare displacements
        let mut transition_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for (hash, &old_disp) in &disps_before {
            if let Some(&new_disp) = disps_after.get(hash) {
                *transition_counts.entry((old_disp, new_disp)).or_insert(0) += 1;
            }
        }

        let mut transitions: Vec<_> = transition_counts.into_iter().collect();
        transitions.sort();

        println!("=== {} entries, cap {} → {} (entries after: {}) ===",
                 target, cap_before, cap_after, target + extra);
        println!("  {:>8} → {:>8} {:>8} {:>8} {:>8}",
                 "old_disp", "new_disp", "old/2", "match?", "count");
        println!("  {}", "-".repeat(55));

        let mut total = 0usize;
        let mut exact_half = 0usize;
        let mut off_by_one = 0usize;

        for ((old_d, new_d), count) in &transitions {
            let half = old_d / 2;
            let is_exact = *new_d == half;
            let is_close = (*new_d as i64 - half as i64).unsigned_abs() <= 1;
            if is_exact { exact_half += count; }
            if is_close { off_by_one += count; }
            total += count;

            let off = *new_d as i64 - half as i64;
            println!("  {:>8} → {:>8} {:>8} {:>8} {:>8}",
                     old_d, new_d, half,
                     if is_exact { "✓" } else if is_close { "~" } else { "✗" },
                     count);
        }
        println!("  Total: {}, exact old/2: {} ({:.1}%), within ±1: {} ({:.1}%)",
                 total,
                 exact_half, exact_half as f64 / total as f64 * 100.0,
                 off_by_one, off_by_one as f64 / total as f64 * 100.0);
        println!();
    }
}
