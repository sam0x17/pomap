//! Analyzes tag designs for SIMD scanning of PoMap windows.
//!
//! Key insight: we KNOW the target's ideal_slot during scan, and every entry
//! is at or after its ideal_slot. So instead of storing left-of-pivot hash bits,
//! we store DISPLACEMENT (position - ideal_slot) which exactly reconstructs
//! the ideal_slot. Only sub-bucket bits (right of pivot) are needed in the tag.
//!
//! Run: cargo bench --bench scan_depth

use ahash::AHasher;
use pomap::PoMap;
use std::hash::BuildHasherDefault;

type Map = PoMap<u64, u64, BuildHasherDefault<AHasher>>;

fn main() {
    let sizes: Vec<usize> = vec![
        100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
    ];

    // =========================================================================
    // Section 1: Displacement distribution — does it fit in 4 bits?
    // =========================================================================
    println!();
    println!("=== Displacement Distribution (position - ideal_slot) ===");
    println!();
    println!("Each entry is at position p >= ideal_slot(hash).");
    println!("Displacement = p - ideal_slot. Must fit in the tag's displacement field.");
    println!();

    println!(
        "{:>12} {:>5} {:>4} | {:>5} {:>5} {:>5} | {:>6} {:>6}",
        "entries", "ibits", "ms", "p50", "p99", "max", "fits4b", "fits3b"
    );
    println!("{}", "-".repeat(70));

    for &n in &sizes {
        let mut map = Map::default();
        for i in 0..n as u64 {
            map.insert(i, i);
        }

        let index_shift = map.index_shift();
        let index_bits = 64 - index_shift;
        let max_scan = map.max_scan();
        let slots = map.occupied_slot_hashes();

        let mut displacements: Vec<usize> = Vec::with_capacity(slots.len());
        for &(pos, hash) in &slots {
            let ideal = (hash >> index_shift) as usize;
            let disp = pos - ideal;
            displacements.push(disp);
        }

        displacements.sort_unstable();
        let np = displacements.len();
        let p50 = displacements[np / 2];
        let p99 = displacements[(np * 99 / 100).min(np - 1)];
        let max = *displacements.last().unwrap();
        let fits_4bit = displacements.iter().filter(|&&d| d < 16).count();
        let fits_3bit = displacements.iter().filter(|&&d| d < 8).count();

        println!(
            "{:>12} {:>5} {:>4} | {:>5} {:>5} {:>5} | {:>5.1}% {:>5.1}%",
            n,
            index_bits,
            max_scan,
            p50,
            p99,
            max,
            100.0 * fits_4bit as f64 / np as f64,
            100.0 * fits_3bit as f64 / np as f64
        );
    }

    // =========================================================================
    // Section 2: Tag effectiveness — displacement + sub-bucket bits
    // =========================================================================
    println!();
    println!("=== Tag Design: displacement + sub-bucket MSB ===");
    println!();
    println!("Tag = (displacement: D bits, sub_bucket_msb: R bits)");
    println!("During scan at position p, offset = p - target_ideal_slot:");
    println!("  displacement > offset → entry from earlier bucket → SKIP");
    println!("  displacement < offset → entry from later bucket → STOP");
    println!("  displacement == offset → same bucket → compare R sub-bucket bits");
    println!("  sub-bucket tag differs → ordering determined (RESOLVED)");
    println!("  sub-bucket tag matches → must recompute full hash (TIEBREAK)");
    println!("  NO OVERFLOW possible — displacement is exact.");
    println!();

    let tag_configs: Vec<(usize, usize, &str)> = vec![
        (4, 4, " 1 byte "),
        (4, 8, "1.5 byte"),
        (4, 12, " 2 bytes"),
        (3, 5, " 1 byte "),
        (3, 13, " 2 bytes"),
    ];

    for &n in &[1_000_000usize, 10_000_000] {
        let mut map = Map::default();
        for i in 0..n as u64 {
            map.insert(i, i);
        }

        let index_shift = map.index_shift();
        let index_bits = 64 - index_shift;
        let max_scan = map.max_scan();
        let ideal_range = map.capacity() - max_scan;
        let slots = map.occupied_slot_hashes();

        println!(
            "--- N={}, index_bits={}, pivot=bit {}, max_scan={} ---",
            n, index_bits, index_shift, max_scan
        );
        println!();
        println!(
            "  {:>2}D+{:<2}R {:>8} {:>5} | {:>10} {:>10} {:>10} | {:>10}",
            "", "", "", "bits", "skip/stop%", "resolved%", "tiebreak%", "total_cmp"
        );

        for &(d_bits, r_bits, label) in &tag_configs {
            let d_max = (1usize << d_bits) - 1;
            let r_mask_shift = index_shift.saturating_sub(r_bits);

            let mut total = 0u64;
            let mut skip_stop = 0u64;
            let mut resolved = 0u64;
            let mut tiebreak = 0u64;
            let mut d_overflow = 0u64;

            for bucket in 0..ideal_range {
                let lo = slots.partition_point(|&(pos, _)| pos < bucket);
                let hi = slots.partition_point(|&(pos, _)| pos < bucket + max_scan);
                let window = &slots[lo..hi];

                // Simulate scanning this window for every possible target in the window
                // Actually, we want per-comparison stats, so just look at adjacent pairs
                // as they represent what the scan loop compares
                for pair in window.windows(2) {
                    let (pos1, h1) = pair[0];
                    let (pos2, h2) = pair[1];
                    total += 1;

                    let ideal1 = (h1 >> index_shift) as usize;
                    let ideal2 = (h2 >> index_shift) as usize;
                    let d1 = pos1 - ideal1;
                    let d2 = pos2 - ideal2;

                    // Check displacement overflow (displacement doesn't fit in D bits)
                    if d1 > d_max || d2 > d_max {
                        d_overflow += 1;
                        continue;
                    }

                    if ideal1 != ideal2 {
                        // Different ideal_slots → displacement comparison resolves it
                        skip_stop += 1;
                    } else {
                        // Same ideal_slot → need sub-bucket comparison
                        let sub1 = (h1 >> r_mask_shift) & ((1u64 << r_bits) - 1);
                        let sub2 = (h2 >> r_mask_shift) & ((1u64 << r_bits) - 1);
                        if sub1 != sub2 {
                            resolved += 1;
                        } else {
                            tiebreak += 1;
                        }
                    }
                }
            }

            if total == 0 {
                continue;
            }
            println!(
                "  {:>2}D+{:<2}R {} {:>5} | {:>9.4}% {:>9.4}% {:>9.4}% | {:>10}{}",
                d_bits,
                r_bits,
                label,
                d_bits + r_bits,
                100.0 * skip_stop as f64 / total as f64,
                100.0 * resolved as f64 / total as f64,
                100.0 * tiebreak as f64 / total as f64,
                total,
                if d_overflow > 0 {
                    format!("  ({}  disp overflow)", d_overflow)
                } else {
                    String::new()
                }
            );
        }
        println!();
    }

    // =========================================================================
    // Section 3: Same-bucket pair frequency and sub-bucket MSB distribution
    // =========================================================================
    println!("=== Same-Bucket Pair Analysis ===");
    println!();
    println!("What fraction of adjacent-pair comparisons within a window are same-bucket?");
    println!("These are the only ones that need sub-bucket bits.");
    println!();

    for &n in &sizes {
        let mut map = Map::default();
        for i in 0..n as u64 {
            map.insert(i, i);
        }

        let index_shift = map.index_shift();
        let index_bits = 64 - index_shift;
        let max_scan = map.max_scan();
        let ideal_range = map.capacity() - max_scan;
        let slots = map.occupied_slot_hashes();

        let mut total = 0u64;
        let mut same_bucket = 0u64;
        let mut sub_msb_hist = vec![0u64; 65]; // MSB position of sub-bucket XOR

        for bucket in 0..ideal_range {
            let lo = slots.partition_point(|&(pos, _)| pos < bucket);
            let hi = slots.partition_point(|&(pos, _)| pos < bucket + max_scan);
            let window = &slots[lo..hi];
            for pair in window.windows(2) {
                let h1 = pair[0].1;
                let h2 = pair[1].1;
                let ideal1 = (h1 >> index_shift) as usize;
                let ideal2 = (h2 >> index_shift) as usize;
                total += 1;

                if ideal1 == ideal2 {
                    same_bucket += 1;
                    // Sub-bucket XOR — only the bits below pivot
                    let sub_xor = (h1 ^ h2) & ((1u64 << index_shift) - 1);
                    if sub_xor > 0 {
                        let msb = 63 - sub_xor.leading_zeros() as usize;
                        sub_msb_hist[msb] += 1;
                    }
                }
            }
        }

        println!(
            "  N={:>10}  ibits={:>2}  ms={:>2}  same_bucket: {:>6.2}% of {} comparisons",
            n,
            index_bits,
            max_scan,
            100.0 * same_bucket as f64 / total as f64,
            total
        );
    }

    // Detailed sub-bucket MSB distribution at 1M
    println!();
    println!("--- Sub-bucket MSB distribution at 1M entries (same-bucket pairs only) ---");
    println!();

    let n = 1_000_000usize;
    let mut map = Map::default();
    for i in 0..n as u64 {
        map.insert(i, i);
    }

    let index_shift = map.index_shift();
    let max_scan = map.max_scan();
    let ideal_range = map.capacity() - max_scan;
    let slots = map.occupied_slot_hashes();

    let mut same_total = 0u64;
    let mut sub_msb_hist = vec![0u64; 65];

    for bucket in 0..ideal_range {
        let lo = slots.partition_point(|&(pos, _)| pos < bucket);
        let hi = slots.partition_point(|&(pos, _)| pos < bucket + max_scan);
        let window = &slots[lo..hi];
        for pair in window.windows(2) {
            let h1 = pair[0].1;
            let h2 = pair[1].1;
            if (h1 >> index_shift) == (h2 >> index_shift) {
                same_total += 1;
                let sub_xor = (h1 ^ h2) & ((1u64 << index_shift) - 1);
                if sub_xor > 0 {
                    let msb = 63 - sub_xor.leading_zeros() as usize;
                    sub_msb_hist[msb] += 1;
                }
            }
        }
    }

    println!(
        "  pivot = bit {}, same-bucket comparisons = {}",
        index_shift, same_total
    );
    println!("  MSB of sub-bucket XOR (distance from pivot, geometric from top):");
    println!();

    let mut cumul = 0u64;
    for bit in (0..index_shift).rev() {
        let c = sub_msb_hist[bit];
        if c == 0 {
            continue;
        }
        cumul += c;
        let dist = index_shift - 1 - bit; // 0 = top sub-bucket bit, 1 = next, etc.
        let bar = "#".repeat((60.0 * c as f64 / same_total as f64) as usize);
        println!(
            "    bit {:>2} (pivot-{:>2}): {:>7.3}%  (cumul {:>6.2}%)  {}",
            bit,
            dist + 1,
            100.0 * c as f64 / same_total as f64,
            100.0 * cumul as f64 / same_total as f64,
            bar
        );
    }

    // =========================================================================
    // Section 4: SIMD feasibility
    // =========================================================================
    println!();
    println!("=== SIMD Feasibility ===");
    println!();
    println!("Tag = (4-bit displacement, 12-bit sub-bucket MSB) = 2 bytes per slot");
    println!();
    println!("Scan algorithm with SIMD:");
    println!("  1. Load max_scan displacement nibbles (one SIMD load)");
    println!("  2. Compare displacement vs offset vector [0,1,2,...,max_scan-1]");
    println!("     displacement > offset → skip (earlier bucket)");
    println!("     displacement < offset → stop (later bucket)");
    println!("     displacement == offset → same bucket → check sub-bucket tag");
    println!("  3. For same-bucket matches, compare 12-bit sub-bucket tags");
    println!("     If different → ordering known. If same → full hash tiebreaker.");
    println!();
    println!("Memory savings:");
    for &n in &[1_000_000usize, 10_000_000] {
        let mut map = Map::default();
        for i in 0..n as u64 {
            map.insert(i, i);
        }
        let cap = map.capacity();
        let hash8 = cap * 8;
        let tag2 = cap * 2;
        println!(
            "  N={:>10}  capacity={:>10}  8-byte hashes: {:>6} KB  2-byte tags: {:>6} KB  saving: {:.0}%",
            n,
            cap,
            hash8 / 1024,
            tag2 / 1024,
            100.0 * (1.0 - tag2 as f64 / hash8 as f64)
        );
    }
}
