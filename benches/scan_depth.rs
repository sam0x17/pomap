//! Measures actual insert probe depth (total hash slot reads per insert) in a PoMap
//! at various sizes, accounting for overlapping windows and cascade displacement.
//!
//! Run: cargo bench --bench scan_depth

use ahash::AHasher;
use pomap::PoMap;
use std::hash::BuildHasherDefault;

type Map = PoMap<u64, u64, BuildHasherDefault<AHasher>>;

fn main() {
    println!();
    println!("=== Final-State Displacement (snapshot after all inserts) ===");
    println!();
    println!("{:>12} {:>10} {:>4} {:>6} {:>8} {:>4} {:>4} {:>4} {:>4} {:>4}",
        "entries", "ideal_rng", "ms", "load%", "mean", "min", "p50", "p90", "p99", "max");
    println!("{}", "-".repeat(80));

    let sizes: Vec<usize> = vec![
        100, 500, 1_000, 5_000, 10_000, 50_000,
        100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
    ];

    for &n in &sizes {
        let mut map = Map::default();
        for i in 0..n as u64 {
            map.insert(i, i);
        }

        let max_scan = map.max_scan();
        let ideal_range = map.capacity() - max_scan;
        let load_pct = 100.0 * n as f64 / ideal_range as f64;

        let histogram = map.displacement_histogram();
        let total: usize = histogram.iter().map(|(_, c)| c).sum();
        let sum: usize = histogram.iter().map(|(d, c)| d * c).sum();
        let mean = sum as f64 / total as f64;
        let min_d = histogram.first().map(|(d, _)| *d).unwrap_or(0);
        let max_d = histogram.last().map(|(d, _)| *d).unwrap_or(0);
        let p50 = percentile_from_histogram(&histogram, total, 500);
        let p90 = percentile_from_histogram(&histogram, total, 900);
        let p99 = percentile_from_histogram(&histogram, total, 990);

        println!("{:>12} {:>10} {:>4} {:>5.1}% {:>8.3} {:>4} {:>4} {:>4} {:>4} {:>4}",
            n, ideal_range, max_scan, load_pct, mean, min_d, p50, p90, p99, max_d);
    }

    // Insert-time probe counts (the real metric — total slot reads per insert)
    println!();
    println!("=== Insert-Time Probe Count (slot reads per insert, including cascade) ===");
    println!();
    println!("This accounts for overlapping windows: when bucket A's displaced entries");
    println!("occupy slots in bucket B's window, B's inserts must scan past them.");
    println!();

    insert_time_probes(&sizes);

    // Theoretical comparison
    println!();
    println!("=== Theory vs Empirical (Knuth's linear probing) ===");
    println!();
    println!("Knuth's formula for successful search in linear probing:");
    println!("  E[probes] = 0.5 * (1 + 1/(1-lambda))");
    println!("This assumes non-overlapping independent probes, so it's a LOWER BOUND");
    println!("for PoMap where overlapping sorted windows create coupling.");
    println!();
    println!("{:>12} {:>6} {:>10} {:>10} {:>10}",
        "entries", "load%", "empirical", "knuth", "ratio");
    println!("{}", "-".repeat(54));

    for &n in &sizes {
        let mut map = Map::default();
        let mut probe_sum = 0usize;
        let mut probe_count = 0usize;

        for i in 0..n as u64 {
            if let Some(p) = map.insert_probe_count(&i) {
                probe_sum += p;
                probe_count += 1;
            }
            map.insert(i, i);
        }

        let max_scan = map.max_scan();
        let ideal_range = map.capacity() - max_scan;
        let lambda = n as f64 / ideal_range as f64;
        let empirical = probe_sum as f64 / probe_count as f64;
        let knuth = if lambda < 0.999 {
            0.5 * (1.0 + 1.0 / (1.0 - lambda))
        } else {
            f64::INFINITY
        };

        println!("{:>12} {:>5.1}% {:>10.3} {:>10.3} {:>9.2}x",
            n, lambda * 100.0, empirical, knuth, empirical / knuth);
    }
}

fn insert_time_probes(sizes: &[usize]) {
    let target = *sizes.last().unwrap();

    let mut map = Map::default();

    // Track probes by load factor band
    struct Band {
        label: &'static str,
        probe_sum: usize,
        count: usize,
        max_probe: usize,
        grows: usize,
        cascade_count: usize,
    }

    impl Band {
        fn new(label: &'static str) -> Self {
            Self { label, probe_sum: 0, count: 0, max_probe: 0, grows: 0, cascade_count: 0 }
        }
        fn mean(&self) -> f64 {
            if self.count == 0 { 0.0 } else { self.probe_sum as f64 / self.count as f64 }
        }
    }

    let mut bands = vec![
        Band::new(" 0-10%"),
        Band::new("10-20%"),
        Band::new("20-30%"),
        Band::new("30-40%"),
        Band::new("40-50%"),
        Band::new("50-60%"),
    ];

    // Also collect all probes for percentiles
    let mut all_probes: Vec<usize> = Vec::with_capacity(target);

    for i in 0..target as u64 {
        let max_scan = map.max_scan();
        let ideal_range = map.capacity() - max_scan;
        let load_pct = 100.0 * map.len() as f64 / ideal_range as f64;
        let band_idx = (load_pct / 10.0) as usize;

        match map.insert_probe_count(&i) {
            Some(probes) => {
                all_probes.push(probes);
                if band_idx < bands.len() {
                    let band = &mut bands[band_idx];
                    band.probe_sum += probes;
                    band.count += 1;
                    if probes > band.max_probe {
                        band.max_probe = probes;
                    }
                    if probes > max_scan {
                        band.cascade_count += 1;
                    }
                }
            }
            None => {
                // Would trigger grow
                if band_idx < bands.len() {
                    bands[band_idx].grows += 1;
                }
            }
        }

        map.insert(i, i);
    }

    println!("{:>8} {:>10} {:>8} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}",
        "load%", "inserts", "mean", "p50", "p90", "p99", "max", "cascades", "grows");
    println!("{}", "-".repeat(88));

    for band in &bands {
        if band.count == 0 && band.grows == 0 {
            continue;
        }
        // Compute percentiles for this band (approximate from all_probes is too mixed)
        // Just show the band-level stats
        println!("{:>8} {:>10} {:>8.2} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}",
            band.label, band.count, band.mean(),
            "-", "-", "-", band.max_probe,
            band.cascade_count, band.grows);
    }

    // Overall with proper percentiles
    all_probes.sort_unstable();
    let n = all_probes.len();
    if n > 0 {
        let sum: usize = all_probes.iter().sum();
        let mean = sum as f64 / n as f64;
        let p50 = all_probes[n * 50 / 100];
        let p90 = all_probes[n * 90 / 100];
        let p99 = all_probes[(n * 99 / 100).min(n - 1)];
        let max_p = *all_probes.last().unwrap();
        let cascade_count: usize = bands.iter().map(|b| b.cascade_count).sum();
        let grow_count: usize = bands.iter().map(|b| b.grows).sum();

        println!("{:>8} {:>10} {:>8.2} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8}",
            "OVERALL", n, mean, p50, p90, p99, max_p, cascade_count, grow_count);
    }
}

fn percentile_from_histogram(histogram: &[(usize, usize)], total: usize, permille: usize) -> usize {
    let target = total * permille / 1000;
    let mut cumulative = 0;
    for (d, c) in histogram {
        cumulative += c;
        if cumulative >= target {
            return *d;
        }
    }
    histogram.last().map(|(d, _)| *d).unwrap_or(0)
}
