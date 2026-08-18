//! Measured retained heap footprint (bytes/entry) after a build-from-empty,
//! per growth factor. Complements the bench memory table with the G8 point.
//! Run: cargo run --release --example footprint

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hash::BuildHasherDefault,
    sync::atomic::{AtomicUsize, Ordering},
};

use pomap::PoMap;

static NET: AtomicUsize = AtomicUsize::new(0);

struct Track;
unsafe impl GlobalAlloc for Track {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            NET.fetch_add(l.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        NET.fetch_sub(l.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static A: Track = Track;

type H = BuildHasherDefault<ahash::AHasher>;

fn measure<const G: usize>(n: usize) -> f64 {
    let before = NET.load(Ordering::Relaxed);
    let mut m: PoMap<u64, u64, H, G> = PoMap::with_hasher(H::default());
    for i in 0..n as u64 {
        m.insert(i.wrapping_mul(0x9E3779B97F4A7C15), i);
    }
    let bytes = NET.load(Ordering::Relaxed).saturating_sub(before);
    drop(m);
    bytes as f64 / n as f64
}

fn main() {
    println!("{:>10} {:>8} {:>8} {:>8}  (bytes/entry, build-from-empty)", "N", "G2", "G4", "G8");
    for n in [500usize, 5_000, 50_000, 500_000, 5_000_000] {
        println!(
            "{:>10} {:>8.1} {:>8.1} {:>8.1}",
            n,
            measure::<2>(n),
            measure::<4>(n),
            measure::<8>(n)
        );
    }
}
