use ahash::AHasher;
use hashbrown::HashMap as HBMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;

type H = BuildHasherDefault<AHasher>;

fn main() {
    let n = 1_000_000usize;
    let keys: Vec<u64> = (0..n as u64).collect();

    let which = std::env::args().nth(1).unwrap_or("pomap".into());

    for _ in 0..3 {
        match which.as_str() {
            "pomap" => {
                let mut map = pomap::PoMap::<u64, u64, H>::with_hasher(H::default());
                for &k in &keys {
                    black_box(map.insert(k, k));
                }
                black_box(&map);
            }
            "hashbrown" => {
                let mut map = HBMap::<u64, u64, H>::with_hasher(H::default());
                for &k in &keys {
                    black_box(map.insert(k, k));
                }
                black_box(&map);
            }
            _ => panic!("usage: prof_insert [pomap|hashbrown]"),
        }
    }
}
