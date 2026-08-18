//! A **prefix-ordered hash map**: a flat open-addressing table kept globally
//! sorted by hash, with each entry's home slot derived from its hash *prefix*
//! (top bits). `no_std` + `alloc` compatible.
//!
//! # The determinism contract
//!
//! 1. **Iteration order is deterministic regardless of insertion order,
//!    always** — entries iterate in hash order (full-hash collisions are
//!    canonicalized by key order), so the same contents always iterate the
//!    same way, whatever history produced them.
//! 2. **After [`PoMap::compact`], content-equal maps have identical bytes** —
//!    the layout becomes a pure function of contents.
//!
//! The contract makes the map itself lawfully `Eq`, `Ord`, `PartialOrd`, and
//! `Hash`: maps can be keys in other maps, members of sets, and sorted.
//!
//! # Design
//!
//! One invariant — *global hash order survives growth* (the prefix map is
//! monotone in the hash) — pays repeatedly:
//!
//! - **One-cache-line probes**: the full 64-bit hash is stored inline with its
//!   `(K, V)`; a lookup scans from the ideal slot and terminates at the first
//!   stored hash greater than the target (`u64::MAX` doubles as the vacancy
//!   sentinel, and the table's final slot is kept permanently vacant so probe
//!   loops need no bounds checks).
//! - **Comparison-free streaming resize**: rehashing is one sequential pass
//!   with a monotone write cursor — no re-hashing (the hash is stored), no
//!   re-sorting, no scatter. With expensive-to-hash keys (strings, paths,
//!   composite keys) this makes resize-heavy workloads outright faster than
//!   re-hashing tables.
//! - **Deterministic endpoints**: [`PoMap::first_key_value`] /
//!   [`PoMap::pop_first`] give a reproducible "arbitrary element" and drain
//!   order.
//! - **Streaming set algebra**: [`PoMap::union`], [`PoMap::intersection`],
//!   [`PoMap::difference`], [`PoMap::symmetric_difference`] (and the
//!   `| & - ^` operators, plus [`PoMap::append`]) run in O(n + m) as sorted
//!   merges of the two tables — no re-hashing, no probing, no shifting — and
//!   their outputs are born compact-canonical.
//!
//! # Example
//!
//! ```
//! use pomap::PoMap;
//!
//! let mut map: PoMap<String, u32> = PoMap::new();
//! map.insert("alpha".into(), 1);
//! map.insert("beta".into(), 2);
//!
//! assert_eq!(map.get("alpha"), Some(&1));      // Borrow<str> lookups
//! *map.entry("beta".into()).or_insert(0) += 10;
//! assert_eq!(map["beta"], 12);
//!
//! // Same contents, different insertion order => identical iteration order.
//! let mut other: PoMap<String, u32> = PoMap::new();
//! other.insert("beta".into(), 12);
//! other.insert("alpha".into(), 1);
//! assert!(map.iter().eq(other.iter()));
//! assert_eq!(map, other);
//! ```
//!
//! # Growth factor
//!
//! The growth multiplier is a const generic (power of two, default 4):
//! `GROWTH = 2` is the memory-lean point, `4` builds from empty at parity
//! with SwissTable-class maps, `8` builds faster than them at a higher
//! worst-case memory slack. All non-growing operations are unaffected.
//!
//! # Hashing, determinism, and HashDoS
//!
//! The default hasher ([`PoMapBuildHasher`]) is **deterministic per type**,
//! which the map-level `Ord`/`Hash` impls and cross-process reproducibility
//! require. The trade-off is inherent: canonical ordering implies
//! predictable hashes, so an adversary who can choose keys can construct
//! clustered hashes (degrading performance and forcing table growth). Use a
//! randomly seeded `BuildHasher` when resistance to hostile keys matters more
//! than cross-map determinism — point operations keep working; only the
//! cross-*map* comparison/hashing guarantees are waived.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod pomap;
pub use pomap::*;
