//! A prefix-ordered hash map with deterministic iteration order.
//!
//! The crate is `no_std` + `alloc` compatible when the `std` feature is disabled.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod pomap;
pub use pomap::*;

#[allow(missing_docs)]
/// Experimental bucket-based variant (prototype).
pub mod pomap2;
