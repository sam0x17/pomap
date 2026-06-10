//! PoMap4: best-of-everything prefix-ordered hash map.
//!
//! Layout: `[entries: (u64_hash, K, V) × N]` — AoS, inline hash (from pomap3).
//! The hash shares a cache line with its (K, V), giving the fastest reads of any
//! variant. The get/insert/remove engine is pomap3's: scan from the ideal slot,
//! shift right into the nearest vacancy to keep runs sorted, backshift on remove.
//!
//! The one addition from `main`: the resize repack preserves inter-run gaps via
//! cursor spacing (bump the write cursor past each vacant slot in the old map).
//! This is applied only when *growing* — where the source is near-full so the bump
//! is small and bounded — and leaves ideal slots open in the new map. Shorter,
//! gappier runs make both inserts (closer vacancies → shorter shifts) and removes
//! (shorter backshift chains) substantially faster, while reads stay pomap3-fast.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    hash::BuildHasher,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
};

use crate::{Key, PoMapBuildHasher, Value};

const MIN_IDEAL_RANGE: usize = 16;
const EMPTY_HASH: u64 = u64::MAX;

/// Grow when `len` reaches `ideal_range * LOAD_NUM / LOAD_DEN` (62.5%). A lower
/// load factor keeps runs short, which keeps insert shifts and remove backshift
/// chains short. Mostly free in memory because `ideal_range` is a power of two.
const LOAD_NUM: usize = 5;
const LOAD_DEN: usize = 8;

/// Multiplier applied to `ideal_range` on grow. Higher = fewer rebuilds (less
/// total repack work, ~f/(f-1)·n moves) but more slack memory. Measured dial:
/// factor 2 → insert_allocate ~1.66× hb, avg ~2.03× hb memory; factor 4 →
/// ~1.13× hb inserts but avg ~2.75× (worst ~6.25×) memory. Kept at 2 for memory.
const GROWTH_FACTOR: usize = 2;

/// Smallest `ideal_range` (power of two) that holds `capacity` entries without
/// growing — i.e. whose grow threshold strictly exceeds `capacity`.
#[inline]
fn ideal_range_for(capacity: usize) -> usize {
    // Need ideal_range * LOAD_NUM / LOAD_DEN > capacity.
    let needed = capacity * LOAD_DEN / LOAD_NUM + 1;
    needed.next_power_of_two().max(MIN_IDEAL_RANGE)
}

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    let h = h.saturating_sub(1);
    if h == EMPTY_HASH { EMPTY_HASH - 1 } else { h }
}

#[inline]
const fn padding_for(ideal_range: usize) -> usize {
    let log2 = (usize::BITS - ideal_range.leading_zeros()) as usize;
    let p = log2 * 10;
    let cap = ideal_range / 4;
    if p < cap { p } else { cap }
}

type Entry<K, V> = (u64, K, V);

struct Slots<K: Key, V: Value> {
    ptr: NonNull<u8>,
    total_slots: usize,
    entries: *mut MaybeUninit<Entry<K, V>>,
    layout: Layout,
    _marker: PhantomData<(K, V)>,
}

impl<K: Key, V: Value> Slots<K, V> {
    fn new(total_slots: usize) -> Self {
        let layout = Layout::array::<MaybeUninit<Entry<K, V>>>(total_slots)
            .unwrap()
            .pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };

        let entries = ptr.as_ptr() as *mut MaybeUninit<Entry<K, V>>;
        for i in 0..total_slots {
            unsafe {
                *(entries.add(i) as *mut u64) = EMPTY_HASH;
            }
        }

        Self {
            ptr,
            total_slots,
            entries,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn hash_at(&self, i: usize) -> u64 {
        unsafe { *(self.entries.add(i) as *const u64) }
    }
}

impl<K: Key, V: Value> Drop for Slots<K, V> {
    fn drop(&mut self) {
        for i in 0..self.total_slots {
            if self.hash_at(i) != EMPTY_HASH {
                unsafe { ptr::drop_in_place((*self.entries.add(i)).as_mut_ptr()) };
            }
        }
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct Meta {
    ideal_range: usize,
    slot_shift: usize,
}

impl Meta {
    fn new(ideal_range: usize) -> Self {
        let slot_bits = ideal_range.trailing_zeros() as usize;
        Self {
            ideal_range,
            slot_shift: 64usize.saturating_sub(slot_bits),
        }
    }

    #[inline(always)]
    fn ideal_slot(&self, hash: u64) -> usize {
        if self.slot_shift >= 64 {
            0
        } else {
            (hash >> self.slot_shift) as usize
        }
    }
}

/// PoMap4: inline-hash AoS map with gap-preserving resize.
pub struct PoMap4<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher> {
    len: usize,
    grow_threshold: usize,
    meta: Meta,
    slots: Slots<K, V>,
    hash_builder: H,
}

impl<K: Key, V: Value, H: BuildHasher> PoMap4<K, V, H> {
    /// Creates an empty map with the given hash builder.
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_IDEAL_RANGE, hash_builder)
    }

    /// Creates an empty map with the given capacity and hash builder.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let ideal_range = ideal_range_for(capacity);
        let total_slots = ideal_range + padding_for(ideal_range);
        Self {
            len: 0,
            grow_threshold: ideal_range * LOAD_NUM / LOAD_DEN,
            meta: Meta::new(ideal_range),
            slots: Slots::new(total_slots),
            hash_builder,
        }
    }

    /// Returns the number of entries.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of entries the map can hold without growing.
    pub fn capacity(&self) -> usize {
        self.grow_threshold
    }

    /// Returns a reference to the hasher.
    pub fn hasher(&self) -> &H {
        &self.hash_builder
    }

    /// Returns a reference to the value for `key`.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let entries = self.slots.entries;
        let mut pos = ideal;
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None; // also catches EMPTY_HASH (u64::MAX)
            }
            if stored == hash {
                let (_, k, v) = unsafe { &*(*entries.add(pos)).as_ptr() };
                if k == key {
                    return Some(v);
                }
            }
            pos += 1;
        }
    }

    /// Returns a mutable reference to the value for `key`.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let entries = self.slots.entries;
        let mut pos = ideal;
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let entry = unsafe { &mut *(*entries.add(pos)).as_mut_ptr() };
                if entry.1 == *key {
                    return Some(&mut entry.2);
                }
            }
            pos += 1;
        }
    }

    /// Inserts a key-value pair, returning the old value if present.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.len >= self.grow_threshold {
            self.grow();
        }

        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let ideal = self.meta.ideal_slot(hash);
        let entries = self.slots.entries;

        if self.slots.hash_at(ideal) == EMPTY_HASH {
            unsafe {
                *entries.add(ideal) = MaybeUninit::new((hash, key, value));
            }
            self.len += 1;
            return None;
        }

        self.insert_impl(key, value, hash, ideal)
    }

    #[inline(never)]
    fn insert_impl(&mut self, key: K, value: V, hash: u64, ideal: usize) -> Option<V> {
        let entries = self.slots.entries;
        let mut pos = ideal;
        loop {
            let stored = self.slots.hash_at(pos);
            if stored == EMPTY_HASH {
                unsafe {
                    *entries.add(pos) = MaybeUninit::new((hash, key, value));
                }
                self.len += 1;
                return None;
            }
            if stored == hash {
                let entry = unsafe { &*(*entries.add(pos)).as_ptr() };
                if entry.1 == key {
                    let e = unsafe { &mut *(*entries.add(pos)).as_mut_ptr() };
                    return Some(mem::replace(&mut e.2, value));
                }
            }
            if stored > hash {
                break;
            }
            pos += 1;
            if pos >= self.slots.total_slots {
                self.grow();
                return self.insert(key, value);
            }
        }

        let mut empty_pos = pos + 1;
        while self.slots.hash_at(empty_pos) != EMPTY_HASH {
            empty_pos += 1;
            if empty_pos >= self.slots.total_slots {
                self.grow();
                return self.insert(key, value);
            }
        }

        let shift = empty_pos - pos;
        unsafe {
            ptr::copy(entries.add(pos), entries.add(pos + 1), shift);
            *entries.add(pos) = MaybeUninit::new((hash, key, value));
        }
        self.len += 1;
        None
    }

    /// Removes the entry for `key`, returning the value if present.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let entries = self.slots.entries;

        let mut pos = ideal;
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let entry = unsafe { &*(*entries.add(pos)).as_ptr() };
                if entry.1 == *key {
                    let (_, _, value) = unsafe { (*entries.add(pos)).assume_init_read() };
                    // Backshift the trailing run of displaced entries left by one.
                    let mut end = pos + 1;
                    loop {
                        let nh = self.slots.hash_at(end);
                        if nh == EMPTY_HASH || self.meta.ideal_slot(nh) >= end {
                            break;
                        }
                        end += 1;
                    }
                    let count = end - pos - 1;
                    if count > 0 {
                        unsafe {
                            ptr::copy(entries.add(pos + 1), entries.add(pos), count);
                        }
                    }
                    unsafe {
                        *(entries.add(end - 1) as *mut u64) = EMPTY_HASH;
                    }
                    self.len -= 1;
                    return Some(value);
                }
            }
            pos += 1;
        }
    }

    /// Shrinks the allocation to fit at least `min_capacity` entries.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let target = self.len.max(min_capacity);
        let needed_ideal = ideal_range_for(target);
        if needed_ideal >= self.meta.ideal_range {
            return;
        }
        // Sparse source: a plain `max(cursor, ideal)` repack (no per-vacant bump).
        self.rebuild(needed_ideal, false);
    }

    fn grow(&mut self) {
        // Near-full source: preserve inter-run gaps with per-vacant cursor bumps.
        self.rebuild(self.meta.ideal_range * GROWTH_FACTOR, true);
    }

    /// Repacks live entries into a fresh allocation sized for `new_ideal_range`.
    ///
    /// Each entry lands at `max(cursor, ideal_slot)` so entries only move forward.
    /// When `bump_gaps` is set, the cursor also advances by one for every vacant
    /// slot in the old map, reproducing the inter-run spacing in the new map so
    /// that future inserts at those ideal slots are direct. Only safe to bump when
    /// the source is near-full (grow); on a sparse source it would race the cursor
    /// far past every window.
    fn rebuild(&mut self, new_ideal_range: usize, bump_gaps: bool) {
        let new_meta = Meta::new(new_ideal_range);
        let new_total = new_ideal_range + padding_for(new_ideal_range);
        let new_slots = Slots::new(new_total);

        let mut cursor = 0usize;
        for i in 0..self.slots.total_slots {
            let h = self.slots.hash_at(i);
            if h == EMPTY_HASH {
                if bump_gaps {
                    cursor += 1;
                }
                continue;
            }
            // Move the entry out. The old Slots is mem::forget-en below (its Drop
            // never runs), so there is no need to clear the moved-from slot.
            let entry = unsafe { (*self.slots.entries.add(i)).assume_init_read() };
            cursor = cursor.max(new_meta.ideal_slot(h));
            unsafe {
                *new_slots.entries.add(cursor) = MaybeUninit::new(entry);
            }
            cursor += 1;
        }

        let old = mem::replace(&mut self.slots, new_slots);
        unsafe { dealloc(old.ptr.as_ptr(), old.layout) };
        mem::forget(old);
        self.meta = new_meta;
        self.grow_threshold = new_ideal_range * LOAD_NUM / LOAD_DEN;
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone> Clone for PoMap4<K, V, H> {
    fn clone(&self) -> Self {
        let new_slots = Slots::new(self.slots.total_slots);
        for i in 0..self.slots.total_slots {
            if self.slots.hash_at(i) != EMPTY_HASH {
                unsafe {
                    let e = &*(*self.slots.entries.add(i)).as_ptr();
                    *new_slots.entries.add(i) = MaybeUninit::new((e.0, e.1.clone(), e.2.clone()));
                }
            }
        }
        Self {
            len: self.len,
            grow_threshold: self.grow_threshold,
            meta: Meta::new(self.meta.ideal_range),
            slots: new_slots,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHasher;
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    type TestMap = PoMap4<u64, u64, BuildHasherDefault<AHasher>>;
    fn new_map() -> TestMap {
        PoMap4::with_hasher(BuildHasherDefault::default())
    }

    #[test]
    fn basic() {
        let mut m = new_map();
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        assert_eq!(m.get(&1), Some(&10));
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.get(&3), Some(&30));
        assert_eq!(m.get(&4), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn replace() {
        let mut m = new_map();
        assert_eq!(m.insert(1, 10), None);
        assert_eq!(m.insert(1, 20), Some(10));
        assert_eq!(m.get(&1), Some(&20));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn remove() {
        let mut m = new_map();
        m.insert(1, 10);
        m.insert(2, 20);
        assert_eq!(m.remove(&1), Some(10));
        assert_eq!(m.get(&1), None);
        assert_eq!(m.get(&2), Some(&20));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn grow_large() {
        let mut m = new_map();
        for i in 0..10_000u64 {
            m.insert(i, i);
        }
        for i in 0..10_000u64 {
            assert_eq!(m.get(&i), Some(&i), "missing {}", i);
        }
        assert_eq!(m.len(), 10_000);
    }

    #[test]
    fn shrink() {
        let mut m = new_map();
        for i in 0..1000u64 {
            m.insert(i, i);
        }
        for i in 0..900u64 {
            m.remove(&i);
        }
        m.shrink_to(0);
        for i in 900..1000u64 {
            assert_eq!(m.get(&i), Some(&i));
        }
        for i in 0..900u64 {
            assert_eq!(m.get(&i), None);
        }
    }

    #[test]
    fn matches_hashmap() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let mut m = new_map();
        let mut expected: HashMap<u64, u64> = HashMap::new();
        for _ in 0..50_000 {
            let op: u8 = rng.random_range(0..3);
            let key: u64 = rng.random_range(0..2000);
            let val: u64 = rng.random();
            match op {
                0 => assert_eq!(m.insert(key, val), expected.insert(key, val)),
                1 => assert_eq!(m.get(&key).copied(), expected.get(&key).copied()),
                _ => assert_eq!(m.remove(&key), expected.remove(&key)),
            }
        }
        assert_eq!(m.len(), expected.len());
        for (k, v) in &expected {
            assert_eq!(m.get(k), Some(v));
        }
    }
}
