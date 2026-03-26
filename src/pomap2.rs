//! PoMap2: Prefix-ordered hash map with overlapping SIMD scan windows.
//!
//! Flat layout: `[tags: u8 × total_slots] [entries: (K,V) × total_slots]`
//! Each entry maps to an ideal slot based on its hash prefix. Entries are
//! globally sorted by hash. When the ideal slot is occupied, entries shift
//! right to the nearest EMPTY slot. SIMD scan from ideal slot for get.
//! Grows at global load factor (75%), not per-bucket overflow.

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    hash::BuildHasher,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
};
use wide::u8x16;

use crate::{Key, PoMapBuildHasher, Value};

const SCAN_WIDTH: usize = 16;
/// Extra slots beyond ideal_range for displacement padding.
const PADDING: usize = SCAN_WIDTH;
const EMPTY: u8 = 0x80;
const MIN_IDEAL_RANGE: usize = 16;

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    h.saturating_sub(1)
}

/// Tag = 7-bit fingerprint from hash bits below the ideal-slot prefix.
#[inline(always)]
const fn make_tag(hash: u64, tag_shift: usize) -> u8 {
    ((hash >> tag_shift) & 0x7F) as u8
}

struct Slots<K: Key, V: Value> {
    ptr: NonNull<u8>,
    total_slots: usize,
    tags: *mut u8,
    entries: *mut MaybeUninit<(K, V)>,
    layout: Layout,
    _marker: PhantomData<(K, V)>,
}

impl<K: Key, V: Value> Slots<K, V> {
    fn new(total_slots: usize) -> Self {
        let tags_layout = Layout::array::<u8>(total_slots).unwrap();
        let entries_layout = Layout::array::<MaybeUninit<(K, V)>>(total_slots).unwrap();
        let (layout, entries_offset) = tags_layout.extend(entries_layout).unwrap();
        let layout = layout.pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };

        let tags = ptr.as_ptr() as *mut u8;
        let entries = unsafe { ptr.as_ptr().add(entries_offset) as *mut MaybeUninit<(K, V)> };
        unsafe { ptr::write_bytes(tags, EMPTY, total_slots) };

        Self { ptr, total_slots, tags, entries, layout, _marker: PhantomData }
    }
}

impl<K: Key, V: Value> Drop for Slots<K, V> {
    fn drop(&mut self) {
        for i in 0..self.total_slots {
            if unsafe { *self.tags.add(i) } & 0x80 == 0 {
                unsafe { ptr::drop_in_place((*self.entries.add(i)).as_mut_ptr()) };
            }
        }
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct Meta {
    /// Number of ideal slot positions (power of 2).
    ideal_range: usize,
    /// hash >> slot_shift = ideal_slot index.
    slot_shift: usize,
    /// hash >> tag_shift & 0x7F = tag.
    tag_shift: usize,
}

impl Meta {
    fn new(ideal_range: usize) -> Self {
        let slot_bits = ideal_range.trailing_zeros() as usize;
        let slot_shift = 64usize.saturating_sub(slot_bits);
        let tag_shift = slot_shift.saturating_sub(7);
        Self { ideal_range, slot_shift, tag_shift }
    }

    #[inline(always)]
    fn ideal_slot(&self, hash: u64) -> usize {
        if self.slot_shift >= 64 { 0 } else { (hash >> self.slot_shift) as usize }
    }

    #[inline(always)]
    fn tag(&self, hash: u64) -> u8 {
        make_tag(hash, self.tag_shift)
    }
}

pub struct PoMap2<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher> {
    len: usize,
    meta: Meta,
    slots: Slots<K, V>,
    hash_builder: H,
}

impl<K: Key, V: Value, H: BuildHasher> PoMap2<K, V, H> {
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_IDEAL_RANGE, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let ideal_range = capacity.next_power_of_two().max(MIN_IDEAL_RANGE);
        let total_slots = ideal_range + PADDING;
        Self {
            len: 0,
            meta: Meta::new(ideal_range),
            slots: Slots::new(total_slots),
            hash_builder,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.len }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn capacity(&self) -> usize { self.slots.total_slots }

    #[inline(always)]
    fn load_tags(&self, start: usize) -> u8x16 {
        let ptr = unsafe { self.slots.tags.add(start) };
        u8x16::new(unsafe { ptr::read_unaligned(ptr as *const [u8; 16]) })
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        // Scan from ideal slot. Entries are at or after their ideal position.
        let mut pos = ideal;
        loop {
            let tags_vec = self.load_tags(pos);
            let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
            while mask != 0 {
                let offset = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let (k, v) = unsafe { &*(*self.slots.entries.add(pos + offset)).as_ptr() };
                if k == key {
                    return Some(v);
                }
            }
            // If any EMPTY in this window, entry can't be further out.
            if tags_vec.move_mask() as u32 & 0xFFFF != 0 {
                return None;
            }
            pos += SCAN_WIDTH;
            if pos >= self.slots.total_slots {
                return None;
            }
        }
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        let mut pos = ideal;
        loop {
            let tags_vec = self.load_tags(pos);
            let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
            while mask != 0 {
                let offset = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let entry = unsafe { &mut *(*self.slots.entries.add(pos + offset)).as_mut_ptr() };
                if &entry.0 == key {
                    return Some(&mut entry.1);
                }
            }
            if tags_vec.move_mask() as u32 & 0xFFFF != 0 {
                return None;
            }
            pos += SCAN_WIDTH;
            if pos >= self.slots.total_slots {
                return None;
            }
        }
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Grow at 75% load.
        if self.len * 4 >= self.meta.ideal_range * 3 {
            self.grow();
        }

        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        // Check for replacement: scan from ideal slot for matching tag + key.
        {
            let mut pos = ideal;
            loop {
                let tags_vec = self.load_tags(pos);
                let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
                while mask != 0 {
                    let offset = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let entry = unsafe { &mut *(*self.slots.entries.add(pos + offset)).as_mut_ptr() };
                    if entry.0 == key {
                        return Some(mem::replace(&mut entry.1, value));
                    }
                }
                if tags_vec.move_mask() as u32 & 0xFFFF != 0 {
                    break;
                }
                pos += SCAN_WIDTH;
                if pos >= self.slots.total_slots { break; }
            }
        }

        // Find insert position: scan forward from ideal_slot.
        // Entries are globally sorted by hash. We need to find the first slot
        // where the incumbent has a higher hash than ours, or is EMPTY.
        let tags_ptr = self.slots.tags;
        let entries_ptr = self.slots.entries;

        let mut insert_pos = ideal;
        loop {
            let slot_tag = unsafe { *tags_ptr.add(insert_pos) };
            if slot_tag & 0x80 != 0 {
                // EMPTY — insert here directly.
                unsafe {
                    *tags_ptr.add(insert_pos) = tag;
                    *entries_ptr.add(insert_pos) = MaybeUninit::new((key, value));
                }
                self.len += 1;
                return None;
            }
            // Check if incumbent has a higher hash (we should go before it).
            let incumbent = unsafe { &*(*entries_ptr.add(insert_pos)).as_ptr() };
            let incumbent_hash = encode_hash(self.hash_builder.hash_one(&incumbent.0));
            if incumbent_hash > hash {
                break; // insert_pos found
            }
            insert_pos += 1;
        }

        // Find nearest EMPTY after insert_pos for the shift target.
        let mut empty_pos = insert_pos + 1;
        while unsafe { *tags_ptr.add(empty_pos) } & 0x80 == 0 {
            empty_pos += 1;
        }

        // Shift [insert_pos, empty_pos) right by 1.
        let shift = empty_pos - insert_pos;
        unsafe {
            ptr::copy(tags_ptr.add(insert_pos), tags_ptr.add(insert_pos + 1), shift);
            ptr::copy(entries_ptr.add(insert_pos), entries_ptr.add(insert_pos + 1), shift);
        }

        unsafe {
            *tags_ptr.add(insert_pos) = tag;
            *entries_ptr.add(insert_pos) = MaybeUninit::new((key, value));
        }
        self.len += 1;
        None
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        let mut pos = ideal;
        loop {
            let tags_vec = self.load_tags(pos);
            let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
            while mask != 0 {
                let offset = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let idx = pos + offset;
                if unsafe { &*(*self.slots.entries.add(idx)).as_ptr() }.0 == *key {
                    let (_, value) = unsafe { (*self.slots.entries.add(idx)).assume_init_read() };
                    // Shift entries left to fill gap: pull back displaced entries
                    // until we find one at its ideal position or EMPTY.
                    let tags_ptr = self.slots.tags;
                    let entries_ptr = self.slots.entries;
                    let mut hole = idx;
                    loop {
                        let next = hole + 1;
                        let next_tag = unsafe { *tags_ptr.add(next) };
                        if next_tag & 0x80 != 0 {
                            // Next is EMPTY — done.
                            break;
                        }
                        // Check if next entry is at its ideal position.
                        let next_entry = unsafe { &*(*entries_ptr.add(next)).as_ptr() };
                        let next_hash = encode_hash(self.hash_builder.hash_one(&next_entry.0));
                        let next_ideal = self.meta.ideal_slot(next_hash);
                        if next_ideal > hole {
                            // Next entry's ideal is past the hole — it's not displaced
                            // into the hole region, so stop.
                            break;
                        }
                        // Pull next entry back into hole.
                        unsafe {
                            *tags_ptr.add(hole) = next_tag;
                            ptr::copy_nonoverlapping(entries_ptr.add(next), entries_ptr.add(hole), 1);
                        }
                        hole = next;
                    }
                    unsafe { *tags_ptr.add(hole) = EMPTY };
                    self.len -= 1;
                    return Some(value);
                }
            }
            if tags_vec.move_mask() as u32 & 0xFFFF != 0 {
                return None;
            }
            pos += SCAN_WIDTH;
            if pos >= self.slots.total_slots { return None; }
        }
    }

    pub fn contains_key(&self, key: &K) -> bool { self.get(key).is_some() }

    fn grow(&mut self) {
        let new_ideal_range = self.meta.ideal_range * 2;
        let new_meta = Meta::new(new_ideal_range);
        let new_total = new_ideal_range + PADDING;
        let new_slots = Slots::new(new_total);

        // Re-insert all entries. Since old array is sorted by hash,
        // iterate forward and place each entry at its new ideal position
        // (or first EMPTY after it). Sorted order is preserved.
        for i in 0..self.slots.total_slots {
            let old_tag = unsafe { *self.slots.tags.add(i) };
            if old_tag & 0x80 != 0 { continue; }

            let (key, value) = unsafe { (*self.slots.entries.add(i)).assume_init_read() };
            unsafe { *self.slots.tags.add(i) = EMPTY };

            let hash = encode_hash(self.hash_builder.hash_one(&key));
            let new_ideal = new_meta.ideal_slot(hash);
            let new_tag = new_meta.tag(hash);

            // Find first EMPTY at or after new_ideal.
            let mut slot = new_ideal;
            while unsafe { *new_slots.tags.add(slot) } & 0x80 == 0 {
                slot += 1;
            }
            unsafe {
                *new_slots.tags.add(slot) = new_tag;
                *new_slots.entries.add(slot) = MaybeUninit::new((key, value));
            }
        }

        self.slots = new_slots;
        self.meta = new_meta;
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        (0..self.slots.total_slots).filter_map(move |i| {
            if unsafe { *self.slots.tags.add(i) } & 0x80 == 0 {
                let (k, v) = unsafe { &*(*self.slots.entries.add(i)).as_ptr() };
                Some((k, v))
            } else { None }
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone> Clone for PoMap2<K, V, H> {
    fn clone(&self) -> Self {
        let total = self.slots.total_slots;
        let new_slots = Slots::new(total);
        for i in 0..total {
            let tag = unsafe { *self.slots.tags.add(i) };
            if tag & 0x80 == 0 {
                unsafe {
                    *new_slots.tags.add(i) = tag;
                    let (k, v) = &*(*self.slots.entries.add(i)).as_ptr();
                    *new_slots.entries.add(i) = MaybeUninit::new((k.clone(), v.clone()));
                }
            }
        }
        Self {
            len: self.len,
            meta: Meta { ideal_range: self.meta.ideal_range, slot_shift: self.meta.slot_shift, tag_shift: self.meta.tag_shift },
            slots: new_slots, hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher> Drop for PoMap2<K, V, H> {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHasher;
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    type TestMap = PoMap2<u64, u64, BuildHasherDefault<AHasher>>;
    fn new_map() -> TestMap { PoMap2::with_hasher(BuildHasherDefault::default()) }

    #[test]
    fn basic_insert_get() {
        let mut map = new_map();
        assert_eq!(map.insert(1, 10), None);
        assert_eq!(map.insert(2, 20), None);
        assert_eq!(map.insert(3, 30), None);
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
        assert_eq!(map.get(&4), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn insert_replace() {
        let mut map = new_map();
        assert_eq!(map.insert(1, 10), None);
        assert_eq!(map.insert(1, 20), Some(10));
        assert_eq!(map.get(&1), Some(&20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove() {
        let mut map = new_map();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn grow_basic() {
        let mut map = new_map();
        for i in 0..100u64 {
            map.insert(i, i * 10);
            for j in 0..=i {
                assert_eq!(map.get(&j), Some(&(j * 10)),
                    "missing key {} after inserting {} (len={})",
                    j, i, map.len());
            }
        }
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn grow_large() {
        let mut map = new_map();
        for i in 0..10_000u64 { map.insert(i, i); }
        for i in 0..10_000u64 { assert_eq!(map.get(&i), Some(&i), "missing key {}", i); }
        assert_eq!(map.len(), 10_000);
    }

    #[test]
    fn matches_hashmap() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let mut map = new_map();
        let mut expected: HashMap<u64, u64> = HashMap::new();
        for _ in 0..5000 {
            let op: u8 = rng.random_range(0..3);
            let key: u64 = rng.random_range(0..1000);
            let val: u64 = rng.random();
            match op {
                0 => { assert_eq!(map.insert(key, val), expected.insert(key, val), "insert mismatch for key {}", key); }
                1 => { assert_eq!(map.get(&key).copied(), expected.get(&key).copied(), "get mismatch for key {}", key); }
                _ => { assert_eq!(map.remove(&key), expected.remove(&key), "remove mismatch for key {}", key); }
            }
        }
        assert_eq!(map.len(), expected.len());
    }

    #[test]
    fn deterministic_iteration() {
        let mut map1 = new_map();
        let mut map2 = new_map();
        let keys = [42u64, 7, 99, 3, 55, 21, 88, 11, 33, 66];
        for &k in &keys { map1.insert(k, k); }
        for &k in keys.iter().rev() { map2.insert(k, k); }
        let order1: Vec<u64> = map1.keys().copied().collect();
        let order2: Vec<u64> = map2.keys().copied().collect();
        assert_eq!(order1, order2, "iteration order should be deterministic");
    }

    #[test]
    fn resize_preserves() {
        let mut map = new_map();
        for i in 0..1000u64 { map.insert(i, i); }
        for i in 0..1000u64 { assert_eq!(map.get(&i), Some(&i), "missing key {} after resizes", i); }
    }
}
