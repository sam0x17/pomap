//! PoMap2: Bucket-based prefix-ordered hash map prototype.
//!
//! Flat layout: `[tags: u8 × total_slots] [entries: (K,V) × total_slots]`
//! Bucket `b` owns slots `[b*16 .. b*16+16)`. Tags are stable across resizes
//! (7 free resizes before rehashing). SIMD splat scan per bucket.
//! Robin Hood ordering within each bucket ensures deterministic iteration.

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

const BUCKET_SIZE: usize = 16;
const MIN_BUCKETS: usize = 2;
const GROWTH_FACTOR: usize = 2;
/// Control byte layout:
/// - 0x00-0x7F: occupied entry with 7-bit hash tag (bit 7 = 0)
/// - EMPTY (0x80): slot is vacant (bit 7 = 1)
const EMPTY: u8 = 0x80;

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    h.saturating_sub(1)
}

/// Tag = 7 bits from hash. Bit 7 is always 0 for valid entries.
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
        let tags_layout = Layout::array::<u8>(total_slots)
            .expect("PoMap2 tag layout overflow");
        let entries_layout = Layout::array::<MaybeUninit<(K, V)>>(total_slots)
            .expect("PoMap2 entry layout overflow");
        let (layout, entries_offset) = tags_layout
            .extend(entries_layout)
            .expect("PoMap2 combined layout overflow");
        let layout = layout.pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };

        let tags = ptr.as_ptr() as *mut u8;
        let entries = unsafe { ptr.as_ptr().add(entries_offset) as *mut MaybeUninit<(K, V)> };

        // All slots start empty.
        unsafe { ptr::write_bytes(tags, EMPTY, total_slots) };

        Self { ptr, total_slots, tags, entries, layout, _marker: PhantomData }
    }
}

impl<K: Key, V: Value> Drop for Slots<K, V> {
    fn drop(&mut self) {
        for i in 0..self.total_slots {
            let tag = unsafe { *self.tags.add(i) };
            if tag & 0x80 == 0 {
                unsafe { ptr::drop_in_place((*self.entries.add(i)).as_mut_ptr()) };
            }
        }
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Robin Hood insert into a bucket region [start, start+16).
/// Places entry at its ideal offset (tag >> 3), displacing higher-tag entries forward.
/// Equal tags are tiebroken by full hash for determinism.
/// Caller must ensure there is at least one EMPTY slot in the bucket.
#[inline]
unsafe fn robin_hood_place<K: Key, V: Value, H: BuildHasher>(
    tags: *mut u8,
    entries: *mut MaybeUninit<(K, V)>,
    start: usize,
    mut tag: u8,
    mut kv: (K, V),
    mut hash: u64,
    hash_builder: &H,
) {
    let mut offset = (tag >> 3) as usize;
    let mut hash_valid = true;
    loop {
        let idx = start + (offset & (BUCKET_SIZE - 1));
        let slot_tag = unsafe { *tags.add(idx) };
        if slot_tag & 0x80 != 0 {
            unsafe {
                *tags.add(idx) = tag;
                *entries.add(idx) = MaybeUninit::new(kv);
            }
            return;
        }
        let should_swap = if slot_tag > tag {
            true
        } else if slot_tag == tag {
            if !hash_valid {
                hash = encode_hash(hash_builder.hash_one(&kv.0));
                hash_valid = true;
            }
            let existing = unsafe { &*(*entries.add(idx)).as_ptr() };
            let existing_hash = encode_hash(hash_builder.hash_one(&existing.0));
            existing_hash > hash
        } else {
            false
        };
        if should_swap {
            let displaced_kv = unsafe { (*entries.add(idx)).assume_init_read() };
            unsafe {
                *tags.add(idx) = tag;
                *entries.add(idx) = MaybeUninit::new(kv);
            }
            tag = slot_tag;
            kv = displaced_kv;
            hash_valid = false;
        }
        offset += 1;
    }
}

struct Meta {
    num_buckets: usize,
    bucket_shift: usize,
    tag_shift: usize,
}

impl Meta {
    fn new(num_buckets: usize) -> Self {
        let bucket_bits = num_buckets.trailing_zeros() as usize;
        let bucket_shift = 64usize.saturating_sub(bucket_bits);
        let tag_shift = bucket_shift.saturating_sub(7);
        Self { num_buckets, bucket_shift, tag_shift }
    }

    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        if self.bucket_shift >= 64 { 0 } else { (hash >> self.bucket_shift) as usize }
    }

    #[inline(always)]
    fn tag(&self, hash: u64) -> u8 {
        make_tag(hash, self.tag_shift)
    }

    #[inline(always)]
    fn bucket_start(&self, bucket_idx: usize) -> usize {
        bucket_idx * BUCKET_SIZE
    }
}

/// PoMap2: Bucket-based prefix-ordered hash map.
pub struct PoMap2<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher> {
    len: usize,
    stale_bits: u8,
    meta: Meta,
    slots: Slots<K, V>,
    hash_builder: H,
}

impl<K: Key, V: Value, H: BuildHasher> PoMap2<K, V, H> {
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_BUCKETS * BUCKET_SIZE, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let num_buckets = ((capacity + BUCKET_SIZE - 1) / BUCKET_SIZE)
            .next_power_of_two()
            .max(MIN_BUCKETS);
        let total_slots = num_buckets * BUCKET_SIZE;
        Self {
            len: 0,
            stale_bits: 0,
            meta: Meta::new(num_buckets),
            slots: Slots::new(total_slots),
            hash_builder,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.len }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn capacity(&self) -> usize { self.meta.num_buckets * BUCKET_SIZE }

    /// Load tags for a bucket as a SIMD vector.
    #[inline(always)]
    fn load_tags(&self, bucket_start: usize) -> u8x16 {
        let ptr = unsafe { self.slots.tags.add(bucket_start) };
        let tags = unsafe { ptr::read_unaligned(ptr as *const [u8; 16]) };
        u8x16::new(tags)
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let bi = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        let start = self.meta.bucket_start(bi);

        let tags_vec = self.load_tags(start);
        let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let (k, v) = unsafe { &*(*self.slots.entries.add(start + offset)).as_ptr() };
            if k == key {
                return Some(v);
            }
        }
        None
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let bi = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        let start = self.meta.bucket_start(bi);

        let tags_vec = self.load_tags(start);
        let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let entry = unsafe { &mut *(*self.slots.entries.add(start + offset)).as_mut_ptr() };
            if &entry.0 == key {
                return Some(&mut entry.1);
            }
        }
        None
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let bi = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        let start = self.meta.bucket_start(bi);

        // Single SIMD load.
        let tags_vec = self.load_tags(start);

        // Check for existing key (replacement).
        let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let entry = unsafe { &mut *(*self.slots.entries.add(start + offset)).as_mut_ptr() };
            if entry.0 == key {
                let old = mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }

        // Check for available slot (bit 7 set = EMPTY).
        let avail_mask = tags_vec.move_mask() as u32;
        if avail_mask == 0 {
            // Bucket full — grow and retry.
            self.grow();
            return self.insert(key, value);
        }

        // Robin Hood placement for deterministic ordering.
        unsafe {
            robin_hood_place(
                self.slots.tags, self.slots.entries,
                start, tag, (key, value), hash, &self.hash_builder,
            );
        }
        self.len += 1;
        None
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let bi = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        let start = self.meta.bucket_start(bi);

        let tags_vec = self.load_tags(start);
        let mut mask = tags_vec.cmp_eq(u8x16::new([tag; 16])).move_mask() as u32;
        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let idx = start + offset;
            let matches = unsafe {
                let (k, _) = &*(*self.slots.entries.add(idx)).as_ptr();
                k == key
            };
            if matches {
                let (_, value) = unsafe { (*self.slots.entries.add(idx)).assume_init_read() };
                // No overflow → safe to mark EMPTY directly.
                unsafe { *self.slots.tags.add(idx) = EMPTY };
                self.len -= 1;
                return Some(value);
            }
        }
        None
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    fn grow(&mut self) {
        let new_num_buckets = self.meta.num_buckets * GROWTH_FACTOR;
        let new_meta = Meta::new(new_num_buckets);
        let new_total = new_num_buckets * BUCKET_SIZE;
        let new_slots = Slots::new(new_total);

        let can_fast = self.stale_bits < 7; // 7-bit tags → 7 free resizes
        let old_tag_shift = self.meta.tag_shift;

        for i in 0..self.slots.total_slots {
            let old_tag = unsafe { *self.slots.tags.add(i) };
            if old_tag & 0x80 != 0 { continue; }

            let (key, value) = unsafe { (*self.slots.entries.add(i)).assume_init_read() };
            unsafe { *self.slots.tags.add(i) = EMPTY };

            let hash = encode_hash(self.hash_builder.hash_one(&key));

            if can_fast {
                let new_bi = new_meta.bucket_index(hash);
                let new_start = new_bi * BUCKET_SIZE;
                unsafe {
                    robin_hood_place(
                        new_slots.tags, new_slots.entries,
                        new_start, old_tag, (key, value), hash, &self.hash_builder,
                    );
                }
            } else {
                let new_bi = new_meta.bucket_index(hash);
                let new_tag = new_meta.tag(hash);
                let new_start = new_bi * BUCKET_SIZE;
                unsafe {
                    robin_hood_place(
                        new_slots.tags, new_slots.entries,
                        new_start, new_tag, (key, value), hash, &self.hash_builder,
                    );
                }
            }
        }

        if can_fast {
            self.stale_bits += 1;
        } else {
            self.stale_bits = 0;
        }

        self.slots = new_slots;
        self.meta = new_meta;
        if can_fast {
            self.meta.tag_shift = old_tag_shift;
        }
    }

    /// Iterate all entries in bucket order, skipping vacancies.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        (0..self.slots.total_slots).filter_map(move |i| {
            let tag = unsafe { *self.slots.tags.add(i) };
            if tag & 0x80 == 0 {
                let (k, v) = unsafe { &*(*self.slots.entries.add(i)).as_ptr() };
                Some((k, v))
            } else {
                None
            }
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
            stale_bits: self.stale_bits,
            meta: Meta {
                num_buckets: self.meta.num_buckets,
                bucket_shift: self.meta.bucket_shift,
                tag_shift: self.meta.tag_shift,
            },
            slots: new_slots,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher> Drop for PoMap2<K, V, H> {
    fn drop(&mut self) {
        // Slots::Drop handles entry cleanup + dealloc.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHasher;
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    type TestMap = PoMap2<u64, u64, BuildHasherDefault<AHasher>>;

    fn new_map() -> TestMap {
        PoMap2::with_hasher(BuildHasherDefault::default())
    }

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
                    "missing key {} after inserting {} (len={}, buckets={})",
                    j, i, map.len(), map.meta.num_buckets);
            }
        }
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn grow_large() {
        let mut map = new_map();
        for i in 0..10_000u64 {
            map.insert(i, i);
        }
        for i in 0..10_000u64 {
            assert_eq!(map.get(&i), Some(&i), "missing key {}", i);
        }
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
                0 => {
                    let a = map.insert(key, val);
                    let b = expected.insert(key, val);
                    assert_eq!(a, b, "insert mismatch for key {}", key);
                }
                1 => {
                    let a = map.get(&key).copied();
                    let b = expected.get(&key).copied();
                    assert_eq!(a, b, "get mismatch for key {}", key);
                }
                _ => {
                    let a = map.remove(&key);
                    let b = expected.remove(&key);
                    assert_eq!(a, b, "remove mismatch for key {}", key);
                }
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
    fn fast_resize_preserves() {
        let mut map = new_map();
        for i in 0..1000u64 {
            map.insert(i, i);
        }
        assert!(map.stale_bits > 0 && map.stale_bits <= 7);
        for i in 0..1000u64 {
            assert_eq!(map.get(&i), Some(&i), "missing key {} after resizes", i);
        }
    }
}
