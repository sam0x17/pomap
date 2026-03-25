//! PoMap2: Bucket-based prefix-ordered hash map prototype.
//!
//! Each bucket holds up to 16 entries with 1-byte tags for SIMD scanning.
//! Tags are stable across resizes (8 free resizes before rehashing).
//! Entries within each bucket are sorted by tag (then hash/key tiebreaker).

use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
};
use wide::u8x16;

use crate::{Key, PoMapBuildHasher, Value};

const BUCKET_SIZE: usize = 16;
const MIN_BUCKETS: usize = 1;
const GROWTH_FACTOR: usize = 2;

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    h.saturating_sub(1)
}

/// Tag = 8 bits from hash, just below the bucket prefix bits.
/// Stable across resizes: each resize consumes the MSB (which becomes the new bucket bit).
#[inline(always)]
const fn make_tag(hash: u64, tag_shift: usize) -> u8 {
    ((hash >> tag_shift) & 0xFF) as u8
}

/// A single bucket: 16 tag bytes + count + 16 entries.
/// Tags and entries at the same index correspond to the same element.
#[repr(C)]
struct Bucket<K: Key, V: Value> {
    tags: [u8; BUCKET_SIZE],
    count: u8,
    _pad: [u8; 15], // align entries
    entries: [MaybeUninit<(K, V)>; BUCKET_SIZE],
}

impl<K: Key, V: Value> Bucket<K, V> {
    fn new() -> Self {
        Self {
            tags: [0xFF; BUCKET_SIZE], // 0xFF = vacant sentinel
            count: 0,
            _pad: [0; 15],
            entries: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    /// SIMD scan for entries matching `target_tag`.
    #[inline(always)]
    fn scan_candidates(&self, target_tag: u8) -> u32 {
        let tags_vec = u8x16::new(self.tags);
        let target_vec = u8x16::new([target_tag; 16]);
        (tags_vec.cmp_eq(target_vec).move_mask() as u32) & ((1u32 << self.count) - 1)
    }

    /// Find a key in this bucket. Returns Some(&V) if found.
    #[inline]
    fn get(&self, target_tag: u8, key: &K) -> Option<&V> {
        let mut mask = self.scan_candidates(target_tag);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let (k, v) = unsafe { self.entries[idx].assume_init_ref() };
            if k == key {
                return Some(v);
            }
        }
        None
    }

    /// Insert a key-value pair in sorted position. Returns old value if key existed.
    /// Returns Err((key, value)) if bucket is full.
    fn insert(
        &mut self,
        target_tag: u8,
        hash: u64,
        key: K,
        value: V,
        hash_builder: &impl BuildHasher,
    ) -> Result<Option<V>, (K, V)> {
        // Check for existing key (replacement).
        let mut mask = self.scan_candidates(target_tag);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let entry = unsafe { self.entries[idx].assume_init_mut() };
            if entry.0 == key {
                let old = core::mem::replace(&mut entry.1, value);
                return Ok(Some(old));
            }
        }

        // Bucket full?
        if self.count as usize >= BUCKET_SIZE {
            return Err((key, value));
        }

        // Find insertion point: scan tags for sorted position.
        let count = self.count as usize;
        let mut insert_idx = count; // default: append at end
        for i in 0..count {
            if self.tags[i] > target_tag {
                insert_idx = i;
                break;
            }
            if self.tags[i] == target_tag {
                // Tiebreaker: compare full hashes.
                let (slot_key, _) = unsafe { self.entries[i].assume_init_ref() };
                let slot_hash = encode_hash(hash_builder.hash_one(slot_key));
                if slot_hash > hash || (slot_hash == hash && slot_key > &key) {
                    insert_idx = i;
                    break;
                }
            }
        }

        // Shift elements right to make room.
        if insert_idx < count {
            unsafe {
                // Shift tags
                ptr::copy(
                    self.tags.as_ptr().add(insert_idx),
                    self.tags.as_mut_ptr().add(insert_idx + 1),
                    count - insert_idx,
                );
                // Shift entries
                ptr::copy(
                    self.entries.as_ptr().add(insert_idx),
                    self.entries.as_mut_ptr().add(insert_idx + 1),
                    count - insert_idx,
                );
            }
        }

        self.tags[insert_idx] = target_tag;
        self.entries[insert_idx] = MaybeUninit::new((key, value));
        self.count += 1;

        Ok(None)
    }

    /// Remove a key from this bucket. Returns the value if found.
    fn remove(&mut self, target_tag: u8, key: &K) -> Option<V> {
        let mut mask = self.scan_candidates(target_tag);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let matches = {
                let (k, _) = unsafe { self.entries[idx].assume_init_ref() };
                k == key
            };
            if matches {
                let count = self.count as usize;
                let (_, value) = unsafe { self.entries[idx].assume_init_read() };

                // Shift elements left to fill gap.
                let remaining = count - idx - 1;
                if remaining > 0 {
                    unsafe {
                        ptr::copy(
                            self.tags.as_ptr().add(idx + 1),
                            self.tags.as_mut_ptr().add(idx),
                            remaining,
                        );
                        ptr::copy(
                            self.entries.as_ptr().add(idx + 1),
                            self.entries.as_mut_ptr().add(idx),
                            remaining,
                        );
                    }
                }
                self.tags[count - 1] = 0xFF;
                self.count -= 1;
                return Some(value);
            }
        }
        None
    }
}

impl<K: Key, V: Value> Drop for Bucket<K, V> {
    fn drop(&mut self) {
        for i in 0..self.count as usize {
            unsafe { self.entries[i].assume_init_drop() };
        }
    }
}

/// Metadata for the bucket array.
struct Meta {
    /// Number of buckets (power of 2).
    num_buckets: usize,
    /// Shift to extract bucket index from hash: `hash >> bucket_shift`.
    bucket_shift: usize,
    /// Shift to extract tag from hash: 8 bits starting at `tag_shift`.
    tag_shift: usize,
}

impl Meta {
    fn new(num_buckets: usize) -> Self {
        let bucket_bits = num_buckets.trailing_zeros() as usize;
        let bucket_shift = 64usize.saturating_sub(bucket_bits);
        // Tag is the 8 bits just below the bucket prefix.
        let tag_shift = bucket_shift.saturating_sub(8);
        Self {
            num_buckets,
            bucket_shift,
            tag_shift,
        }
    }

    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        if self.bucket_shift >= 64 { 0 } else { (hash >> self.bucket_shift) as usize }
    }

    #[inline(always)]
    fn tag(&self, hash: u64) -> u8 {
        make_tag(hash, self.tag_shift)
    }
}

/// PoMap2: Bucket-based prefix-ordered hash map.
pub struct PoMap2<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher> {
    len: usize,
    /// How many tag bits have been consumed by fast resizes (0 = fresh, 8 = must rehash).
    stale_bits: u8,
    meta: Meta,
    buckets: NonNull<Bucket<K, V>>,
    layout: Layout,
    hash_builder: H,
    _marker: PhantomData<Bucket<K, V>>,
}

impl<K: Key, V: Value, H: BuildHasher> PoMap2<K, V, H> {
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_BUCKETS * BUCKET_SIZE, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let num_buckets = ((capacity + BUCKET_SIZE - 1) / BUCKET_SIZE)
            .next_power_of_two()
            .max(MIN_BUCKETS);
        let layout = Layout::array::<Bucket<K, V>>(num_buckets)
            .expect("PoMap2 layout overflow");
        let ptr = unsafe { alloc(layout) as *mut Bucket<K, V> };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| handle_alloc_error(layout));

        // Initialize all buckets.
        for i in 0..num_buckets {
            unsafe { ptr::write(ptr.as_ptr().add(i), Bucket::new()) };
        }

        Self {
            len: 0,
            stale_bits: 0,
            meta: Meta::new(num_buckets),
            buckets: ptr,
            layout,
            hash_builder,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.meta.num_buckets * BUCKET_SIZE
    }

    #[inline(always)]
    fn bucket(&self, idx: usize) -> &Bucket<K, V> {
        unsafe { &*self.buckets.as_ptr().add(idx) }
    }

    #[inline(always)]
    fn bucket_mut(&mut self, idx: usize) -> &mut Bucket<K, V> {
        unsafe { &mut *self.buckets.as_ptr().add(idx) }
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let bucket_idx = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        self.bucket(bucket_idx).get(tag, key)
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let bucket_idx = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);

        let bucket = unsafe { &mut *self.buckets.as_ptr().add(bucket_idx) };
        match bucket.insert(tag, hash, key, value, &self.hash_builder) {
            Ok(old) => {
                if old.is_none() {
                    self.len += 1;
                }
                old
            }
            Err((key, value)) => {
                // Bucket full — grow and retry.
                self.grow();
                self.insert(key, value)
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let bucket_idx = self.meta.bucket_index(hash);
        let tag = self.meta.tag(hash);
        let result = self.bucket_mut(bucket_idx).remove(tag, key);
        if result.is_some() {
            self.len -= 1;
        }
        result
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    fn grow(&mut self) {
        let old_num_buckets = self.meta.num_buckets;
        let new_num_buckets = old_num_buckets * GROWTH_FACTOR;
        let new_meta = Meta::new(new_num_buckets);

        let new_layout = Layout::array::<Bucket<K, V>>(new_num_buckets)
            .expect("PoMap2 layout overflow on grow");
        let new_ptr = unsafe { alloc(new_layout) as *mut Bucket<K, V> };
        let new_ptr = NonNull::new(new_ptr).unwrap_or_else(|| handle_alloc_error(new_layout));
        for i in 0..new_num_buckets {
            unsafe { ptr::write(new_ptr.as_ptr().add(i), Bucket::new()) };
        }

        let can_fast_resize = self.stale_bits < 8;
        let old_tag_shift = self.meta.tag_shift;

        if can_fast_resize {
            // FAST RESIZE: each old bucket splits into two new buckets.
            // The split bit is the next unconsumed tag bit, counting down from bit 7.
            // After k fast resizes (stale_bits = k), the next split uses bit (7 - k).
            // Since entries are sorted by tag, the split point is a clean pivot.
            // Tags stay UNCHANGED.
            let split_bit = 7 - self.stale_bits;
            let split_mask = 1u8 << split_bit;

            for bi in 0..old_num_buckets {
                let old_bucket = unsafe { &*self.buckets.as_ptr().add(bi) };
                let count = old_bucket.count as usize;
                if count == 0 {
                    continue;
                }

                // Find pivot: first entry where the split bit is set.
                let mut pivot = count;
                for i in 0..count {
                    if (old_bucket.tags[i] & split_mask) != 0 {
                        pivot = i;
                        break;
                    }
                }

                let lo_count = pivot;
                let hi_count = count - pivot;

                // Bulk copy low half (tags 0x00-0x7F) → bucket 2*bi
                if lo_count > 0 {
                    let lo_bucket = unsafe { &mut *new_ptr.as_ptr().add(bi * 2) };
                    unsafe {
                        ptr::copy_nonoverlapping(
                            old_bucket.tags.as_ptr(),
                            lo_bucket.tags.as_mut_ptr(),
                            lo_count,
                        );
                        ptr::copy_nonoverlapping(
                            old_bucket.entries.as_ptr(),
                            lo_bucket.entries.as_mut_ptr(),
                            lo_count,
                        );
                    }
                    lo_bucket.count = lo_count as u8;
                }

                // Bulk copy high half (tags 0x80-0xFE) → bucket 2*bi+1
                if hi_count > 0 {
                    let hi_bucket = unsafe { &mut *new_ptr.as_ptr().add(bi * 2 + 1) };
                    unsafe {
                        ptr::copy_nonoverlapping(
                            old_bucket.tags.as_ptr().add(pivot),
                            hi_bucket.tags.as_mut_ptr(),
                            hi_count,
                        );
                        ptr::copy_nonoverlapping(
                            old_bucket.entries.as_ptr().add(pivot),
                            hi_bucket.entries.as_mut_ptr(),
                            hi_count,
                        );
                    }
                    hi_bucket.count = hi_count as u8;
                }
            }
            self.stale_bits += 1;
        } else {
            // FULL REHASH: recompute tags from hash. Each old bucket's entries
            // are rehashed and placed into new buckets with fresh tags.
            for bi in 0..old_num_buckets {
                let old_bucket = unsafe { &*self.buckets.as_ptr().add(bi) };
                for ei in 0..old_bucket.count as usize {
                    let (key, value) = unsafe { old_bucket.entries[ei].assume_init_read() };
                    let hash = encode_hash(self.hash_builder.hash_one(&key));
                    let new_bucket_idx = new_meta.bucket_index(hash);
                    let new_tag = new_meta.tag(hash);

                    let new_bucket = unsafe { &mut *new_ptr.as_ptr().add(new_bucket_idx) };
                    let count = new_bucket.count as usize;

                    // Find sorted insertion point.
                    let mut insert_idx = count;
                    for j in 0..count {
                        if new_bucket.tags[j] > new_tag {
                            insert_idx = j;
                            break;
                        }
                        if new_bucket.tags[j] == new_tag {
                            let (sk, _) = unsafe { new_bucket.entries[j].assume_init_ref() };
                            let sh = encode_hash(self.hash_builder.hash_one(sk));
                            if sh > hash || (sh == hash && sk > &key) {
                                insert_idx = j;
                                break;
                            }
                        }
                    }

                    if insert_idx < count {
                        unsafe {
                            ptr::copy(
                                new_bucket.tags.as_ptr().add(insert_idx),
                                new_bucket.tags.as_mut_ptr().add(insert_idx + 1),
                                count - insert_idx,
                            );
                            ptr::copy(
                                new_bucket.entries.as_ptr().add(insert_idx),
                                new_bucket.entries.as_mut_ptr().add(insert_idx + 1),
                                count - insert_idx,
                            );
                        }
                    }
                    new_bucket.tags[insert_idx] = new_tag;
                    new_bucket.entries[insert_idx] = MaybeUninit::new((key, value));
                    new_bucket.count += 1;
                }
            }
            self.stale_bits = 0;
        }

        // Free old buckets. Zero counts first so Drop doesn't double-free moved entries.
        for bi in 0..old_num_buckets {
            unsafe { (*self.buckets.as_ptr().add(bi)).count = 0 };
        }
        for bi in 0..old_num_buckets {
            unsafe { ptr::drop_in_place(self.buckets.as_ptr().add(bi)) };
        }
        unsafe { dealloc(self.buckets.as_ptr() as *mut u8, self.layout) };

        self.buckets = new_ptr;
        self.layout = new_layout;
        self.meta = new_meta;
        // On fast resize, preserve the old tag_shift (tags are unchanged).
        if can_fast_resize {
            self.meta.tag_shift = old_tag_shift;
        }
    }

    /// Iterate all entries in hash-sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        (0..self.meta.num_buckets).flat_map(move |bi| {
            let bucket = self.bucket(bi);
            (0..bucket.count as usize).map(move |ei| {
                let (k, v) = unsafe { bucket.entries[ei].assume_init_ref() };
                (k, v)
            })
        })
    }

    /// Collect all keys in iteration order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }
}

impl<K: Key, V: Value, H: BuildHasher> Drop for PoMap2<K, V, H> {
    fn drop(&mut self) {
        for bi in 0..self.meta.num_buckets {
            unsafe { ptr::drop_in_place(self.buckets.as_ptr().add(bi)) };
        }
        unsafe { dealloc(self.buckets.as_ptr() as *mut u8, self.layout) };
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
        // Insert enough to trigger growth.
        for i in 0..100u64 {
            map.insert(i, i * 10);
            // Verify all previous inserts still findable after each insert.
            for j in 0..=i {
                assert_eq!(
                    map.get(&j),
                    Some(&(j * 10)),
                    "missing key {} after inserting {} (len={}, num_buckets={})",
                    j,
                    i,
                    map.len(),
                    map.meta.num_buckets
                );
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

        // Insert same keys in different order.
        let keys = [42u64, 7, 99, 3, 55, 21, 88, 11, 33, 66];
        for &k in &keys {
            map1.insert(k, k);
        }
        for &k in keys.iter().rev() {
            map2.insert(k, k);
        }

        let order1: Vec<u64> = map1.keys().copied().collect();
        let order2: Vec<u64> = map2.keys().copied().collect();
        assert_eq!(order1, order2, "iteration order should be deterministic");
    }

    #[test]
    fn deterministic_across_capacities() {
        // Same keys inserted into maps with different starting capacities
        // should produce the same iteration order.
        let keys: Vec<u64> = (0..200).collect();

        let mut map1 = PoMap2::with_capacity_and_hasher(16, BuildHasherDefault::<AHasher>::default());
        let mut map2 = PoMap2::with_capacity_and_hasher(1024, BuildHasherDefault::<AHasher>::default());

        for &k in &keys {
            map1.insert(k, k);
            map2.insert(k, k);
        }

        let order1: Vec<u64> = map1.keys().copied().collect();
        let order2: Vec<u64> = map2.keys().copied().collect();
        assert_eq!(order1, order2, "iteration order should be same across capacities");
    }

    #[test]
    fn fast_resize_count() {
        // Verify that fast resizes happen and entries survive.
        let mut map = new_map();
        for i in 0..10_000u64 {
            map.insert(i, i);
        }
        // Starting from 1 bucket, we need log2(10000/16) ≈ 10 resizes.
        // First 8 are fast, then 1 full rehash (resets stale), then more fast.
        assert!(map.stale_bits <= 8, "stale_bits out of range: {}", map.stale_bits);

        // Verify everything is still findable.
        for i in 0..10_000u64 {
            assert_eq!(map.get(&i), Some(&i), "missing key {} after resizes", i);
        }
    }
}
