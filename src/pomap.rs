use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    slice,
};
use std::hash::DefaultHasher;

/// Minimum capacity we will allow for PoMap
const MIN_CAPACITY: usize = 16;

/// Maximum number of slots to linearly scan starting at the ideal slot.
/// We design the layout so that `[ideal_slot, ideal_slot + MAX_SCAN)` is
/// always in-bounds for the backing Vec.
pub const MAX_SCAN: usize = 16;

/// Number of bits in the hashcode
const HASH_BITS: usize = 64; // we use a 64-bit hashcode

const GROWTH_FACTOR: usize = 2;
const VACANT_HASH: u64 = u64::MAX;

// Simple logging macro gated on the `verbose` feature.
#[cfg(feature = "verbose")]
macro_rules! verbose_log {
    ($($t:tt)*) => {
        println!($($t)*);
    };
}

#[cfg(not(feature = "verbose"))]
macro_rules! verbose_log {
    ($($t:tt)*) => {
        // no-op when verbose is disabled
    };
}

pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

pub trait Value: Clone {}
impl<V: Clone> Value for V {}

struct Entry<K: Key, V: Value> {
    hash: u64,
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
}

#[inline(always)]
fn encode_hash(h: u64) -> u64 {
    if h == VACANT_HASH {
        VACANT_HASH - 1
    } else {
        h
    }
}

impl<K: Key, V: Value> Entry<K, V> {
    #[inline(always)]
    fn vacant() -> Self {
        Self {
            hash: VACANT_HASH,
            key: MaybeUninit::uninit(),
            value: MaybeUninit::uninit(),
        }
    }

    #[inline(always)]
    fn occupied(hash: u64, key: K, value: V) -> Self {
        debug_assert!(hash != VACANT_HASH);
        Self {
            hash,
            key: MaybeUninit::new(key),
            value: MaybeUninit::new(value),
        }
    }

    #[inline(always)]
    fn is_vacant(&self) -> bool {
        self.hash == VACANT_HASH
    }

    #[inline(always)]
    fn occupy(&mut self, hash: u64, key: K, value: V) {
        debug_assert!(hash != VACANT_HASH);
        debug_assert!(self.is_vacant());
        self.hash = hash;
        self.key.write(key);
        self.value.write(value);
    }

    #[inline(always)]
    unsafe fn key_ref(&self) -> &K {
        debug_assert!(!self.is_vacant());
        unsafe { self.key.assume_init_ref() }
    }

    #[inline(always)]
    unsafe fn value_ref(&self) -> &V {
        debug_assert!(!self.is_vacant());
        unsafe { self.value.assume_init_ref() }
    }

    #[inline(always)]
    unsafe fn value_mut(&mut self) -> &mut V {
        debug_assert!(!self.is_vacant());
        unsafe { self.value.assume_init_mut() }
    }
}

impl<K: Key, V: Value> Clone for Entry<K, V> {
    fn clone(&self) -> Self {
        if self.is_vacant() {
            Self::vacant()
        } else {
            Self::occupied(self.hash, unsafe { self.key_ref().clone() }, unsafe {
                self.value_ref().clone()
            })
        }
    }
}

impl<K: Key, V: Value> Drop for Entry<K, V> {
    fn drop(&mut self) {
        if !self.is_vacant() {
            unsafe {
                self.key.assume_init_drop();
                self.value.assume_init_drop();
            }
        }
    }
}

impl<K: Key + core::fmt::Debug, V: Value + core::fmt::Debug> core::fmt::Debug for Entry<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_vacant() {
            f.write_str("Vacant")
        } else {
            f.debug_struct("Occupied")
                .field("hash", &self.hash)
                .field("key", unsafe { self.key_ref() })
                .field("value", unsafe { self.value_ref() })
                .finish()
        }
    }
}

#[derive(Clone)]
pub struct PoMap<K: Key, V: Value, H: Hasher + Default = DefaultHasher> {
    len: usize,
    meta: PoMapMeta,
    slots: Vec<Entry<K, V>>,
    _phantom: PhantomData<H>,
}

impl<K: Key, V: Value, H: Hasher + Default> PoMap<K, V, H> {
    /// Create a new [`PoMap`] with _at least_ the given capacity.
    ///
    /// Note that the actual internal capacity will always be scaled up to the next power of
    /// two (if not already a power of two) plus [`MAX_SCAN`].
    ///
    /// Also note that the minimum capacity is [`MIN_CAPACITY`].
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let (meta, vec_capacity) = PoMapMeta::new(capacity);
        let slots = vec![Entry::vacant(); vec_capacity];
        Self {
            len: 0,
            meta,
            slots,
            _phantom: PhantomData,
        }
    }

    /// Create a new [`PoMap`] with [`MIN_CAPACITY`] + [`MAX_SCAN`] internal capacity.
    #[inline(always)]
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Current number of occupied entries.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map contains no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Hot path for `get` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline]
    pub fn get_with_hash(&self, hash: u64, key: &K) -> Option<&V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + MAX_SCAN;

        verbose_log!();
        verbose_log!(
            "get_with_hash: hash={} ideal_slot={} scan_end={}",
            hash,
            ideal_slot,
            scan_end
        );

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing Vec.
        let slots_ptr = self.slots.as_ptr();

        for idx in ideal_slot..scan_end {
            let slot = unsafe { &*slots_ptr.add(idx) };
            let slot_hash = slot.hash;
            let _vacant = slot_hash == VACANT_HASH;
            verbose_log!(
                "get_with_hash: idx={} slot_hash={} vacant={}",
                idx,
                slot_hash,
                _vacant
            );

            if slot_hash < hash {
                continue;
            }

            // we guarantee the slots are sorted by hash
            if slot_hash > hash {
                verbose_log!(
                    "get_with_hash: stop at idx={}, slot_hash {} > hash {}",
                    idx,
                    slot_hash,
                    hash
                );
                return None;
            }

            // SAFETY: we guard with VACANT_HASH above.
            if unsafe { slot.key_ref() } == key {
                // SAFETY: slot is occupied.
                verbose_log!("get_with_hash: found at idx={}", idx);
                return Some(unsafe { slot.value_ref() });
            }
            verbose_log!("get_with_hash: hash match but key mismatch at idx={}", idx);
        }
        verbose_log!("get_with_hash: not found in window");
        None
    }

    /// Gets a reference to the value corresponding to the specified key, or `None` if not found.
    ///
    /// Calls [`Self::get_with_hash`] internally after computing the hash.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self.get_with_hash(hash, key)
    }

    /// Inserts a key-value pair into the map, replacing any existing value.
    ///
    /// Computes the hash of the key and then delegates to [`Self::insert_with_hash`].
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self.insert_with_hash(hash, key, value)
    }

    #[inline]
    pub fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        loop {
            let ideal_slot = self.meta.ideal_slot(hash);
            let scan_end = ideal_slot + MAX_SCAN;
            verbose_log!();
            verbose_log!(
                "insert_with_hash: hash={} ideal_slot={} scan_end={}",
                hash,
                ideal_slot,
                scan_end
            );

            // SAFETY: ideal_slot..scan_end is always in-bounds for the backing Vec.
            let slots_ptr = self.slots.as_mut_ptr();

            let mut idx = ideal_slot;
            while idx < scan_end {
                let slot = unsafe { &mut *slots_ptr.add(idx) };
                let slot_hash = slot.hash;
                let _vacant = slot_hash == VACANT_HASH;
                verbose_log!(
                    "insert_with_hash: idx={} slot_hash={} vacant={}",
                    idx,
                    slot_hash,
                    _vacant
                );

                if slot_hash == VACANT_HASH {
                    verbose_log!("insert_with_hash: inserting into vacant idx={}", idx);
                    slot.occupy(hash, key, value);
                    self.len += 1;
                    return None;
                }

                // Common case: existing hash < new hash → keep scanning forward.
                if slot_hash < hash {
                    idx += 1;
                    continue;
                }

                // Existing hash == new hash → maybe update in place.
                if slot_hash == hash && unsafe { slot.key_ref() } == &key {
                    verbose_log!("insert_with_hash: update in place at idx={}", idx);
                    let old_value = core::mem::replace(unsafe { slot.value_mut() }, value);
                    return Some(old_value);
                }

                // Here: slot_hash > hash  OR (== but different key).
                // We need to insert before this slot; attempt in-window shift.
                // Search for a Vacant in (idx, scan_end).
                let mut search = idx + 1;
                while search < scan_end {
                    let candidate = unsafe { &*slots_ptr.add(search) };
                    let cand_hash = candidate.hash;
                    let cand_vacant = cand_hash == VACANT_HASH;
                    verbose_log!(
                        "insert_with_hash: search idx={} slot_hash={} vacant={}",
                        search,
                        cand_hash,
                        cand_vacant
                    );
                    if cand_vacant {
                        verbose_log!(
                            "insert_with_hash: shift window idx={}..={} (len {})",
                            idx,
                            search,
                            search - idx + 1
                        );
                        // Rotate [idx..=search] right by 1 and drop new element at idx.
                        let len = search - idx + 1;
                        let slice = unsafe { slice::from_raw_parts_mut(slots_ptr.add(idx), len) };
                        slice.rotate_right(1);
                        slice[0] = Entry::occupied(hash, key, value);
                        self.len += 1;
                        return None;
                    }
                    search += 1;
                }

                verbose_log!(
                    "insert_with_hash: no room in window [{}, {}), growing",
                    ideal_slot,
                    scan_end
                );
                // No room in this window; grow and retry.
                break;
            }

            // If we make it here, we either ran out of room in the scan window
            // or the table is saturated for this hash window. Grow and retry.
            // let old_cap = self.slots.capacity();
            // let load_factor = self.len as f64 / old_cap as f64;
            self.reserve(self.slots.capacity() * GROWTH_FACTOR);
            // let new_cap = self.slots.capacity();
            // let new_load_factor = self.len as f64 / new_cap as f64;
            /*println!(
                "PoMap resized from {} to {} (load factor {:.2}), {:.2} after",
                old_cap, new_cap, load_factor, new_load_factor
            );*/
        }
    }

    /// Reserve capacity for at least `requested_capacity` elements if there is not enough
    /// capacity already.
    ///
    /// Returns the new capacity.
    #[cold]
    pub fn reserve(&mut self, requested_capacity: usize) -> usize {
        // generate new meta
        let current_capacity = self.slots.capacity();
        let (new_meta, new_vec_capacity) = PoMapMeta::new(requested_capacity);
        if new_vec_capacity <= current_capacity {
            return current_capacity;
        }

        // allocate new vec
        let new_slots: Vec<Entry<K, V>> = vec![Entry::vacant(); new_vec_capacity];

        // re-insert all existing elements into the new vec in the same order, with spaces
        // added based on hash prefix / the new meta
        let old_slots = core::mem::replace(&mut self.slots, new_slots);
        let mut cursor = 0;
        for slot in old_slots.into_iter() {
            if slot.is_vacant() {
                cursor += 1;
                continue;
            }
            let hash = slot.hash;

            // calculate ideal slot in the new layout
            let ideal_slot = new_meta.ideal_slot(hash);

            // advance cursor to at least the ideal slot
            cursor = ideal_slot.max(cursor);

            // insert the slot, we should be at or past the ideal slot now. Because the
            // previous layout was already valid and is strictly smaller than the new one, this
            // can never cause us to exceed MAX_SCAN slots past the ideal slot because we are
            // always gaining more room.
            self.slots[cursor] = slot;

            // advance cursor
            cursor += 1;
        }

        // apply the new meta
        self.meta = new_meta;

        new_vec_capacity
    }
}

impl Default for PoMap<(), ()> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// PoMAP meta for a flat, sorted-by-hash array with direct prefix indexing.
///
/// We pick an internal `ideal_range` that is a power of two, and guarantee:
///
///   ideal_slot(hash) ∈ [0, ideal_range)
///   and you can safely scan `[ideal_slot, ideal_slot + MAX_SCAN)`
///   inside a Vec of length `capacity = ideal_range + MAX_SCAN`.
///
/// The caller is responsible for using the returned `capacity` to size the backing Vec.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PoMapMeta {
    /// Number of index bits m where `ideal_range = 1 << m`.
    index_bits: usize,

    /// Shift to extract the top-m bits from the 64-bit hash:
    ///   ideal_slot = (hash >> index_shift)
    index_shift: usize,
}

impl PoMapMeta {
    /// Build meta from a *requested* logical capacity (or element count).
    ///
    /// We choose:
    ///   base        = max(requested, MIN_CAPACITY)
    ///   ideal_range = base.next_power_of_two()
    ///
    /// This means:
    ///   - `ideal_range` is a power of two
    ///   - full Vec capacity should be `ideal_range + MAX_SCAN`
    ///
    /// There are no panics; we just round up.
    ///
    /// Returns `(meta, capacity)`, where `capacity` is what you should use
    /// for the backing `Vec<Entry<..>>`.
    #[inline(always)]
    const fn new(requested: usize) -> (Self, usize) {
        let requested = requested.saturating_sub(MAX_SCAN);
        let base = if requested > MIN_CAPACITY {
            requested
        } else {
            MIN_CAPACITY
        };
        let ideal_range = base.next_power_of_two();

        let index_bits: usize = ideal_range.trailing_zeros() as usize; // m
        debug_assert!(index_bits <= HASH_BITS);
        let index_shift: usize = HASH_BITS - index_bits; // use top-m bits of the u64 hash

        (
            Self {
                index_bits,
                index_shift,
            },
            ideal_range + MAX_SCAN,
        )
    }

    /// **Ideal slot** for this hash: top-m bits of the 64-bit hash.
    ///
    /// Because `ideal_range = 2^m` and we take exactly `m` bits, the result
    /// is guaranteed `< ideal_range`. No masking or wrap needed; `MAX_SCAN`
    /// is handled by the caller.
    #[inline(always)]
    const fn ideal_slot(&self, hash: u64) -> usize {
        (hash >> self.index_shift) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::collections::HashMap;

    #[derive(Default)]
    struct IdentityHasher(u64);

    impl Hasher for IdentityHasher {
        fn write(&mut self, bytes: &[u8]) {
            let mut acc = 0u64;
            for &b in bytes {
                acc = acc.wrapping_mul(31).wrapping_add(b as u64);
            }
            self.0 = acc;
        }

        fn write_u8(&mut self, i: u8) {
            self.write(&[i]);
        }

        fn write_u16(&mut self, i: u16) {
            self.write(&i.to_le_bytes());
        }

        fn write_u32(&mut self, i: u32) {
            self.write(&i.to_le_bytes());
        }

        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }

        fn write_u128(&mut self, i: u128) {
            self.write(&i.to_le_bytes());
        }

        fn write_usize(&mut self, i: usize) {
            self.write_u64(i as u64);
        }

        fn finish(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn test_resize_goes_up_by_power_of_two_and_never_down() {
        let mut map: PoMap<u64, u64> = PoMap::new();
        assert_eq!(map.slots.capacity(), MIN_CAPACITY + MAX_SCAN);

        // reserve more than the current capacity
        let new_capacity = map.reserve(100);
        assert_eq!(map.slots.capacity(), new_capacity);
        assert_eq!(new_capacity, 128 + MAX_SCAN); // 128 is the next power of two after 100

        // reserve less than the current capacity
        let old_capacity = map.slots.capacity();
        let new_capacity = map.reserve(50);
        assert_eq!(new_capacity, old_capacity);
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut map: PoMap<u64, u64, IdentityHasher> = PoMap::new();

        // Spread keys across distinct ideal slots to avoid forced rehashing during the test.
        for i in 0..8u64 {
            let key = (i << 60) | i;
            assert_eq!(map.insert(key, i * 10), None);
            assert_eq!(map.get(&key), Some(&(i * 10)));
        }

        assert_eq!(map.get(&99), None);
        assert_eq!(map.len(), 8);
        assert!(!map.is_empty());
    }

    #[test]
    fn insert_replaces_existing_value() {
        let mut map: PoMap<u64, u64, IdentityHasher> = PoMap::new();

        assert_eq!(map.insert(42, 1), None);
        assert_eq!(map.insert(42, 2), Some(1));
        assert_eq!(map.get(&42), Some(&2));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn insertion_shifts_to_keep_hash_order() {
        let mut map: PoMap<u64, u64, IdentityHasher> = PoMap::new();

        let base = 0b01010u64 << 59;
        let higher = base | 30;
        let lower = base | 10;

        let slot = map.meta.ideal_slot(encode_hash(lower));
        assert_eq!(slot, map.meta.ideal_slot(encode_hash(higher)));

        assert_eq!(map.insert(higher, 1), None);
        assert_eq!(map.insert(lower, 2), None);

        let first = &map.slots[slot];
        let second = &map.slots[slot + 1];

        assert!(!first.is_vacant());
        assert!(!second.is_vacant());

        // Check hash order with encoded hashes
        let lower_h = encode_hash(lower);
        let higher_h = encode_hash(higher);

        assert_eq!(first.hash, lower_h);
        assert_eq!(second.hash, higher_h);

        // And make sure key/value order matches expectation
        assert_eq!(
            unsafe { (*first.key_ref(), *first.value_ref()) },
            (lower, 2),
        );
        assert_eq!(
            unsafe { (*second.key_ref(), *second.value_ref()) },
            (higher, 1),
        );

        assert_eq!(map.get(&higher), Some(&1));
        assert_eq!(map.get(&lower), Some(&2));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn mass_inserts_and_gets() {
        let total = 1_000_000usize;
        let mut map: PoMap<u64, u64, IdentityHasher> = PoMap::new();
        let mut expected: HashMap<u64, u64> = HashMap::new();

        // Fixed seed for determinism while still using fully random values to stress layout.
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        for _ in 0..total {
            let key_val: u64 = rng.random();
            let val: u64 = rng.random();

            let prev_expected = expected.insert(key_val, val);
            let prev_actual = map.insert(key_val, val);
            assert_eq!(prev_actual, prev_expected);
        }

        for (key, val) in expected.iter() {
            assert_eq!(map.get(key), Some(val));
        }
        assert_eq!(map.len(), expected.len());
        assert!(!map.is_empty());
    }
}
