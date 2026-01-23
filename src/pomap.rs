use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    slice,
};
use std::{
    alloc::{Layout, alloc, dealloc, handle_alloc_error},
    hash::DefaultHasher,
};

/// Minimum capacity we will allow for PoMap
const MIN_CAPACITY: usize = 4;

/// Number of bits in the hashcode
const HASH_BITS: usize = 64; // we use a 64-bit hashcode

const GROWTH_FACTOR: usize = 4;
const VACANT_HASH: u64 = u64::MAX;

#[inline(always)]
const fn max_scan_for_capacity(capacity: usize) -> usize {
    // log2 rounded down; capacity is always >= MIN_CAPACITY (non-zero).
    capacity.next_power_of_two().trailing_zeros() as usize
}

pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

pub trait Value: Clone {}
impl<V: Clone> Value for V {}

struct Entry<K: Key, V: Value> {
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
}

#[inline(always)]
fn encode_hash(h: u64) -> u64 {
    h.saturating_sub(1)
    //f h == VACANT_HASH { VACANT_HASH - 1 } else { h }
}

impl<K: Key, V: Value> Entry<K, V> {
    #[inline(always)]
    fn occupy(&mut self, key: K, value: V) {
        self.key.write(key);
        self.value.write(value);
    }

    #[inline(always)]
    unsafe fn key_ref(&self) -> &K {
        unsafe { self.key.assume_init_ref() }
    }

    #[inline(always)]
    unsafe fn value_ref(&self) -> &V {
        unsafe { self.value.assume_init_ref() }
    }

    #[inline(always)]
    unsafe fn value_mut(&mut self) -> &mut V {
        unsafe { self.value.assume_init_mut() }
    }

    #[inline(always)]
    unsafe fn drop_contents(&mut self) {
        unsafe {
            self.key.assume_init_drop();
            self.value.assume_init_drop();
        }
    }
}

struct Slots<K: Key, V: Value> {
    ptr: NonNull<u8>,
    capacity: usize,
    hashes: *mut u64,
    entries: *mut Entry<K, V>,
    layout: Layout,
    _marker: PhantomData<Entry<K, V>>,
}

impl<K: Key, V: Value> Slots<K, V> {
    #[inline(always)]
    fn new(capacity: usize) -> Self {
        let hashes_layout =
            Layout::array::<u64>(capacity).expect("PoMap hash layout overflow on creation");
        let entries_layout = Layout::array::<Entry<K, V>>(capacity)
            .expect("PoMap entry layout overflow on creation");
        let (layout, entries_offset) = hashes_layout
            .extend(entries_layout)
            .expect("PoMap combined layout overflow on creation");
        let layout = layout.pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };

        let hashes = ptr.as_ptr() as *mut u64;
        let entries = unsafe { ptr.as_ptr().add(entries_offset) as *mut Entry<K, V> };

        unsafe {
            // Mark all hashes as vacant.
            ptr::write_bytes(hashes, 0xFF, capacity);
            // SAFETY: ok to leave entries uninitialized
        }

        Self {
            ptr,
            hashes,
            entries,
            capacity,
            layout,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    const fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    fn hashes_ptr(&self) -> *mut u64 {
        self.hashes
    }

    #[inline(always)]
    fn entries_ptr(&self) -> *mut Entry<K, V> {
        self.entries
    }

    #[inline(always)]
    unsafe fn set_hash(&mut self, idx: usize, hash: u64) {
        debug_assert!(idx < self.capacity);
        unsafe { *self.hashes.add(idx) = hash };
    }

    #[inline(always)]
    unsafe fn clear_slot(&mut self, idx: usize) {
        unsafe { self.set_hash(idx, VACANT_HASH) };
    }

    #[inline(always)]
    unsafe fn write_slot(&mut self, idx: usize, hash: u64, key: K, value: V) {
        unsafe {
            self.set_hash(idx, hash);
            let entry = &mut *self.entries.add(idx);
            entry.occupy(key, value);
        }
    }
}

impl<K: Key, V: Value> Drop for Slots<K, V> {
    fn drop(&mut self) {
        unsafe {
            let hashes = slice::from_raw_parts(self.hashes, self.capacity);
            for (idx, &hash) in hashes.iter().enumerate() {
                if hash != VACANT_HASH {
                    let entry = &mut *self.entries.add(idx);
                    entry.drop_contents();
                }
            }
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

pub struct PoMap<K: Key, V: Value, H: Hasher + Default = DefaultHasher> {
    len: usize,
    meta: PoMapMeta,
    slots: Slots<K, V>,
    _phantom: PhantomData<H>,
}

impl<K: Key, V: Value, H: Hasher + Default> PoMap<K, V, H> {
    /// Create a new [`PoMap`] with _at least_ the given capacity.
    ///
    /// Note that the actual internal capacity will always be scaled up to the next power of
    /// two (if not already a power of two) plus `max_scan_for_capacity(ideal_range)`.
    ///
    /// Also note that the minimum capacity is [`MIN_CAPACITY`].
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let (meta, vec_capacity) = PoMapMeta::new(capacity);
        let slots = Slots::new(vec_capacity);
        Self {
            len: 0,
            meta,
            slots,
            _phantom: PhantomData,
        }
    }

    /// Create a new [`PoMap`] with [`MIN_CAPACITY`] + max-scan internal capacity.
    #[inline(always)]
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    #[inline(always)]
    pub const fn max_scan(&self) -> usize {
        self.meta.max_scan
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
    #[inline(always)]
    pub fn get_with_hash(&self, hash: u64, key: &K) -> Option<&V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.max_scan;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in ideal_slot..scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &*entries_ptr.add(idx) };
                if unsafe { slot_entry.key_ref() } == key {
                    return Some(unsafe { slot_entry.value_ref() });
                }
                // hash collision: keep scanning
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }
        }
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

    #[inline(always)]
    pub fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        loop {
            let ideal_slot = self.meta.ideal_slot(hash);
            let scan_end = ideal_slot + self.meta.max_scan;

            // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
            let hashes_ptr = self.slots.hashes_ptr();
            let entries_ptr = self.slots.entries_ptr();

            let mut idx = ideal_slot;
            while idx < scan_end {
                let slot_hash = unsafe { *hashes_ptr.add(idx) };

                // Fast path: insert immediately into an empty slot.
                if slot_hash == VACANT_HASH {
                    unsafe {
                        *hashes_ptr.add(idx) = hash;
                        (*entries_ptr.add(idx)).occupy(key, value);
                    }
                    self.len += 1;
                    return None;
                }

                // Common scan: skip hashes below the insertion point.
                if slot_hash < hash {
                    idx += 1;
                    continue;
                }

                // Now slot_hash >= hash and occupied.
                if slot_hash == hash {
                    let slot = unsafe { &mut *entries_ptr.add(idx) };
                    let slot_key = unsafe { slot.key_ref() };
                    if slot_key == &key {
                        // Key already exists; replace value.
                        let old_value = if core::mem::needs_drop::<V>() {
                            core::mem::replace(unsafe { slot.value_mut() }, value)
                        } else {
                            // For POD values, avoid drop glue by doing a raw read+write.
                            unsafe {
                                let old = slot.value.assume_init_read();
                                slot.value.write(value);
                                old
                            }
                        };
                        return Some(old_value);
                    }
                    if slot_key < &key {
                        idx += 1;
                        continue;
                    }
                    // slot_key > key → fall through to shift
                }

                // slot_hash > hash or slot_key > key → fall through to shift

                // Here: (hash, key) should be inserted before this slot.
                // Search for a Vacant in (idx, scan_end).
                let mut search = idx + 1;
                while search < scan_end {
                    let cand_hash = unsafe { *hashes_ptr.add(search) };
                    if cand_hash == VACANT_HASH {
                        // Rotate [idx..=search] right by 1 and drop new element at idx.
                        unsafe {
                            // Shift [idx..search) right by 1 into the vacant at `search`.
                            // Do it backwards to avoid overlap issues.
                            let mut i = search;
                            while i > idx {
                                *hashes_ptr.add(i) = *hashes_ptr.add(i - 1);
                                core::ptr::write(
                                    entries_ptr.add(i),
                                    core::ptr::read(entries_ptr.add(i - 1)),
                                );
                                i -= 1;
                            }

                            *hashes_ptr.add(idx) = hash;
                            (*entries_ptr.add(idx)).occupy(key, value);
                        }
                        self.len += 1;
                        return None;
                    }
                    search += 1;
                }

                // No room in this window; grow and retry.
                break;
            }

            // If we make it here, we either ran out of room in the scan window
            // or the table is saturated for this hash window. Grow and retry.
            #[cfg(feature = "resize-logging")]
            let old_cap = self.slots.capacity();
            #[cfg(feature = "resize-logging")]
            let load_factor = self.len as f64 / old_cap as f64;
            self.reserve((self.slots.capacity() - self.meta.max_scan) * GROWTH_FACTOR);
            #[cfg(feature = "resize-logging")]
            let new_cap = self.slots.capacity();
            #[cfg(feature = "resize-logging")]
            let new_load_factor = self.len as f64 / new_cap as f64;
            #[cfg(feature = "resize-logging")]
            println!(
                "PoMap resized from {} to {} (load factor {:.2}), {:.2} after",
                old_cap, new_cap, load_factor, new_load_factor
            );
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
            /*println!(
                "PoMap::reserve requested={} resulting_capacity={}",
                requested_capacity, current_capacity
            );*/
            return current_capacity;
        }

        // allocate new slots and keep the old ones for re-distribution.
        let mut old_slots = core::mem::replace(&mut self.slots, Slots::new(new_vec_capacity));

        // re-insert all existing elements into the new vec in the same order, with spaces
        // added based on hash prefix / the new meta
        let mut cursor = 0;
        let hashes_ptr = old_slots.hashes_ptr();
        let entries_ptr = old_slots.entries_ptr();
        for idx in 0..current_capacity {
            let hash = unsafe { *hashes_ptr.add(idx) };
            if hash == VACANT_HASH {
                cursor += 1;
                continue;
            }

            let ideal_slot = new_meta.ideal_slot(hash);
            cursor = ideal_slot.max(cursor);

            let entry = unsafe { &mut *entries_ptr.add(idx) };
            let key = unsafe { entry.key.assume_init_read() };
            let value = unsafe { entry.value.assume_init_read() };

            // Prevent the old slot from dropping the moved value on destruction.
            unsafe { old_slots.clear_slot(idx) };

            // insert the slot, we should be at or past the ideal slot now. Because the
            // previous layout was already valid and is strictly smaller than the new one, this
            // can never cause us to exceed the scan limit past the ideal slot because we are
            // always gaining more room.
            unsafe { self.slots.write_slot(cursor, hash, key, value) };

            // advance cursor
            cursor += 1;
        }

        // apply the new meta
        self.meta = new_meta;

        /*println!(
            "PoMap::reserve requested={} resulting_capacity={}",
            requested_capacity, new_vec_capacity
        );*/

        new_vec_capacity
    }
}

impl<K: Key, V: Value, H: Hasher + Default> Clone for PoMap<K, V, H> {
    fn clone(&self) -> Self {
        let capacity = self.capacity();
        let mut slots = Slots::new(capacity);

        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in 0..capacity {
            let hash = unsafe { *hashes_ptr.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            let entry = unsafe { &*entries_ptr.add(idx) };
            let key = unsafe { entry.key_ref().clone() };
            let value = unsafe { entry.value_ref().clone() };
            unsafe { slots.write_slot(idx, hash, key, value) };
        }

        Self {
            len: self.len,
            meta: self.meta,
            slots,
            _phantom: PhantomData,
        }
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
///   and you can safely scan `[ideal_slot, ideal_slot + max_scan)`
///   inside backing storage of length `capacity = ideal_range + max_scan`.
///
/// The caller is responsible for using the returned `capacity` to size the backing allocation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PoMapMeta {
    /// Number of index bits m where `ideal_range = 1 << m`.
    index_bits: usize,

    /// Shift to extract the top-m bits from the 64-bit hash:
    ///   ideal_slot = (hash >> index_shift)
    index_shift: usize,

    /// Maximum scan distance for this capacity.
    max_scan: usize,
}

impl PoMapMeta {
    /// Build meta from a *requested* logical capacity (or element count).
    ///
    /// We choose:
    ///   base        = max(requested, MIN_CAPACITY)
    ///   ideal_range = base.next_power_of_two()
    ///   max_scan    = max_scan_for_capacity(ideal_range)
    ///
    /// This means:
    ///   - `ideal_range` is a power of two
    ///   - full Vec capacity should be `ideal_range + max_scan`
    ///
    /// There are no panics; we just round up.
    ///
    /// Returns `(meta, capacity)`, where `capacity` is what you should use
    /// for the backing allocation.
    #[inline(always)]
    const fn new(requested: usize) -> (Self, usize) {
        let base = if requested > MIN_CAPACITY {
            requested
        } else {
            MIN_CAPACITY
        };
        let ideal_range = base.next_power_of_two();
        let max_scan = max_scan_for_capacity(ideal_range);

        let index_bits: usize = ideal_range.trailing_zeros() as usize; // m
        debug_assert!(index_bits <= HASH_BITS);
        let index_shift: usize = HASH_BITS - index_bits; // use top-m bits of the u64 hash

        (
            Self {
                index_bits,
                index_shift,
                max_scan,
            },
            ideal_range + max_scan,
        )
    }

    /// **Ideal slot** for this hash: top-m bits of the 64-bit hash.
    ///
    /// Because `ideal_range = 2^m` and we take exactly `m` bits, the result
    /// is guaranteed `< ideal_range`. No masking or wrap needed; scan length
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
    use std::{cell::Cell, collections::HashMap, rc::Rc};

    #[derive(Clone)]
    struct DropCounter {
        drops: Rc<Cell<usize>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

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

        let hashes_ptr = map.slots.hashes_ptr();
        let entries_ptr = map.slots.entries_ptr();
        let first_hash = unsafe { *hashes_ptr.add(slot) };
        let second_hash = unsafe { *hashes_ptr.add(slot + 1) };
        let first_entry = unsafe { &*entries_ptr.add(slot) };
        let second_entry = unsafe { &*entries_ptr.add(slot + 1) };

        assert_ne!(first_hash, VACANT_HASH);
        assert_ne!(second_hash, VACANT_HASH);

        // Check hash order with encoded hashes
        let lower_h = encode_hash(lower);
        let higher_h = encode_hash(higher);

        assert_eq!(first_hash, lower_h);
        assert_eq!(second_hash, higher_h);

        // And make sure key/value order matches expectation
        assert_eq!(
            unsafe { (*first_entry.key_ref(), *first_entry.value_ref(),) },
            (lower, 2),
        );
        assert_eq!(
            unsafe { (*second_entry.key_ref(), *second_entry.value_ref(),) },
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

    #[test]
    fn overwrite_drops_old_value() {
        let drops = Rc::new(Cell::new(0));
        let mut map: PoMap<u64, DropCounter, IdentityHasher> = PoMap::new();

        assert!(
            map.insert(
                1,
                DropCounter {
                    drops: drops.clone()
                }
            )
            .is_none()
        );
        let previous = map
            .insert(
                1,
                DropCounter {
                    drops: drops.clone(),
                },
            )
            .expect("expected to replace an existing value");

        assert_eq!(drops.get(), 0);
        drop(previous);
        assert_eq!(drops.get(), 1);

        drop(map);
        assert_eq!(drops.get(), 2);
    }
}
