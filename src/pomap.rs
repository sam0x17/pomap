use ahash::AHasher;
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::Index,
    ptr::{self, NonNull},
    slice,
};

/// Minimum capacity we will allow for PoMap
const MIN_CAPACITY: usize = 4;

/// Number of bits in the hashcode
const HASH_BITS: usize = 64; // we use a 64-bit hashcode

const GROWTH_FACTOR: usize = 4;
const VACANT_HASH: u64 = u64::MAX;

/// Marker trait for keys stored in a [`PoMap`].
pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

/// Marker trait for values stored in a [`PoMap`].
pub trait Value: Clone {}
impl<V: Clone> Value for V {}

struct SlotEntry<K: Key, V: Value> {
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
}

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    // Ensure we never produce VACANT_HASH as a valid hash by shifting down by 1. This will
    // cause an astronomically tiny bias in the hash distribution with 0 colliding with 1, but
    // this is perfectly acceptable for our purposes
    h.saturating_sub(1)
}

struct Slots<K: Key, V: Value> {
    ptr: NonNull<u8>,
    capacity: usize,
    hashes: *mut u64,
    entries: *mut SlotEntry<K, V>,
    layout: Layout,
    _marker: PhantomData<SlotEntry<K, V>>,
}

impl<K: Key, V: Value> Slots<K, V> {
    #[inline(always)]
    fn new(capacity: usize) -> Self {
        let hashes_layout =
            Layout::array::<u64>(capacity).expect("PoMap hash layout overflow on creation");
        let entries_layout = Layout::array::<SlotEntry<K, V>>(capacity)
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
        let entries = unsafe { ptr.as_ptr().add(entries_offset) as *mut SlotEntry<K, V> };

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
    fn entries_ptr(&self) -> *mut SlotEntry<K, V> {
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
            entry.key.write(key);
            entry.value.write(value);
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
                    entry.key.assume_init_drop();
                    entry.value.assume_init_drop();
                }
            }
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Prefix-ordered hash map with a fixed max-scan window.
pub struct PoMap<K: Key, V: Value, H: Hasher + Default = AHasher> {
    len: usize,
    meta: PoMapMeta,
    slots: Slots<K, V>,
    _phantom: PhantomData<H>,
}

impl<K: Key, V: Value, H: Hasher + Default> PoMap<K, V, H> {
    #[inline(always)]
    fn pack_from_slots(
        dst: &mut Slots<K, V>,
        meta: &PoMapMeta,
        cursor: &mut usize,
        src: &mut Slots<K, V>,
        start_idx: usize,
    ) -> Result<(), usize> {
        let dst_hashes = dst.hashes_ptr();
        let dst_entries = dst.entries_ptr();
        let src_hashes = src.hashes_ptr();
        let src_entries = src.entries_ptr();
        let src_capacity = src.capacity();
        let max_scan = meta.index_bits;
        let index_shift = meta.index_shift;

        let mut idx = start_idx;
        while idx < src_capacity {
            let hash = unsafe { *src_hashes.add(idx) };
            if hash != VACANT_HASH {
                let ideal_slot = (hash >> index_shift) as usize;
                if *cursor < ideal_slot {
                    *cursor = ideal_slot;
                }
                if *cursor >= ideal_slot + max_scan {
                    return Err(idx);
                }
                unsafe {
                    let entry = &mut *src_entries.add(idx);
                    let key = entry.key.assume_init_read();
                    let value = entry.value.assume_init_read();
                    *src_hashes.add(idx) = VACANT_HASH;

                    *dst_hashes.add(*cursor) = hash;
                    let dst_entry = &mut *dst_entries.add(*cursor);
                    dst_entry.key.write(key);
                    dst_entry.value.write(value);
                }
                *cursor += 1;
            }
            idx += 1;
        }

        Ok(())
    }

    /// Create a new [`PoMap`] with _at least_ the given capacity.
    ///
    /// Note that the actual internal capacity will always be scaled up to the next power of
    /// two (if not already a power of two) plus `index_bits` (the max-scan window).
    ///
    /// Also note that the minimum capacity is [`MIN_CAPACITY`].
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let (meta, vec_capacity) = PoMapMeta::new(capacity.next_power_of_two());
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
    /// Returns the total backing capacity (including the max-scan window).
    pub const fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    #[inline(always)]
    /// Returns the max-scan window length for the current capacity.
    pub const fn max_scan(&self) -> usize {
        self.meta.index_bits
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

    /// Gets a reference to the value corresponding to the specified key, or `None` if not found.
    ///
    /// Calls [`Self::get_with_hash`] internally after computing the hash.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._get_with_hash(hash, key)
    }

    #[inline]
    /// Gets a reference to the value corresponding to the specified key using a precomputed hash.
    pub fn get_with_hash(&self, hash: u64, key: &K) -> Option<&V> {
        self._get_with_hash(encode_hash(hash), key)
    }

    /// Returns `true` if the map contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._get_key_value_with_hash(hash, key)
    }

    /// Hot path for `get` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline(always)]
    pub fn _get_with_hash(&self, hash: u64, key: &K) -> Option<&V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in ideal_slot..scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &*entries_ptr.add(idx) };
                if unsafe { slot_entry.key.assume_init_ref() } == key {
                    return Some(unsafe { slot_entry.value.assume_init_ref() });
                }
                // hash collision: keep scanning
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }
        }
        None
    }

    #[inline(always)]
    fn _get_key_value_with_hash(&self, hash: u64, key: &K) -> Option<(&K, &V)> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in ideal_slot..scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &*entries_ptr.add(idx) };
                let slot_key = unsafe { slot_entry.key.assume_init_ref() };
                if slot_key == key {
                    return Some((slot_key, unsafe { slot_entry.value.assume_init_ref() }));
                }
                // hash collision: keep scanning
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }
        }
        None
    }

    #[inline(always)]
    fn _find_index_by_hash<F>(&self, hash: u64, mut is_match: F) -> Option<usize>
    where
        F: FnMut(&K) -> bool,
    {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in ideal_slot..scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };
            if slot_hash == hash {
                let entry = unsafe { &*entries_ptr.add(idx) };
                let key = unsafe { entry.key.assume_init_ref() };
                if is_match(key) {
                    return Some(idx);
                }
            } else if slot_hash > hash {
                return None;
            }
        }
        None
    }

    /// Gets a mutable reference to the value corresponding to the specified key, or `None` if not found.
    ///
    /// Calls [`Self::get_mut_with_hash`] internally after computing the hash.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._get_mut_with_hash(hash, key)
    }

    #[inline]
    /// Gets a mutable reference to the value corresponding to the specified key using a precomputed hash.
    pub fn get_mut_with_hash(&mut self, hash: u64, key: &K) -> Option<&mut V> {
        self._get_mut_with_hash(encode_hash(hash), key)
    }

    /// Hot path for `get_mut` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline(always)]
    fn _get_mut_with_hash(&mut self, hash: u64, key: &K) -> Option<&mut V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        for idx in ideal_slot..scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &mut *entries_ptr.add(idx) };
                if unsafe { slot_entry.key.assume_init_ref() } == key {
                    return Some(unsafe { slot_entry.value.assume_init_mut() });
                }
                // hash collision: keep scanning
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }
        }
        None
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, H> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._entry_with_hash(hash, key)
    }

    #[inline]
    /// Gets the entry for a key using a precomputed hash.
    pub fn entry_with_hash(&mut self, hash: u64, key: K) -> Entry<'_, K, V, H> {
        self._entry_with_hash(encode_hash(hash), key)
    }

    /// Creates a raw mutable entry builder for advanced lookups.
    #[inline]
    pub fn raw_entry_mut(&mut self) -> RawEntryBuilderMut<'_, K, V, H> {
        RawEntryBuilderMut { map: self }
    }

    /// Creates a raw immutable entry builder for advanced lookups.
    #[inline]
    pub fn raw_entry(&self) -> RawEntryBuilder<'_, K, V, H> {
        RawEntryBuilder { map: self }
    }

    /// Returns an iterator over key-value pairs in deterministic order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            hashes: self.slots.hashes_ptr().cast_const(),
            entries: self.slots.entries_ptr().cast_const(),
            index: 0,
            capacity: self.slots.capacity(),
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over key-value pairs in deterministic order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            hashes: self.slots.hashes_ptr().cast_const(),
            entries: self.slots.entries_ptr(),
            index: 0,
            capacity: self.slots.capacity(),
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over all keys in deterministic order.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator over all values in deterministic order.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { iter: self.iter() }
    }

    /// Returns a mutable iterator over all values in deterministic order.
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }

    /// Converts the map into an iterator over all keys in deterministic order.
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys {
            iter: self.into_iter(),
        }
    }

    /// Converts the map into an iterator over all values in deterministic order.
    #[inline]
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues {
            iter: self.into_iter(),
        }
    }

    #[inline(always)]
    fn find_entry_location(&self, hash: u64, key: &K) -> EntryLocation {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        let mut idx = ideal_slot;
        while idx < scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == VACANT_HASH {
                return EntryLocation::Vacant {
                    insert: VacantInsert::Direct { index: idx },
                };
            }

            if slot_hash < hash {
                idx += 1;
                continue;
            }

            if slot_hash == hash {
                let slot_entry = unsafe { &*entries_ptr.add(idx) };
                let slot_key = unsafe { slot_entry.key.assume_init_ref() };
                if slot_key == key {
                    return EntryLocation::Occupied { index: idx };
                }
                if slot_key < key {
                    idx += 1;
                    continue;
                }
                // slot_key > key → fall through to shift
            }

            // slot_hash > hash or slot_key > key → search for a vacant slot to shift into.
            let mut search = idx + 1;
            while search < scan_end {
                let cand_hash = unsafe { *hashes_ptr.add(search) };
                if cand_hash == VACANT_HASH {
                    return EntryLocation::Vacant {
                        insert: VacantInsert::Shift {
                            insert_index: idx,
                            vacant_index: search,
                        },
                    };
                }
                search += 1;
            }

            return EntryLocation::Vacant {
                insert: VacantInsert::NeedsGrow,
            };
        }

        EntryLocation::Vacant {
            insert: VacantInsert::NeedsGrow,
        }
    }

    #[inline(always)]
    fn _entry_with_hash(&mut self, hash: u64, key: K) -> Entry<'_, K, V, H> {
        match self.find_entry_location(hash, &key) {
            EntryLocation::Occupied { index } => {
                Entry::Occupied(OccupiedEntry { map: self, index })
            }
            EntryLocation::Vacant { insert } => Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
                insert,
            }),
        }
    }

    /// Inserts a key-value pair into the map, replacing any existing value.
    ///
    /// Computes the hash of the key and then delegates to [`Self::insert_with_hash`].
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._insert_with_hash(hash, key, value)
    }

    #[inline]
    /// Inserts a key-value pair into the map using a precomputed hash, replacing any existing value.
    pub fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        self._insert_with_hash(encode_hash(hash), key, value)
    }

    #[inline(always)]
    fn _insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        match self._insert_entry_with_hash(hash, key, value) {
            InsertResult::Inserted { .. } => None,
            InsertResult::Replaced { old, .. } => Some(old),
        }
    }

    #[inline(always)]
    fn _insert_entry_with_hash(&mut self, hash: u64, key: K, value: V) -> InsertResult<V> {
        loop {
            let ideal_slot = self.meta.ideal_slot(hash);
            let scan_end = ideal_slot + self.meta.index_bits;

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
                        let entry = &mut *entries_ptr.add(idx);
                        entry.key.write(key);
                        entry.value.write(value);
                    }
                    self.len += 1;
                    return InsertResult::Inserted { index: idx };
                }

                // Common scan: skip hashes below the insertion point.
                if slot_hash < hash {
                    idx += 1;
                    continue;
                }

                // Now slot_hash >= hash and occupied.
                if slot_hash == hash {
                    let slot = unsafe { &mut *entries_ptr.add(idx) };
                    let slot_key = unsafe { slot.key.assume_init_ref() };
                    if slot_key == &key {
                        // Key already exists; replace value.
                        let old_value = if core::mem::needs_drop::<V>() {
                            core::mem::replace(unsafe { slot.value.assume_init_mut() }, value)
                        } else {
                            // For POD values, avoid drop glue by doing a raw read+write.
                            unsafe {
                                let old = slot.value.assume_init_read();
                                slot.value.write(value);
                                old
                            }
                        };
                        return InsertResult::Replaced {
                            index: idx,
                            old: old_value,
                        };
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
                            let entry = &mut *entries_ptr.add(idx);
                            entry.key.write(key);
                            entry.value.write(value);
                        }
                        self.len += 1;
                        return InsertResult::Inserted { index: idx };
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
            self.reserve((self.slots.capacity() - self.meta.index_bits) * GROWTH_FACTOR);
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

    /// Removes a key from the map, returning the value if it existed.
    ///
    /// Computes the hash of the key and then delegates to [`Self::remove_with_hash`].
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._remove_with_hash(hash, key)
    }

    #[inline]
    /// Removes a key from the map using a precomputed hash, returning the value if it existed.
    pub fn remove_with_hash(&mut self, hash: u64, key: &K) -> Option<V> {
        self._remove_with_hash(encode_hash(hash), key)
    }

    /// Removes a key from the map, returning the key and value if it existed.
    #[inline]
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self._remove_entry_with_hash(hash, key)
    }

    /// Clears the map, removing all key-value pairs.
    #[inline]
    pub fn clear(&mut self) {
        if self.len == 0 {
            return;
        }

        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();
        let capacity = self.slots.capacity();

        for idx in 0..capacity {
            let hash = unsafe { *hashes_ptr.add(idx) };
            if hash != VACANT_HASH {
                unsafe {
                    let entry = &mut *entries_ptr.add(idx);
                    entry.key.assume_init_drop();
                    entry.value.assume_init_drop();
                    *hashes_ptr.add(idx) = VACANT_HASH;
                }
            }
        }

        self.len = 0;
    }

    /// Retains only the entries specified by the predicate.
    ///
    /// The predicate is applied to each key and mutable value. Entries for
    /// which it returns `false` are removed.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        if self.len == 0 {
            return;
        }

        let capacity = self.slots.capacity();
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        let mut idx = 0usize;
        while idx < capacity {
            let hash = unsafe { *hashes_ptr.add(idx) };
            if hash == VACANT_HASH {
                idx += 1;
                continue;
            }

            let keep = {
                let entry = unsafe { &mut *entries_ptr.add(idx) };
                let key = unsafe { entry.key.assume_init_ref() };
                let value = unsafe { entry.value.assume_init_mut() };
                f(key, value)
            };

            if keep {
                idx += 1;
            } else {
                self.remove_entry_at(idx);
            }
        }
    }

    /// Removes all entries and returns an iterator over them in deterministic order.
    ///
    /// The map is emptied even if the iterator is not fully consumed.
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        let remaining = self.len;
        self.len = 0;
        Drain {
            hashes: self.slots.hashes_ptr(),
            entries: self.slots.entries_ptr(),
            index: 0,
            capacity: self.slots.capacity(),
            remaining,
            _marker: PhantomData,
        }
    }

    /// Hot path for `remove` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline(always)]
    fn _remove_with_hash(&mut self, hash: u64, key: &K) -> Option<V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();
        let capacity = self.slots.capacity();

        let mut idx = ideal_slot;
        while idx < scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &mut *entries_ptr.add(idx) };
                let slot_key = unsafe { slot_entry.key.assume_init_ref() };
                if slot_key == key {
                    // Found the entry. Extract the value and drop the key.
                    let value = unsafe { slot_entry.value.assume_init_read() };
                    unsafe { slot_entry.key.assume_init_drop() };

                    // Backshift-delete: move subsequent entries left while they can
                    // legally occupy the vacancy.
                    let mut vacancy = idx;
                    let mut scan = idx + 1;
                    while scan < capacity {
                        let next_hash = unsafe { *hashes_ptr.add(scan) };
                        if next_hash == VACANT_HASH {
                            break;
                        }
                        let next_ideal = self.meta.ideal_slot(next_hash);
                        if next_ideal > vacancy {
                            break;
                        }
                        vacancy += 1;
                        scan += 1;
                    }

                    if scan > idx + 1 {
                        let count = scan - (idx + 1);
                        unsafe {
                            ptr::copy(hashes_ptr.add(idx + 1), hashes_ptr.add(idx), count);
                            ptr::copy(entries_ptr.add(idx + 1), entries_ptr.add(idx), count);
                        }
                    }

                    unsafe { *hashes_ptr.add(vacancy) = VACANT_HASH };
                    self.len -= 1;
                    return Some(value);
                }

                // Hash collision, but key order lets us stop early.
                if slot_key > key {
                    return None;
                }
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }

            idx += 1;
        }

        None
    }

    #[inline(always)]
    fn _remove_entry_with_hash(&mut self, hash: u64, key: &K) -> Option<(K, V)> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + self.meta.index_bits;

        // SAFETY: ideal_slot..scan_end is always in-bounds for the backing allocation.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        let mut idx = ideal_slot;
        while idx < scan_end {
            let slot_hash = unsafe { *hashes_ptr.add(idx) };

            if slot_hash == hash {
                let slot_entry = unsafe { &*entries_ptr.add(idx) };
                let slot_key = unsafe { slot_entry.key.assume_init_ref() };
                if slot_key == key {
                    return Some(self.remove_entry_at(idx));
                }

                // Hash collision, but key order lets us stop early.
                if slot_key > key {
                    return None;
                }
            } else if slot_hash > hash {
                return None; // also catches VACANT_HASH if VACANT_HASH > any valid hash
            }

            idx += 1;
        }

        None
    }

    #[inline(always)]
    fn remove_entry_at(&mut self, idx: usize) -> (K, V) {
        // SAFETY: idx is a valid occupied slot when called.
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();
        let capacity = self.slots.capacity();

        let entry = unsafe { &mut *entries_ptr.add(idx) };
        let key = unsafe { entry.key.assume_init_read() };
        let value = unsafe { entry.value.assume_init_read() };

        // Backshift-delete: move subsequent entries left while they can
        // legally occupy the vacancy.
        let mut vacancy = idx;
        let mut scan = idx + 1;
        while scan < capacity {
            let next_hash = unsafe { *hashes_ptr.add(scan) };
            if next_hash == VACANT_HASH {
                break;
            }
            let next_ideal = self.meta.ideal_slot(next_hash);
            if next_ideal > vacancy {
                break;
            }
            vacancy += 1;
            scan += 1;
        }

        if scan > idx + 1 {
            let count = scan - (idx + 1);
            unsafe {
                ptr::copy(hashes_ptr.add(idx + 1), hashes_ptr.add(idx), count);
                ptr::copy(entries_ptr.add(idx + 1), entries_ptr.add(idx), count);
            }
        }

        unsafe { *hashes_ptr.add(vacancy) = VACANT_HASH };
        self.len -= 1;

        (key, value)
    }

    /// Shrinks the map to at least `min_capacity` elements (logical capacity).
    ///
    /// If the target size cannot satisfy max-scan constraints, we retry once with the
    /// next larger power-of-two ideal range. Returns the resulting capacity.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) -> usize {
        let requested = min_capacity.max(self.len).max(MIN_CAPACITY);
        let (new_meta, new_capacity) = PoMapMeta::new(requested.next_power_of_two());
        let current_capacity = self.slots.capacity();
        let current_meta = self.meta;

        if new_capacity >= current_capacity {
            return current_capacity;
        }

        let mut old_slots = core::mem::replace(&mut self.slots, Slots::new(new_capacity));

        let mut cursor = 0usize;
        let first_attempt =
            Self::pack_from_slots(&mut self.slots, &new_meta, &mut cursor, &mut old_slots, 0);
        if first_attempt.is_ok() {
            self.meta = new_meta;
            return new_capacity;
        }

        let failed_idx = first_attempt.err().unwrap();

        let ideal_range = 1usize << new_meta.index_bits;
        let Some(bumped_ideal_range) = ideal_range.checked_mul(2) else {
            return current_capacity;
        };
        let (bumped_meta, bumped_capacity) = PoMapMeta::new(bumped_ideal_range);

        if bumped_capacity < current_capacity {
            let mut bumped_slots = Slots::new(bumped_capacity);
            let mut bump_cursor = 0usize;
            let bump_prefix = Self::pack_from_slots(
                &mut bumped_slots,
                &bumped_meta,
                &mut bump_cursor,
                &mut self.slots,
                0,
            );

            match bump_prefix {
                Ok(()) => {
                    let bump_old = Self::pack_from_slots(
                        &mut bumped_slots,
                        &bumped_meta,
                        &mut bump_cursor,
                        &mut old_slots,
                        failed_idx,
                    );

                    if bump_old.is_ok() {
                        self.slots = bumped_slots;
                        self.meta = bumped_meta;
                        return bumped_capacity;
                    }

                    let failed_old_idx = bump_old.err().unwrap();
                    let mut final_slots = Slots::new(current_capacity);
                    let mut final_cursor = 0usize;
                    Self::pack_from_slots(
                        &mut final_slots,
                        &current_meta,
                        &mut final_cursor,
                        &mut bumped_slots,
                        0,
                    )
                    .expect("repack into current capacity failed");
                    Self::pack_from_slots(
                        &mut final_slots,
                        &current_meta,
                        &mut final_cursor,
                        &mut old_slots,
                        failed_old_idx,
                    )
                    .expect("repack into current capacity failed");
                    self.slots = final_slots;
                    return current_capacity;
                }
                Err(failed_prefix_idx) => {
                    let mut final_slots = Slots::new(current_capacity);
                    let mut final_cursor = 0usize;
                    Self::pack_from_slots(
                        &mut final_slots,
                        &current_meta,
                        &mut final_cursor,
                        &mut bumped_slots,
                        0,
                    )
                    .expect("repack into current capacity failed");
                    Self::pack_from_slots(
                        &mut final_slots,
                        &current_meta,
                        &mut final_cursor,
                        &mut self.slots,
                        failed_prefix_idx,
                    )
                    .expect("repack into current capacity failed");
                    Self::pack_from_slots(
                        &mut final_slots,
                        &current_meta,
                        &mut final_cursor,
                        &mut old_slots,
                        failed_idx,
                    )
                    .expect("repack into current capacity failed");
                    self.slots = final_slots;
                    return current_capacity;
                }
            }
        }

        let mut final_slots = Slots::new(current_capacity);
        let mut final_cursor = 0usize;
        Self::pack_from_slots(
            &mut final_slots,
            &current_meta,
            &mut final_cursor,
            &mut self.slots,
            0,
        )
        .expect("repack into current capacity failed");
        Self::pack_from_slots(
            &mut final_slots,
            &current_meta,
            &mut final_cursor,
            &mut old_slots,
            failed_idx,
        )
        .expect("repack into current capacity failed");
        self.slots = final_slots;
        current_capacity
    }

    /// Shrinks the map as much as possible based on the current length.
    #[inline]
    pub fn shrink_to_fit(&mut self) -> usize {
        self.shrink_to(self.len)
    }

    /// Reserve capacity for at least `requested_capacity` elements if there is not enough
    /// capacity already.
    ///
    /// Returns the new capacity.
    #[cold]
    pub fn reserve(&mut self, requested_capacity: usize) -> usize {
        // generate new meta
        let current_capacity = self.slots.capacity();
        let (new_meta, new_vec_capacity) = PoMapMeta::new(requested_capacity.next_power_of_two());
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

/// Iterator over shared references to key-value pairs in deterministic order.
#[must_use]
pub struct Iter<'a, K: Key, V: Value> {
    hashes: *const u64,
    entries: *const SlotEntry<K, V>,
    index: usize,
    capacity: usize,
    remaining: usize,
    _marker: PhantomData<&'a (K, V)>,
}

impl<'a, K: Key, V: Value> Clone for Iter<'a, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            hashes: self.hashes,
            entries: self.entries,
            index: self.index,
            capacity: self.capacity,
            remaining: self.remaining,
            _marker: PhantomData,
        }
    }
}

impl<'a, K: Key, V: Value> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.capacity {
            let idx = self.index;
            self.index += 1;
            let hash = unsafe { *self.hashes.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            self.remaining -= 1;
            let entry = unsafe { &*self.entries.add(idx) };
            let key = unsafe { entry.key.assume_init_ref() };
            let value = unsafe { entry.value.assume_init_ref() };
            return Some((key, value));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for Iter<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, K: Key, V: Value> FusedIterator for Iter<'a, K, V> {}

/// Iterator over mutable references to values in deterministic order.
#[must_use]
pub struct IterMut<'a, K: Key, V: Value> {
    hashes: *const u64,
    entries: *mut SlotEntry<K, V>,
    index: usize,
    capacity: usize,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K: Key, V: Value> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.capacity {
            let idx = self.index;
            self.index += 1;
            let hash = unsafe { *self.hashes.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            self.remaining -= 1;
            let entry = unsafe { &mut *self.entries.add(idx) };
            let key = unsafe { entry.key.assume_init_ref() };
            let value = unsafe { entry.value.assume_init_mut() };
            return Some((key, value));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for IterMut<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, K: Key, V: Value> FusedIterator for IterMut<'a, K, V> {}

/// Iterator over shared references to keys in deterministic order.
#[must_use]
pub struct Keys<'a, K: Key, V: Value> {
    iter: Iter<'a, K, V>,
}

impl<'a, K: Key, V: Value> Clone for Keys<'a, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K: Key, V: Value> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for Keys<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: Key, V: Value> FusedIterator for Keys<'a, K, V> {}

/// Iterator over shared references to values in deterministic order.
#[must_use]
pub struct Values<'a, K: Key, V: Value> {
    iter: Iter<'a, K, V>,
}

impl<'a, K: Key, V: Value> Clone for Values<'a, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K: Key, V: Value> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for Values<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: Key, V: Value> FusedIterator for Values<'a, K, V> {}

/// Iterator over mutable references to values in deterministic order.
#[must_use]
pub struct ValuesMut<'a, K: Key, V: Value> {
    iter: IterMut<'a, K, V>,
}

impl<'a, K: Key, V: Value> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for ValuesMut<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K: Key, V: Value> FusedIterator for ValuesMut<'a, K, V> {}

/// Owning iterator over key-value pairs in deterministic order.
#[must_use]
pub struct IntoIter<K: Key, V: Value> {
    slots: Slots<K, V>,
    index: usize,
    capacity: usize,
    remaining: usize,
}

impl<K: Key, V: Value> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let hashes_ptr = self.slots.hashes_ptr();
        let entries_ptr = self.slots.entries_ptr();

        while self.index < self.capacity {
            let idx = self.index;
            self.index += 1;
            let hash = unsafe { *hashes_ptr.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            self.remaining -= 1;
            unsafe { *hashes_ptr.add(idx) = VACANT_HASH };
            let entry = unsafe { &mut *entries_ptr.add(idx) };
            let key = unsafe { entry.key.assume_init_read() };
            let value = unsafe { entry.value.assume_init_read() };
            return Some((key, value));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K: Key, V: Value> ExactSizeIterator for IntoIter<K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: Key, V: Value> FusedIterator for IntoIter<K, V> {}

/// Draining iterator over key-value pairs in deterministic order.
#[must_use]
pub struct Drain<'a, K: Key, V: Value> {
    hashes: *mut u64,
    entries: *mut SlotEntry<K, V>,
    index: usize,
    capacity: usize,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

impl<'a, K: Key, V: Value> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.capacity {
            let idx = self.index;
            self.index += 1;
            let hash = unsafe { *self.hashes.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            self.remaining -= 1;
            unsafe { *self.hashes.add(idx) = VACANT_HASH };
            let entry = unsafe { &mut *self.entries.add(idx) };
            let key = unsafe { entry.key.assume_init_read() };
            let value = unsafe { entry.value.assume_init_read() };
            return Some((key, value));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K: Key, V: Value> ExactSizeIterator for Drain<'a, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, K: Key, V: Value> FusedIterator for Drain<'a, K, V> {}

impl<'a, K: Key, V: Value> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        while self.index < self.capacity {
            let idx = self.index;
            self.index += 1;
            let hash = unsafe { *self.hashes.add(idx) };
            if hash == VACANT_HASH {
                continue;
            }
            unsafe { *self.hashes.add(idx) = VACANT_HASH };
            let entry = unsafe { &mut *self.entries.add(idx) };
            unsafe {
                entry.key.assume_init_drop();
                entry.value.assume_init_drop();
            }
        }
        self.remaining = 0;
    }
}

/// Owning iterator over keys in deterministic order.
#[must_use]
pub struct IntoKeys<K: Key, V: Value> {
    iter: IntoIter<K, V>,
}

impl<K: Key, V: Value> Iterator for IntoKeys<K, V> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: Key, V: Value> ExactSizeIterator for IntoKeys<K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Key, V: Value> FusedIterator for IntoKeys<K, V> {}

/// Owning iterator over values in deterministic order.
#[must_use]
pub struct IntoValues<K: Key, V: Value> {
    iter: IntoIter<K, V>,
}

impl<K: Key, V: Value> Iterator for IntoValues<K, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K: Key, V: Value> ExactSizeIterator for IntoValues<K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Key, V: Value> FusedIterator for IntoValues<K, V> {}

impl<'a, K: Key, V: Value, H: Hasher + Default> IntoIterator for &'a PoMap<K, V, H> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> IntoIterator for &'a mut PoMap<K, V, H> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V: Value, H: Hasher + Default> IntoIterator for PoMap<K, V, H> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let capacity = self.slots.capacity();
        IntoIter {
            slots: self.slots,
            index: 0,
            capacity,
            remaining: self.len,
        }
    }
}

enum InsertResult<V> {
    Inserted { index: usize },
    Replaced { index: usize, old: V },
}

enum VacantInsert {
    Direct {
        index: usize,
    },
    Shift {
        insert_index: usize,
        vacant_index: usize,
    },
    NeedsGrow,
}

enum EntryLocation {
    Occupied { index: usize },
    Vacant { insert: VacantInsert },
}

/// A view into a single entry in a [`PoMap`], similar to `std::collections::hash_map::Entry`.
#[must_use]
pub enum Entry<'a, K: Key, V: Value, H: Hasher + Default> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, H>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, H>),
}

/// A view into an occupied entry in a [`PoMap`].
pub struct OccupiedEntry<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a mut PoMap<K, V, H>,
    index: usize,
}

/// A view into a vacant entry in a [`PoMap`].
pub struct VacantEntry<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a mut PoMap<K, V, H>,
    hash: u64,
    key: K,
    insert: VacantInsert,
}

impl<'a, K: Key, V: Value, H: Hasher + Default> Entry<'a, K, V, H> {
    #[inline]
    /// Ensures a value is in the entry by inserting the default if vacant.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    #[inline]
    /// Ensures a value is in the entry by inserting the result of the function if vacant.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    #[inline]
    /// Ensures a value is in the entry by inserting the result of the function if vacant.
    /// The function receives a reference to the entry's key.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = default(entry.key());
                entry.insert(value)
            }
        }
    }

    #[inline]
    /// Ensures a value is in the entry by inserting `V::default()` if vacant.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(V::default()),
        }
    }

    #[inline]
    /// Applies a function to the value if the entry is occupied.
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Entry<'a, K, V, H> {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            Entry::Vacant(entry) => Entry::Vacant(entry),
        }
    }

    #[inline]
    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> OccupiedEntry<'a, K, V, H> {
    #[inline]
    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(self.index) };
        unsafe { entry.key.assume_init_ref() }
    }

    #[inline]
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_ref() }
    }

    #[inline]
    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_mut() }
    }

    #[inline]
    /// Converts the entry into a mutable reference to the value.
    pub fn into_mut(self) -> &'a mut V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_mut() }
    }

    #[inline]
    /// Replaces the entry's value, returning the old value.
    pub fn insert(&mut self, value: V) -> V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        if core::mem::needs_drop::<V>() {
            core::mem::replace(unsafe { entry.value.assume_init_mut() }, value)
        } else {
            unsafe {
                let old = entry.value.assume_init_read();
                entry.value.write(value);
                old
            }
        }
    }

    #[inline]
    /// Removes the entry from the map and returns the stored value.
    pub fn remove(self) -> V {
        let (key, value) = self.remove_entry();
        drop(key);
        value
    }

    #[inline]
    /// Removes the entry from the map and returns the stored key and value.
    pub fn remove_entry(self) -> (K, V) {
        let index = self.index;
        self.map.remove_entry_at(index)
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> VacantEntry<'a, K, V, H> {
    #[inline]
    /// Returns a reference to the key that would be inserted.
    pub const fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    /// Takes ownership of the key.
    pub fn into_key(self) -> K {
        self.key
    }

    #[inline]
    /// Inserts the value for this vacant entry and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        let hash = self.hash;
        let key = self.key;
        let map = self.map;
        let insert = self.insert;

        let index = match insert {
            VacantInsert::Direct { index } => {
                let hashes_ptr = map.slots.hashes_ptr();
                let entries_ptr = map.slots.entries_ptr();
                unsafe {
                    *hashes_ptr.add(index) = hash;
                    let entry = &mut *entries_ptr.add(index);
                    entry.key.write(key);
                    entry.value.write(value);
                }
                map.len += 1;
                index
            }
            VacantInsert::Shift {
                insert_index,
                vacant_index,
            } => {
                let hashes_ptr = map.slots.hashes_ptr();
                let entries_ptr = map.slots.entries_ptr();
                unsafe {
                    let mut i = vacant_index;
                    while i > insert_index {
                        *hashes_ptr.add(i) = *hashes_ptr.add(i - 1);
                        core::ptr::write(
                            entries_ptr.add(i),
                            core::ptr::read(entries_ptr.add(i - 1)),
                        );
                        i -= 1;
                    }

                    *hashes_ptr.add(insert_index) = hash;
                    let entry = &mut *entries_ptr.add(insert_index);
                    entry.key.write(key);
                    entry.value.write(value);
                }
                map.len += 1;
                insert_index
            }
            VacantInsert::NeedsGrow => {
                let result = map._insert_entry_with_hash(hash, key, value);
                match result {
                    InsertResult::Inserted { index } => index,
                    InsertResult::Replaced { index, old } => {
                        debug_assert!(false, "VacantEntry::insert replaced an existing value");
                        drop(old);
                        index
                    }
                }
            }
        };

        let entries_ptr = map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(index) };
        unsafe { entry.value.assume_init_mut() }
    }
}

/// A builder for computing where in a [`PoMap`] a key-value pair would be stored.
#[must_use]
pub struct RawEntryBuilderMut<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a mut PoMap<K, V, H>,
}

/// A builder for computing where in a [`PoMap`] a key-value pair would be stored.
#[must_use]
pub struct RawEntryBuilder<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a PoMap<K, V, H>,
}

/// A view into a single raw entry in a [`PoMap`].
#[must_use]
pub enum RawEntryMut<'a, K: Key, V: Value, H: Hasher + Default> {
    /// An occupied entry.
    Occupied(RawOccupiedEntryMut<'a, K, V, H>),
    /// A vacant entry.
    Vacant(RawVacantEntryMut<'a, K, V, H>),
}

/// A view into an occupied raw entry in a [`PoMap`].
pub struct RawOccupiedEntryMut<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a mut PoMap<K, V, H>,
    index: usize,
}

/// A view into a vacant raw entry in a [`PoMap`].
pub struct RawVacantEntryMut<'a, K: Key, V: Value, H: Hasher + Default> {
    map: &'a mut PoMap<K, V, H>,
    hash: u64,
}

impl<'a, K: Key, V: Value, H: Hasher + Default> RawEntryBuilderMut<'a, K, V, H> {
    /// Creates a [`RawEntryMut`] from the given key.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key(self, key: &K) -> RawEntryMut<'a, K, V, H> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.from_key_hashed_nocheck(hash, key)
    }

    /// Creates a [`RawEntryMut`] from a key and its precomputed hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck(self, hash: u64, key: &K) -> RawEntryMut<'a, K, V, H> {
        let hash = encode_hash(hash);
        match self.map.find_entry_location(hash, key) {
            EntryLocation::Occupied { index } => RawEntryMut::Occupied(RawOccupiedEntryMut {
                map: self.map,
                index,
            }),
            EntryLocation::Vacant { .. } => RawEntryMut::Vacant(RawVacantEntryMut {
                map: self.map,
                hash,
            }),
        }
    }

    /// Creates a [`RawEntryMut`] from a hash and matching function.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, mut is_match: F) -> RawEntryMut<'a, K, V, H>
    where
        F: FnMut(&K) -> bool,
    {
        let hash = encode_hash(hash);
        match self.map._find_index_by_hash(hash, &mut is_match) {
            Some(index) => RawEntryMut::Occupied(RawOccupiedEntryMut {
                map: self.map,
                index,
            }),
            None => RawEntryMut::Vacant(RawVacantEntryMut {
                map: self.map,
                hash,
            }),
        }
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> RawEntryBuilder<'a, K, V, H> {
    /// Accesses an immutable entry by key.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key(self, key: &K) -> Option<(&'a K, &'a V)> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = encode_hash(hasher.finish());
        self.map._get_key_value_with_hash(hash, key)
    }

    /// Accesses an immutable entry by a key and its precomputed hash.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_key_hashed_nocheck(self, hash: u64, key: &K) -> Option<(&'a K, &'a V)> {
        let hash = encode_hash(hash);
        self.map._get_key_value_with_hash(hash, key)
    }

    /// Accesses an immutable entry by hash and matching function.
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_hash<F>(self, hash: u64, mut is_match: F) -> Option<(&'a K, &'a V)>
    where
        F: FnMut(&K) -> bool,
    {
        let hash = encode_hash(hash);
        let index = self.map._find_index_by_hash(hash, &mut is_match)?;
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(index) };
        let key = unsafe { entry.key.assume_init_ref() };
        let value = unsafe { entry.value.assume_init_ref() };
        Some((key, value))
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> RawEntryMut<'a, K, V, H> {
    /// Inserts a key-value pair into the entry, returning an occupied entry view.
    #[inline]
    pub fn insert(self, key: K, value: V) -> RawOccupiedEntryMut<'a, K, V, H> {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                entry.insert(value);
                drop(key);
                entry
            }
            RawEntryMut::Vacant(entry) => entry.insert_entry(key, value),
        }
    }

    /// Ensures a value is in the entry by inserting the default if vacant.
    #[inline]
    pub fn or_insert(self, default_key: K, default_val: V) -> (&'a K, &'a mut V) {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => entry.insert(default_key, default_val),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the function if vacant.
    #[inline]
    pub fn or_insert_with<F>(self, default: F) -> (&'a K, &'a mut V)
    where
        F: FnOnce() -> (K, V),
    {
        match self {
            RawEntryMut::Occupied(entry) => entry.into_key_value(),
            RawEntryMut::Vacant(entry) => {
                let (key, value) = default();
                entry.insert(key, value)
            }
        }
    }

    /// Applies a function to the entry if it is occupied.
    #[inline]
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&K, &mut V),
    {
        match self {
            RawEntryMut::Occupied(mut entry) => {
                {
                    let (key, value) = entry.get_key_value_mut();
                    f(key, value);
                }
                RawEntryMut::Occupied(entry)
            }
            RawEntryMut::Vacant(entry) => RawEntryMut::Vacant(entry),
        }
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> RawOccupiedEntryMut<'a, K, V, H> {
    /// Returns a reference to the entry's key.
    #[inline]
    pub fn key(&self) -> &K {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(self.index) };
        unsafe { entry.key.assume_init_ref() }
    }

    /// Gets a reference to the value in the entry.
    #[inline]
    pub fn get(&self) -> &V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_ref() }
    }

    /// Gets a mutable reference to the value in the entry.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_mut() }
    }

    /// Converts the entry into a mutable reference to the value.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        unsafe { entry.value.assume_init_mut() }
    }

    /// Gets a reference to the key and value in the entry.
    #[inline]
    pub fn get_key_value(&self) -> (&K, &V) {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &*entries_ptr.add(self.index) };
        let key = unsafe { entry.key.assume_init_ref() };
        let value = unsafe { entry.value.assume_init_ref() };
        (key, value)
    }

    /// Gets a reference to the key and a mutable reference to the value in the entry.
    #[inline]
    pub fn get_key_value_mut(&mut self) -> (&K, &mut V) {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        let key = unsafe { entry.key.assume_init_ref() };
        let value = unsafe { entry.value.assume_init_mut() };
        (key, value)
    }

    /// Converts the entry into a reference to the key and a mutable reference to the value.
    #[inline]
    pub fn into_key_value(self) -> (&'a K, &'a mut V) {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        let key = unsafe { entry.key.assume_init_ref() };
        let value = unsafe { entry.value.assume_init_mut() };
        (key, value)
    }

    /// Replaces the entry's value, returning the old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        let entries_ptr = self.map.slots.entries_ptr();
        let entry = unsafe { &mut *entries_ptr.add(self.index) };
        if core::mem::needs_drop::<V>() {
            core::mem::replace(unsafe { entry.value.assume_init_mut() }, value)
        } else {
            unsafe {
                let old = entry.value.assume_init_read();
                entry.value.write(value);
                old
            }
        }
    }

    /// Removes the entry from the map and returns the stored value.
    #[inline]
    pub fn remove(self) -> V {
        let (key, value) = self.remove_entry();
        drop(key);
        value
    }

    /// Removes the entry from the map and returns the stored key and value.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let index = self.index;
        self.map.remove_entry_at(index)
    }
}

impl<'a, K: Key, V: Value, H: Hasher + Default> RawVacantEntryMut<'a, K, V, H> {
    /// Inserts the key and value, returning a reference to the key and a mutable reference to the value.
    ///
    /// The hash supplied to the raw entry must correspond to the key.
    #[inline]
    pub fn insert(self, key: K, value: V) -> (&'a K, &'a mut V) {
        self.insert_entry(key, value).into_key_value()
    }

    #[inline]
    fn insert_entry(self, key: K, value: V) -> RawOccupiedEntryMut<'a, K, V, H> {
        let map = self.map;
        let hash = self.hash;
        let index = match map._insert_entry_with_hash(hash, key, value) {
            InsertResult::Inserted { index } => index,
            InsertResult::Replaced { index, old } => {
                debug_assert!(
                    false,
                    "RawVacantEntryMut::insert replaced an existing value"
                );
                drop(old);
                index
            }
        };
        RawOccupiedEntryMut { map, index }
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
            let key = unsafe { entry.key.assume_init_ref().clone() };
            let value = unsafe { entry.value.assume_init_ref().clone() };
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

impl<K: Key, V: Value, H: Hasher + Default> Default for PoMap<K, V, H> {
    #[inline(always)]
    /// Creates an empty map using [`PoMap::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V: Value, H: Hasher + Default> fmt::Debug for PoMap<K, V, H>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in self.iter() {
            map.entry(k, v);
        }
        map.finish()
    }
}

impl<K: Key, V: Value + PartialEq, H: Hasher + Default> PartialEq for PoMap<K, V, H> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<K: Key, V: Value + Eq, H: Hasher + Default> Eq for PoMap<K, V, H> {}

impl<K: Key, V: Value + PartialOrd, H: Hasher + Default> PartialOrd for PoMap<K, V, H> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<K: Key, V: Value + Ord, H: Hasher + Default> Ord for PoMap<K, V, H> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<K: Key, V: Value + Hash, H: Hasher + Default> Hash for PoMap<K, V, H> {
    #[inline]
    fn hash<S: Hasher>(&self, state: &mut S) {
        self.len.hash(state);
        for (k, v) in self.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: Key, V: Value, H: Hasher + Default> Extend<(K, V)> for PoMap<K, V, H> {
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: Hasher + Default> Extend<(&'a K, &'a V)>
    for PoMap<K, V, H>
{
    #[inline]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key.clone(), value.clone());
        }
    }
}

impl<K: Key, V: Value, H: Hasher + Default> FromIterator<(K, V)> for PoMap<K, V, H> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = PoMap::with_capacity(lower);
        map.extend(iter);
        map
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: Hasher + Default> FromIterator<(&'a K, &'a V)>
    for PoMap<K, V, H>
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (&'a K, &'a V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = PoMap::with_capacity(lower);
        map.extend(iter);
        map
    }
}

impl<K: Key, V: Value, H: Hasher + Default> Index<&K> for PoMap<K, V, H> {
    type Output = V;

    #[inline]
    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).expect("key not found")
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
    /// Number of index bits m where `ideal_range = 1 << m`, also equal to max_scan for this capacity.
    index_bits: usize,

    /// Shift to extract the top-m bits from the 64-bit hash:
    ///   ideal_slot = (hash >> index_shift)
    index_shift: usize,
}

impl PoMapMeta {
    /// Build meta from a *requested* ideal range (power of two).
    ///
    /// We choose:
    ///   ideal_range = max(requested, MIN_CAPACITY)
    ///   index_bits  = log2(ideal_range)
    ///
    /// This means:
    ///   - `ideal_range` is a power of two
    ///   - full Vec capacity should be `ideal_range + index_bits`
    ///
    /// Returns `(meta, capacity)`, where `capacity` is what you should use
    /// for the backing allocation.
    #[inline(always)]
    const fn new(requested: usize) -> (Self, usize) {
        let ideal_range = if requested > MIN_CAPACITY {
            requested
        } else {
            MIN_CAPACITY
        };

        debug_assert!(ideal_range.is_power_of_two());
        let index_bits = ideal_range.trailing_zeros() as usize; // m
        debug_assert!(index_bits <= HASH_BITS);
        let index_shift: usize = HASH_BITS - index_bits; // use top-m bits of the u64 hash

        (
            Self {
                index_bits,
                index_shift,
            },
            ideal_range + index_bits,
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
    use ahash::AHasher;
    use alloc::{format, vec, vec::Vec};
    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::{
        cell::Cell, cmp::Ordering, collections::BTreeSet, collections::HashMap,
        hash::DefaultHasher, rc::Rc,
    };

    #[derive(Clone)]
    struct DropCounter {
        drops: Rc<Cell<usize>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[derive(Clone)]
    struct DropKey {
        id: u64,
        drops: Rc<Cell<usize>>,
    }

    impl Drop for DropKey {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    impl Hash for DropKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    impl PartialEq for DropKey {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }

    impl Eq for DropKey {}

    impl PartialOrd for DropKey {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.id.cmp(&other.id))
        }
    }

    impl Ord for DropKey {
        fn cmp(&self, other: &Self) -> Ordering {
            self.id.cmp(&other.id)
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct CollisionKey(u64);

    const COLLISION_HASH: u64 = 0xBAD5EED_u64;

    impl Hash for CollisionKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u64(COLLISION_HASH);
        }
    }

    fn encoded_hash<K: Hash>(key: &K) -> u64 {
        let mut hasher = AHasher::default();
        key.hash(&mut hasher);
        encode_hash(hasher.finish())
    }

    fn raw_hash<K: Hash>(key: &K) -> u64 {
        let mut hasher = AHasher::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn order_by_hash_then_key(keys: &[u64]) -> Vec<u64> {
        let mut with_hash = keys
            .iter()
            .map(|&k| (encoded_hash(&k), k))
            .collect::<Vec<_>>();
        with_hash.sort();
        with_hash.into_iter().map(|(_, k)| k).collect()
    }

    fn find_keys_with_same_slot(meta: &PoMapMeta, count: usize) -> Vec<u64> {
        assert!(count > 0);
        let ideal_range = 1usize << meta.index_bits;
        let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); ideal_range];
        let mut key = 0u64;
        loop {
            let slot = meta.ideal_slot(encoded_hash(&key));
            let bucket = &mut buckets[slot];
            if bucket.len() < count {
                bucket.push(key);
                if bucket.len() == count {
                    return bucket.clone();
                }
            }
            key = key.wrapping_add(1);
        }
    }

    fn find_keys_distinct_slots(meta: &PoMapMeta) -> (u64, u64) {
        let ideal_range = 1usize << meta.index_bits;
        let mut first_for_slot: Vec<Option<u64>> = vec![None; ideal_range];
        let mut key = 0u64;
        loop {
            let slot = meta.ideal_slot(encoded_hash(&key));
            if first_for_slot[slot].is_none() {
                first_for_slot[slot] = Some(key);
                let mut first: Option<(usize, u64)> = None;
                let mut second: Option<(usize, u64)> = None;
                for (idx, val) in first_for_slot.iter().enumerate() {
                    if let Some(k) = val {
                        if first.is_none() {
                            first = Some((idx, *k));
                        } else {
                            second = Some((idx, *k));
                            break;
                        }
                    }
                }
                if let (Some((s1, k1)), Some((s2, k2))) = (first, second) {
                    return if s1 < s2 { (k1, k2) } else { (k2, k1) };
                }
            }
            key = key.wrapping_add(1);
        }
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut map: PoMap<u64, u64> = PoMap::new();

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
        let mut map: PoMap<u64, u64> = PoMap::new();

        assert_eq!(map.insert(42, 1), None);
        assert_eq!(map.insert(42, 2), Some(1));
        assert_eq!(map.get(&42), Some(&2));
        assert_eq!(map.len(), 1);
    }

    fn build_map_with_order(keys: &[u64], capacity: usize) -> PoMap<u64, u64> {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(capacity);
        for &key in keys {
            map.insert(key, key + 100);
        }
        map
    }

    fn hash_map(map: &PoMap<u64, u64>) -> u64 {
        let mut hasher = DefaultHasher::new();
        map.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn determinism_across_insertion_order_and_capacity() {
        let keys = [10u64, 3, 7, 1, 9, 4, 2, 8, 6, 5];
        let mut reversed = keys;
        reversed.reverse();

        let map_a = build_map_with_order(&keys, 4);
        let map_b = build_map_with_order(&reversed, 128);

        let a_items = map_a.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        let b_items = map_b.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        assert_eq!(a_items, b_items);

        let a_keys = map_a.keys().copied().collect::<Vec<_>>();
        let b_keys = map_b.keys().copied().collect::<Vec<_>>();
        assert_eq!(a_keys, b_keys);

        let a_values = map_a.values().copied().collect::<Vec<_>>();
        let b_values = map_b.values().copied().collect::<Vec<_>>();
        assert_eq!(a_values, b_values);
    }

    #[test]
    fn determinism_with_hash_collisions_orders_by_key() {
        let keys = [CollisionKey(1), CollisionKey(0), CollisionKey(2)];
        let mut reversed = keys.clone();
        reversed.reverse();

        let mut map_a: PoMap<CollisionKey, u64> = PoMap::with_capacity(4);
        for key in keys.iter().cloned() {
            map_a.insert(key.clone(), key.0 + 100);
        }

        let mut map_b: PoMap<CollisionKey, u64> = PoMap::with_capacity(4);
        for key in reversed.iter().cloned() {
            map_b.insert(key.clone(), key.0 + 100);
        }

        let a_keys = map_a.keys().map(|k| k.0).collect::<Vec<_>>();
        let b_keys = map_b.keys().map(|k| k.0).collect::<Vec<_>>();
        assert_eq!(a_keys, b_keys);
        assert_eq!(a_keys, vec![0, 1, 2]);
    }

    #[test]
    fn iter_mut_and_values_mut_update_values() {
        let keys = [5u64, 1, 3];
        let mut map = build_map_with_order(&keys, 8);

        let mut seen = Vec::new();
        for (k, v) in map.iter_mut() {
            seen.push(*k);
            *v += 1;
        }
        assert_eq!(seen, map.keys().copied().collect::<Vec<_>>());
        for (k, v) in map.iter() {
            assert_eq!(*v, *k + 101);
        }

        for v in map.values_mut() {
            *v += 1;
        }
        for (k, v) in map.iter() {
            assert_eq!(*v, *k + 102);
        }
    }

    #[test]
    fn into_iter_and_into_keys_values_are_deterministic() {
        let keys = [4u64, 2, 6, 1];
        let map = build_map_with_order(&keys, 8);
        let expected = map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();

        let into_items = map.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(into_items, expected);

        let into_keys = map.clone().into_keys().collect::<Vec<_>>();
        assert_eq!(
            into_keys,
            expected.iter().map(|(k, _)| *k).collect::<Vec<_>>()
        );

        let into_values = map.into_values().collect::<Vec<_>>();
        assert_eq!(
            into_values,
            expected.iter().map(|(_, v)| *v).collect::<Vec<_>>()
        );
    }

    #[test]
    fn contains_get_key_value_and_index_work() {
        let map = build_map_with_order(&[7u64, 1, 9], 4);
        assert!(map.contains_key(&7));
        assert!(!map.contains_key(&8));

        let (k, v) = map.get_key_value(&7).expect("expected key");
        assert_eq!((*k, *v), (7, 107));

        assert_eq!(map[&1], 101);
    }

    #[test]
    #[should_panic(expected = "key not found")]
    fn index_panics_on_missing_key() {
        let map = build_map_with_order(&[1u64], 4);
        let _ = map[&2];
    }

    #[test]
    fn remove_entry_and_clear_work() {
        let mut map = build_map_with_order(&[2u64, 5, 1], 4);
        let removed = map.remove_entry(&5).expect("expected removal");
        assert_eq!(removed, (5, 105));
        assert!(!map.contains_key(&5));

        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.iter().next(), None);
    }

    #[test]
    fn retain_filters_and_mutates() {
        let mut map = build_map_with_order(&[1u64, 2, 3, 4], 8);

        map.retain(|k, v| {
            *v += 1;
            k % 2 == 0
        });

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&2), Some(&103));
        assert_eq!(map.get(&4), Some(&105));
        assert!(!map.contains_key(&1));
        assert!(!map.contains_key(&3));
    }

    #[test]
    fn drain_clears_and_yields_in_order() {
        let mut map = build_map_with_order(&[3u64, 1, 4, 2], 8);
        let expected = map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();

        let drained = map.drain().collect::<Vec<_>>();

        assert_eq!(drained, expected);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert(9, 99);
        assert_eq!(map.get(&9), Some(&99));
    }

    #[test]
    fn drain_drop_clears_remaining() {
        let mut map = build_map_with_order(&[1u64, 2, 3], 4);

        {
            let mut drain = map.drain();
            let _ = drain.next();
        }

        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn deterministic_eq_ord_hash_and_debug() {
        let keys = [9u64, 3, 6, 1];
        let mut reversed = keys;
        reversed.reverse();

        let map_a = build_map_with_order(&keys, 4);
        let map_b = build_map_with_order(&reversed, 64);

        assert_eq!(map_a, map_b);
        assert_eq!(map_a.cmp(&map_b), Ordering::Equal);
        assert_eq!(hash_map(&map_a), hash_map(&map_b));
        assert_eq!(format!("{:?}", map_a), format!("{:?}", map_b));

        let map_c = build_map_with_order(&[1u64, 4], 4);
        let map_d = build_map_with_order(&[1u64, 5], 4);
        let c_items = map_c.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        let d_items = map_d.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        assert_eq!(map_c.cmp(&map_d), c_items.cmp(&d_items));
    }

    #[test]
    fn raw_entry_lookup_variants() {
        let map = build_map_with_order(&[5u64, 1, 3], 8);
        let hash = raw_hash(&1u64);

        let kv = map.raw_entry().from_key(&1).map(|(k, v)| (*k, *v));
        assert_eq!(kv, Some((1, 101)));

        let kv = map
            .raw_entry()
            .from_key_hashed_nocheck(hash, &1)
            .map(|(k, v)| (*k, *v));
        assert_eq!(kv, Some((1, 101)));

        let kv = map
            .raw_entry()
            .from_hash(hash, |k| *k == 1)
            .map(|(k, v)| (*k, *v));
        assert_eq!(kv, Some((1, 101)));

        assert_eq!(map.raw_entry().from_key(&999), None);
    }

    #[test]
    fn raw_entry_mut_insert_and_modify() {
        let mut map: PoMap<u64, u64> = PoMap::new();

        match map.raw_entry_mut().from_key(&1) {
            RawEntryMut::Vacant(entry) => {
                let (k, v) = entry.insert(1, 10);
                assert_eq!((*k, *v), (1, 10));
            }
            RawEntryMut::Occupied(_) => panic!("expected vacant entry"),
        }
        assert_eq!(map.get(&1), Some(&10));

        match map.raw_entry_mut().from_key(&1) {
            RawEntryMut::Occupied(mut entry) => {
                assert_eq!(entry.get(), &10);
                entry.insert(11);
            }
            RawEntryMut::Vacant(_) => panic!("expected occupied entry"),
        }
        assert_eq!(map.get(&1), Some(&11));

        let hash = raw_hash(&2u64);
        let (k, v) = map
            .raw_entry_mut()
            .from_key_hashed_nocheck(hash, &2)
            .or_insert(2, 20);
        assert_eq!((*k, *v), (2, 20));
        *v = 25;

        let _ = map.raw_entry_mut().from_key(&2).and_modify(|k, v| *v += *k);
        assert_eq!(map.get(&2), Some(&27));

        let hash = raw_hash(&1u64);
        match map.raw_entry_mut().from_hash(hash, |k| *k == 1) {
            RawEntryMut::Occupied(entry) => {
                let (k, v) = entry.remove_entry();
                assert_eq!((k, v), (1, 11));
            }
            RawEntryMut::Vacant(_) => panic!("expected occupied entry"),
        }
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), Some(&27));
    }

    #[test]
    fn entry_or_insert_and_modify() {
        let mut map: PoMap<u64, u64> = PoMap::new();

        let v = map.entry(1).or_insert(10);
        assert_eq!(*v, 10);
        *v = 11;

        let v = map.entry(1).or_insert(99);
        assert_eq!(*v, 11);
        assert_eq!(map.len(), 1);

        let v = map.entry(1).and_modify(|val| *val += 1).or_insert(0);
        assert_eq!(*v, 12);
        assert_eq!(map.get(&1), Some(&12));
    }

    #[test]
    fn entry_or_insert_with_key_uses_key() {
        let mut map: PoMap<u64, u64> = PoMap::new();
        let v = map.entry(7).or_insert_with_key(|k| k + 1);
        assert_eq!(*v, 8);
        assert_eq!(map.get(&7), Some(&8));
    }

    #[test]
    fn entry_remove_entry_returns_pair() {
        let mut map: PoMap<u64, u64> = PoMap::new();
        map.insert(10, 20);

        match map.entry(10) {
            Entry::Occupied(entry) => {
                let (k, v) = entry.remove_entry();
                assert_eq!((k, v), (10, 20));
            }
            Entry::Vacant(_) => panic!("expected occupied entry"),
        }

        assert!(map.is_empty());
    }

    #[test]
    fn insertion_shifts_to_keep_hash_order() {
        let mut map: PoMap<u64, u64> = PoMap::new();
        let keys = find_keys_with_same_slot(&map.meta, 2);
        let ordered = order_by_hash_then_key(&keys);
        let lower = ordered[0];
        let higher = ordered[1];

        assert_eq!(map.insert(higher, 1), None);
        assert_eq!(map.insert(lower, 2), None);

        let items = map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        assert_eq!(items, vec![(lower, 2), (higher, 1)]);
        assert_eq!(map.get(&higher), Some(&1));
        assert_eq!(map.get(&lower), Some(&2));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn mass_inserts_and_gets() {
        let total = 1_000_000usize;
        let mut map: PoMap<u64, u64> = PoMap::new();
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
        let mut map: PoMap<u64, DropCounter> = PoMap::new();

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

    #[test]
    fn remove_returns_value_and_clears_slot() {
        let mut map: PoMap<u64, u64> = PoMap::new();
        assert_eq!(map.remove(&123), None);

        assert_eq!(map.insert(42, 7), None);
        assert_eq!(map.remove(&42), Some(7));
        assert_eq!(map.get(&42), None);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn remove_backshifts_contiguous_run() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(8);
        let keys = find_keys_with_same_slot(&map.meta, 3);
        let ordered = order_by_hash_then_key(&keys);
        let k1 = ordered[0];
        let k2 = ordered[1];
        let k3 = ordered[2];

        map.insert(k1, 10);
        map.insert(k2, 20);
        map.insert(k3, 30);

        assert_eq!(map.remove(&k1), Some(10));

        let remaining_keys = map.keys().copied().collect::<Vec<_>>();
        assert_eq!(remaining_keys, vec![k2, k3]);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&k2), Some(&20));
        assert_eq!(map.get(&k3), Some(&30));
    }

    #[test]
    fn remove_compacts_middle_of_run() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(8);
        let keys = find_keys_with_same_slot(&map.meta, 3);
        let ordered = order_by_hash_then_key(&keys);
        let k1 = ordered[0];
        let k2 = ordered[1];
        let k3 = ordered[2];

        map.insert(k1, 10);
        map.insert(k2, 20);
        map.insert(k3, 30);

        assert_eq!(map.remove(&k2), Some(20));

        let remaining_keys = map.keys().copied().collect::<Vec<_>>();
        assert_eq!(remaining_keys, vec![k1, k3]);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&k1), Some(&10));
        assert_eq!(map.get(&k3), Some(&30));
    }

    #[test]
    fn remove_does_not_shift_higher_ideal_slot() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(8);
        let (k1, k2) = find_keys_distinct_slots(&map.meta);

        let slot1 = map.meta.ideal_slot(encoded_hash(&k1));
        let slot2 = map.meta.ideal_slot(encoded_hash(&k2));
        assert!(slot1 < slot2);

        map.insert(k1, 10);
        map.insert(k2, 20);

        assert_eq!(map.remove(&k1), Some(10));
        assert_eq!(map.get(&k2), Some(&20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_with_hash_and_collisions() {
        let mut map: PoMap<CollisionKey, u64> = PoMap::with_capacity(8);
        let k1 = CollisionKey(1);
        let k3 = CollisionKey(3);
        let k5 = CollisionKey(5);

        map.insert(k1.clone(), 10);
        map.insert(k3.clone(), 30);
        map.insert(k5.clone(), 50);

        assert_eq!(map.remove(&CollisionKey(2)), None);
        assert_eq!(map.len(), 3);

        let hash = raw_hash(&k3);
        assert_eq!(map.remove_with_hash(hash, &k3), Some(30));
        assert_eq!(map.get(&k3), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&k1), Some(&10));
        assert_eq!(map.get(&k5), Some(&50));
    }

    #[test]
    fn remove_drops_key_and_returns_value_without_drop() {
        let key_drops = Rc::new(Cell::new(0));
        let value_drops = Rc::new(Cell::new(0));
        let key = DropKey {
            id: 7,
            drops: key_drops.clone(),
        };
        let value = DropCounter {
            drops: value_drops.clone(),
        };

        let mut map: PoMap<DropKey, DropCounter> = PoMap::new();
        map.insert(key.clone(), value);

        let removed = map.remove(&key).expect("expected value to be removed");
        assert_eq!(key_drops.get(), 1);
        assert_eq!(value_drops.get(), 0);

        drop(map);
        assert_eq!(key_drops.get(), 1);
        assert_eq!(value_drops.get(), 0);

        drop(removed);
        assert_eq!(value_drops.get(), 1);

        drop(key);
        assert_eq!(key_drops.get(), 2);
    }

    #[test]
    fn shrink_to_reduces_capacity_and_preserves_entries() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(256);

        for i in 0..8u64 {
            let key = (i << 61) | i;
            map.insert(key, i * 10);
        }

        let old_capacity = map.capacity();
        let new_capacity = map.shrink_to(4);
        let (_, expected_capacity) = PoMapMeta::new(8);

        assert!(new_capacity < old_capacity);
        assert!(new_capacity >= expected_capacity);
        assert_eq!(map.len(), 8);

        for i in 0..8u64 {
            let key = (i << 61) | i;
            assert_eq!(map.get(&key), Some(&(i * 10)));
        }
    }

    #[test]
    fn shrink_to_sizes_up_on_scan_overflow() {
        let mut map: PoMap<CollisionKey, u64> = PoMap::with_capacity(64);
        let k1 = CollisionKey(1);
        let k2 = CollisionKey(2);
        let k3 = CollisionKey(3);
        map.insert(k1.clone(), 10);
        map.insert(k2.clone(), 20);
        map.insert(k3.clone(), 30);

        let old_capacity = map.capacity();
        let (base_meta, _) = PoMapMeta::new(4);
        let bumped_ideal_range = (1usize << base_meta.index_bits) * 2;
        let (_, expected_capacity) = PoMapMeta::new(bumped_ideal_range);

        let new_capacity = map.shrink_to(4);
        assert!(new_capacity < old_capacity);
        assert_eq!(new_capacity, expected_capacity);

        assert_eq!(map.get(&k1), Some(&10));
        assert_eq!(map.get(&k2), Some(&20));
        assert_eq!(map.get(&k3), Some(&30));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn shrink_to_empty_is_min_capacity() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(128);
        let old_capacity = map.capacity();
        let (_, expected_capacity) = PoMapMeta::new(MIN_CAPACITY);

        let new_capacity = map.shrink_to(0);

        assert!(new_capacity < old_capacity);
        assert_eq!(new_capacity, expected_capacity);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn shrink_to_honors_len_when_min_is_smaller() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(64);
        let keys = [
            1u64,
            (1u64 << 61) + 1,
            (2u64 << 61) + 1,
            (3u64 << 61) + 1,
            (4u64 << 61) + 1,
            (5u64 << 61) + 1,
        ];
        for (i, &key) in keys.iter().enumerate() {
            map.insert(key, i as u64);
        }

        let old_capacity = map.capacity();
        let new_capacity = map.shrink_to(1);
        let (_, expected_capacity) = PoMapMeta::new(6usize.next_power_of_two());
        assert!(new_capacity >= expected_capacity);
        assert!(new_capacity <= old_capacity);

        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(map.get(&key), Some(&(i as u64)));
        }
    }

    #[test]
    fn shrink_to_noop_when_target_not_smaller() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(32);
        for i in 0..4u64 {
            map.insert(i << 60, i);
        }
        let old_capacity = map.capacity();

        let new_capacity = map.shrink_to(old_capacity);

        assert_eq!(new_capacity, old_capacity);
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn shrink_to_fit_uses_len() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(128);
        for i in 0..5u64 {
            let key = (i << 61) | i;
            map.insert(key, i);
        }

        let old_capacity = map.capacity();
        let (_, expected_capacity) = PoMapMeta::new(5usize.next_power_of_two());
        let new_capacity = map.shrink_to_fit();

        assert!(new_capacity < old_capacity);
        assert_eq!(new_capacity, expected_capacity);
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn shrink_to_returns_current_when_repack_still_overflows() {
        let mut map: PoMap<CollisionKey, u64> = PoMap::with_capacity(16);
        for i in 0..4u64 {
            map.insert(CollisionKey(i), i * 10);
        }

        let old_capacity = map.capacity();
        let new_capacity = map.shrink_to(4);

        assert_eq!(new_capacity, old_capacity);
        for i in 0..4u64 {
            let key = CollisionKey(i);
            assert_eq!(map.get(&key), Some(&(i * 10)));
        }
    }

    #[test]
    fn mass_inserts_and_removes_match_hashmap() {
        let total = 200_000usize;
        let mut map: PoMap<u64, u64> = PoMap::new();
        let mut expected: HashMap<u64, u64> = HashMap::new();

        let mut rng = StdRng::seed_from_u64(0xFEEDBEEF);
        for _ in 0..total {
            let key: u64 = rng.random();
            let val: u64 = rng.random();
            expected.insert(key, val);
            map.insert(key, val);
        }

        let mut remove_rng = StdRng::seed_from_u64(0xBADBEEF);
        for _ in 0..total {
            let key: u64 = remove_rng.random();
            let expected_val = expected.remove(&key);
            let actual_val = map.remove(&key);
            assert_eq!(actual_val, expected_val);
        }

        assert_eq!(map.len(), expected.len());
        for (key, val) in expected.iter() {
            assert_eq!(map.get(key), Some(val));
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(u64, u64),
        Remove(u64),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (any::<u64>(), any::<u64>()).prop_map(|(k, v)| Op::Insert(k, v)),
            any::<u64>().prop_map(Op::Remove),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_deterministic_across_orders_and_capacities(
            map in prop::collection::btree_map(0u64..1_000_000, any::<u64>(), 0..32),
            cap_offset in 0usize..8,
        ) {
            let entries = map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            let mut reversed = entries.clone();
            reversed.reverse();

            let cap_a = entries.len().saturating_add(1 + cap_offset).max(1);
            let cap_b = entries.len().saturating_mul(4).saturating_add(3).max(1);

            let mut map_a: PoMap<u64, u64> = PoMap::with_capacity(cap_a);
            for (k, v) in entries.iter() {
                map_a.insert(*k, *v);
            }

            let mut map_b: PoMap<u64, u64> = PoMap::with_capacity(cap_b);
            for (k, v) in reversed.iter() {
                map_b.insert(*k, *v);
            }

            let a_items = map_a.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
            let b_items = map_b.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();

            prop_assert_eq!(a_items, b_items);
            prop_assert_eq!(&map_a, &map_b);
            prop_assert_eq!(map_a.cmp(&map_b), Ordering::Equal);
        }

        #[test]
        fn prop_collision_keys_iterate_in_key_order(
            map in prop::collection::btree_map(0u64..10_000, any::<u64>(), 0..10),
        ) {
            let entries = map.iter().map(|(k, v)| (CollisionKey(*k), *v)).collect::<Vec<_>>();
            let mut reversed = entries.clone();
            reversed.reverse();

            let len = entries.len().max(1);
            let cap_small = 1usize << len;
            let cap_large = cap_small.saturating_mul(4);

            let mut map_a: PoMap<CollisionKey, u64> = PoMap::with_capacity(cap_small);
            for (k, v) in entries.iter().cloned() {
                map_a.insert(k, v);
            }

            let mut map_b: PoMap<CollisionKey, u64> = PoMap::with_capacity(cap_large);
            for (k, v) in reversed.iter().cloned() {
                map_b.insert(k, v);
            }

            let a_keys = map_a.keys().map(|k| k.0).collect::<Vec<_>>();
            let b_keys = map_b.keys().map(|k| k.0).collect::<Vec<_>>();
            let mut sorted_keys = a_keys.clone();
            sorted_keys.sort();

            prop_assert_eq!(&a_keys, &b_keys);
            prop_assert_eq!(&a_keys, &sorted_keys);
        }

        #[test]
        fn prop_matches_std_hashmap_for_ops(ops in prop::collection::vec(op_strategy(), 0..128)) {
            let mut map: PoMap<u64, u64> = PoMap::new();
            let mut expected: HashMap<u64, u64> = HashMap::new();
            let mut touched: BTreeSet<u64> = BTreeSet::new();

            for op in ops {
                match op {
                    Op::Insert(k, v) => {
                        map.insert(k, v);
                        expected.insert(k, v);
                        touched.insert(k);
                    }
                    Op::Remove(k) => {
                        map.remove(&k);
                        expected.remove(&k);
                        touched.insert(k);
                    }
                }
            }

            prop_assert_eq!(map.len(), expected.len());
            for key in touched {
                prop_assert_eq!(map.get(&key), expected.get(&key));
                prop_assert_eq!(map.contains_key(&key), expected.contains_key(&key));
            }
        }
    }
}
