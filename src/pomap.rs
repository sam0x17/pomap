//! Prefix-ordered hash map with overlapping SIMD scan windows.
//!
//! Flat layout: `[tags: u8 × total_slots] [entries: (K,V) × total_slots]`
//! Each entry maps to an ideal slot based on its hash prefix. Entries are
//! globally sorted by hash. When the ideal slot is occupied, entries shift
//! right to the nearest EMPTY slot. SIMD scan from ideal slot for get.
//! Grows at global load factor (75%), not per-bucket overflow.

use ahash::AHasher;
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::{
    alloc::Layout,
    fmt,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ops::Index,
    ptr::{self, NonNull},
};
use wide::u8x16;

const SCAN_WIDTH: usize = 16;
const EMPTY: u8 = 0x80;
const MIN_IDEAL_RANGE: usize = 16;

/// Padding slots beyond ideal_range to handle displacement at 75% load.
/// 10 × log2(n) covers the worst case empirically; capped at ir/4 to
/// keep overhead ≤25% at small sizes. The insert safety check triggers
/// a grow if ever exceeded (correctness backstop).
#[inline]
const fn padding_for(ideal_range: usize) -> usize {
    let log2 = (usize::BITS - ideal_range.leading_zeros()) as usize;
    let p = log2 * 10;
    let cap = ideal_range / 4;
    if p < cap { p } else { cap }
}

/// Default build hasher for [`PoMap`], backed by [`AHasher`].
#[derive(Clone, Default)]
pub struct PoMapBuildHasher;

impl BuildHasher for PoMapBuildHasher {
    type Hasher = AHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        AHasher::default()
    }
}

/// The error type returned by [`PoMap::try_reserve`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TryReserveError {
    /// Error due to the computed capacity exceeding the collection's maximum.
    CapacityOverflow,
    /// The memory allocator returned an error.
    AllocError {
        /// The layout of the allocation request that failed.
        layout: Layout,
    },
}

/// Marker trait for keys stored in a [`PoMap`].
pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

/// Marker trait for values stored in a [`PoMap`].
pub trait Value: Clone {}
impl<V: Clone> Value for V {}

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    h.saturating_sub(1)
}

/// Tag = 7-bit fingerprint from hash bits below the ideal-slot prefix.
#[inline(always)]
const fn make_tag(hash: u64, tag_shift: usize) -> u8 {
    ((hash >> tag_shift) & 0x7F) as u8
}

// ---------------------------------------------------------------------------
// Slots: the raw allocation holding tags + entries
// ---------------------------------------------------------------------------

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

        Self {
            ptr,
            total_slots,
            tags,
            entries,
            layout,
            _marker: PhantomData,
        }
    }

    /// Try to allocate; returns `Err` on allocation failure instead of aborting.
    fn try_new(total_slots: usize) -> Result<Self, TryReserveError> {
        let tags_layout =
            Layout::array::<u8>(total_slots).map_err(|_| TryReserveError::CapacityOverflow)?;
        let entries_layout = Layout::array::<MaybeUninit<(K, V)>>(total_slots)
            .map_err(|_| TryReserveError::CapacityOverflow)?;
        let (layout, entries_offset) = tags_layout
            .extend(entries_layout)
            .map_err(|_| TryReserveError::CapacityOverflow)?;
        let layout = layout.pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => return Err(TryReserveError::AllocError { layout }),
        };

        let tags = ptr.as_ptr() as *mut u8;
        let entries = unsafe { ptr.as_ptr().add(entries_offset) as *mut MaybeUninit<(K, V)> };
        unsafe { ptr::write_bytes(tags, EMPTY, total_slots) };

        Ok(Self {
            ptr,
            total_slots,
            tags,
            entries,
            layout,
            _marker: PhantomData,
        })
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

// SAFETY: If K and V are Send, the raw pointers are derived from a private
// allocation and only accessed through &self / &mut self.
unsafe impl<K: Key + Send, V: Value + Send> Send for Slots<K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Slots<K, V> {}

// ---------------------------------------------------------------------------
// Meta: hash → slot / tag mapping
// ---------------------------------------------------------------------------

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
        Self {
            ideal_range,
            slot_shift,
            tag_shift,
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

    #[inline(always)]
    fn tag(&self, hash: u64) -> u8 {
        make_tag(hash, self.tag_shift)
    }
}

// ---------------------------------------------------------------------------
// PoMap
// ---------------------------------------------------------------------------

/// A prefix-ordered hash map with deterministic iteration order.
///
/// Entries are stored in a flat array sorted by hash, enabling SIMD-accelerated
/// lookups and deterministic iteration regardless of insertion order.
pub struct PoMap<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher> {
    len: usize,
    meta: Meta,
    slots: Slots<K, V>,
    hash_builder: H,
}

// SAFETY: PoMap's raw pointers live inside Slots which has its own impls.
unsafe impl<K: Key + Send, V: Value + Send, H: BuildHasher + Send> Send for PoMap<K, V, H> {}
unsafe impl<K: Key + Sync, V: Value + Sync, H: BuildHasher + Sync> Sync for PoMap<K, V, H> {}

impl<K: Key, V: Value> PoMap<K, V, PoMapBuildHasher> {
    /// Creates an empty `PoMap` with the default hasher.
    #[inline]
    pub fn new() -> Self {
        Self::with_hasher(PoMapBuildHasher)
    }

    /// Creates an empty `PoMap` with the specified capacity and the default hasher.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, PoMapBuildHasher)
    }
}

impl<K: Key, V: Value, H: BuildHasher + Default> Default for PoMap<K, V, H> {
    #[inline]
    fn default() -> Self {
        Self::with_hasher(H::default())
    }
}

impl<K: Key, V: Value, H: BuildHasher> PoMap<K, V, H> {
    /// Creates an empty `PoMap` which will use the given hash builder.
    #[inline]
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_IDEAL_RANGE, hash_builder)
    }

    /// Creates an empty `PoMap` with the specified capacity and hash builder.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let ideal_range = capacity.next_power_of_two().max(MIN_IDEAL_RANGE);
        let total_slots = ideal_range + padding_for(ideal_range);
        Self {
            len: 0,
            meta: Meta::new(ideal_range),
            slots: Slots::new(total_slots),
            hash_builder,
        }
    }

    /// Returns a reference to the map's [`BuildHasher`].
    #[inline]
    pub fn hasher(&self) -> &H {
        &self.hash_builder
    }

    /// Returns the number of elements in the map.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the map can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        // Usable capacity is 75% of ideal_range.
        self.meta.ideal_range * 3 / 4
    }

    #[inline(always)]
    fn load_tags(&self, start: usize) -> u8x16 {
        let ptr = unsafe { self.slots.tags.add(start) };
        u8x16::new(unsafe { ptr::read_unaligned(ptr as *const [u8; 16]) })
    }

    /// Returns a reference to the value corresponding to `key`.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        // Scalar fast path: check ideal slot directly (most common case).
        let ideal_tag = unsafe { *self.slots.tags.add(ideal) };
        if ideal_tag == tag {
            let (k, v) = unsafe { &*(*self.slots.entries.add(ideal)).as_ptr() };
            if k == key {
                return Some(v);
            }
        } else if ideal_tag & 0x80 != 0 {
            return None;
        }

        // SIMD scan from ideal slot.
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
            if tags_vec.move_mask() as u32 & 0xFFFF != 0 {
                return None;
            }
            pos += SCAN_WIDTH;
            if pos >= self.slots.total_slots {
                return None;
            }
        }
    }

    /// Returns a mutable reference to the value corresponding to `key`.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);

        // Scalar fast path: check ideal slot directly.
        let ideal_tag = unsafe { *self.slots.tags.add(ideal) };
        if ideal_tag == tag {
            let entry = unsafe { &mut *(*self.slots.entries.add(ideal)).as_mut_ptr() };
            if entry.0 == *key {
                return Some(&mut entry.1);
            }
        } else if ideal_tag & 0x80 != 0 {
            return None;
        }

        // SIMD scan.
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

    /// Returns a reference to the key-value pair corresponding to `key`.
    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
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
                let (k, v) = unsafe { &*(*self.slots.entries.add(pos + offset)).as_ptr() };
                if k == key {
                    return Some((k, v));
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

    /// Returns `true` if the map contains the given key.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Inserts a key-value pair, returning the previous value if the key was present.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Grow at 75% load.
        if self.len * 4 >= self.meta.ideal_range * 3 {
            self.grow();
        }

        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let ideal = self.meta.ideal_slot(hash);
        let tag = self.meta.tag(hash);
        let tags_ptr = self.slots.tags;
        let entries_ptr = self.slots.entries;

        // Fast path: ideal slot is EMPTY — write directly, no scan needed.
        if unsafe { *tags_ptr.add(ideal) } & 0x80 != 0 {
            unsafe {
                *tags_ptr.add(ideal) = tag;
                *entries_ptr.add(ideal) = MaybeUninit::new((key, value));
            }
            self.len += 1;
            return None;
        }

        self.insert_slow(key, value, hash, ideal, tag)
    }

    #[cold]
    #[inline(never)]
    fn insert_slow(
        &mut self,
        key: K,
        value: V,
        hash: u64,
        ideal: usize,
        tag: u8,
    ) -> Option<V> {
        let tags_ptr = self.slots.tags;
        let entries_ptr = self.slots.entries;

        // Combined replacement check + position finding in a single forward scan.
        let mut pos = ideal;
        loop {
            let slot_tag = unsafe { *tags_ptr.add(pos) };
            if slot_tag & 0x80 != 0 {
                // EMPTY — insert here.
                unsafe {
                    *tags_ptr.add(pos) = tag;
                    *entries_ptr.add(pos) = MaybeUninit::new((key, value));
                }
                self.len += 1;
                return None;
            }
            let incumbent = unsafe { &*(*entries_ptr.add(pos)).as_ptr() };
            let incumbent_hash = encode_hash(self.hash_builder.hash_one(&incumbent.0));
            if incumbent_hash == hash && incumbent.0 == key {
                let entry = unsafe { &mut *(*entries_ptr.add(pos)).as_mut_ptr() };
                return Some(mem::replace(&mut entry.1, value));
            }
            if incumbent_hash > hash {
                break;
            }
            pos += 1;
            if pos >= self.slots.total_slots {
                self.grow();
                return self.insert(key, value);
            }
        }

        // Find nearest EMPTY after pos for the shift target.
        let mut empty_pos = pos + 1;
        while unsafe { *tags_ptr.add(empty_pos) } & 0x80 == 0 {
            empty_pos += 1;
            if empty_pos >= self.slots.total_slots {
                self.grow();
                return self.insert(key, value);
            }
        }

        // Shift [pos, empty_pos) right by 1.
        let shift = empty_pos - pos;
        unsafe {
            ptr::copy(tags_ptr.add(pos), tags_ptr.add(pos + 1), shift);
            ptr::copy(entries_ptr.add(pos), entries_ptr.add(pos + 1), shift);
            *tags_ptr.add(pos) = tag;
            *entries_ptr.add(pos) = MaybeUninit::new((key, value));
        }
        self.len += 1;
        None
    }

    /// Removes a key from the map, returning the value if present.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the key-value pair if present.
    #[inline]
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
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
                    let (key, value) = unsafe { (*self.slots.entries.add(idx)).assume_init_read() };
                    // Backshift: pull entries left to fill the gap.
                    let tags_ptr = self.slots.tags;
                    let entries_ptr = self.slots.entries;
                    let mut hole = idx;
                    loop {
                        let next = hole + 1;
                        let next_tag = unsafe { *tags_ptr.add(next) };
                        if next_tag & 0x80 != 0 {
                            break;
                        }
                        let next_entry = unsafe { &*(*entries_ptr.add(next)).as_ptr() };
                        let next_hash = encode_hash(self.hash_builder.hash_one(&next_entry.0));
                        let next_ideal = self.meta.ideal_slot(next_hash);
                        if next_ideal > hole {
                            break;
                        }
                        unsafe {
                            *tags_ptr.add(hole) = next_tag;
                            ptr::copy_nonoverlapping(
                                entries_ptr.add(next),
                                entries_ptr.add(hole),
                                1,
                            );
                        }
                        hole = next;
                    }
                    unsafe { *tags_ptr.add(hole) = EMPTY };
                    self.len -= 1;
                    return Some((key, value));
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

    /// Clears the map, removing all key-value pairs.
    #[inline]
    pub fn clear(&mut self) {
        for i in 0..self.slots.total_slots {
            if unsafe { *self.slots.tags.add(i) } & 0x80 == 0 {
                unsafe {
                    ptr::drop_in_place((*self.slots.entries.add(i)).as_mut_ptr());
                    *self.slots.tags.add(i) = EMPTY;
                }
            }
        }
        self.len = 0;
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// The predicate is called with each key-value pair; if it returns `false`,
    /// the entry is removed.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let tags_ptr = self.slots.tags;
        let entries_ptr = self.slots.entries;
        let total = self.slots.total_slots;
        let mut i = 0;
        while i < total {
            let tag = unsafe { *tags_ptr.add(i) };
            if tag & 0x80 != 0 {
                i += 1;
                continue;
            }
            let entry = unsafe { &mut *(*entries_ptr.add(i)).as_mut_ptr() };
            if f(&entry.0, &mut entry.1) {
                i += 1;
                continue;
            }
            // Remove this entry using backshift logic.
            unsafe { ptr::drop_in_place((*entries_ptr.add(i)).as_mut_ptr()) };
            let mut hole = i;
            loop {
                let next = hole + 1;
                if next >= total {
                    break;
                }
                let next_tag = unsafe { *tags_ptr.add(next) };
                if next_tag & 0x80 != 0 {
                    break;
                }
                let next_entry = unsafe { &*(*entries_ptr.add(next)).as_ptr() };
                let next_hash = encode_hash(self.hash_builder.hash_one(&next_entry.0));
                let next_ideal = self.meta.ideal_slot(next_hash);
                if next_ideal > hole {
                    break;
                }
                unsafe {
                    *tags_ptr.add(hole) = next_tag;
                    ptr::copy_nonoverlapping(entries_ptr.add(next), entries_ptr.add(hole), 1);
                }
                hole = next;
            }
            unsafe { *tags_ptr.add(hole) = EMPTY };
            self.len -= 1;
            // Do NOT advance i — the next entry shifted into this position.
        }
    }

    /// Returns an iterator over key-value pairs in deterministic order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            tags: self.slots.tags,
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over key-value pairs in deterministic order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            tags: self.slots.tags,
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over keys in deterministic order.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator over values in deterministic order.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V> {
        Values { iter: self.iter() }
    }

    /// Returns a mutable iterator over values in deterministic order.
    #[inline]
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }

    /// Creates a consuming iterator visiting all keys in deterministic order.
    #[inline]
    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys {
            iter: self.into_iter(),
        }
    }

    /// Creates a consuming iterator visiting all values in deterministic order.
    #[inline]
    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues {
            iter: self.into_iter(),
        }
    }

    /// Drains all entries from the map, returning an iterator over them.
    ///
    /// After the drain iterator is dropped (whether fully consumed or not),
    /// the map is empty.
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        let drain = Drain {
            tags: self.slots.tags,
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            _marker: PhantomData,
        };
        self.len = 0;
        drain
    }

    /// Shrinks the capacity of the map as much as possible while maintaining
    /// at least `min_capacity` usable slots.
    ///
    /// The actual resulting capacity may be larger than `min_capacity` due to
    /// power-of-two sizing. No-op if the map is already at or below the target.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let target_len = self.len.max(min_capacity);
        // Need ideal_range such that 75% >= target_len → ideal_range >= ceil(target_len * 4 / 3)
        let needed_ideal = ((target_len * 4 + 2) / 3)
            .next_power_of_two()
            .max(MIN_IDEAL_RANGE);
        if needed_ideal >= self.meta.ideal_range {
            return;
        }
        self.rebuild(needed_ideal);
    }

    /// Shrinks the capacity of the map as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(self.len);
    }

    /// Reserves capacity for at least `additional` more elements.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        if needed <= self.capacity() {
            return;
        }
        // Need ideal_range such that 75% >= needed
        let needed_ideal = ((needed * 4 + 2) / 3)
            .next_power_of_two()
            .max(MIN_IDEAL_RANGE);
        if needed_ideal > self.meta.ideal_range {
            self.rebuild(needed_ideal);
        }
    }

    /// Tries to reserve capacity for at least `additional` more elements.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let needed = self
            .len
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;
        if needed <= self.capacity() {
            return Ok(());
        }
        let needed_ideal = needed
            .checked_mul(4)
            .and_then(|n| n.checked_add(2))
            .ok_or(TryReserveError::CapacityOverflow)?
            / 3;
        let needed_ideal = needed_ideal
            .checked_next_power_of_two()
            .ok_or(TryReserveError::CapacityOverflow)?
            .max(MIN_IDEAL_RANGE);
        if needed_ideal > self.meta.ideal_range {
            self.try_rebuild(needed_ideal)?;
        }
        Ok(())
    }

    fn grow(&mut self) {
        let new_ideal_range = self.meta.ideal_range * 2;
        self.rebuild(new_ideal_range);
    }

    fn rebuild(&mut self, new_ideal_range: usize) {
        let new_meta = Meta::new(new_ideal_range);
        let new_total = new_ideal_range + padding_for(new_ideal_range);
        let new_slots = Slots::new(new_total);

        self.transfer_to(&new_meta, &new_slots);

        let old_slots = mem::replace(&mut self.slots, new_slots);
        unsafe { dealloc(old_slots.ptr.as_ptr(), old_slots.layout) };
        mem::forget(old_slots);
        self.meta = new_meta;
    }

    fn try_rebuild(&mut self, new_ideal_range: usize) -> Result<(), TryReserveError> {
        let new_meta = Meta::new(new_ideal_range);
        let new_total = new_ideal_range + padding_for(new_ideal_range);
        let new_slots = Slots::try_new(new_total)?;

        self.transfer_to(&new_meta, &new_slots);

        let old_slots = mem::replace(&mut self.slots, new_slots);
        unsafe { dealloc(old_slots.ptr.as_ptr(), old_slots.layout) };
        mem::forget(old_slots);
        self.meta = new_meta;
        Ok(())
    }

    /// Transfer all live entries from current slots into `new_slots` using `new_meta`.
    /// Marks old tags as EMPTY after moving. Caller must skip the Drop of old entries.
    fn transfer_to(&mut self, new_meta: &Meta, new_slots: &Slots<K, V>) {
        // Old entries are sorted by hash → new ideal_slots are monotonically
        // non-decreasing. Track last write position to avoid redundant scanning.
        let mut write_pos = 0usize;
        for i in 0..self.slots.total_slots {
            let old_tag = unsafe { *self.slots.tags.add(i) };
            if old_tag & 0x80 != 0 {
                continue;
            }

            let (key, value) = unsafe { (*self.slots.entries.add(i)).assume_init_read() };
            unsafe { *self.slots.tags.add(i) = EMPTY };

            let hash = encode_hash(self.hash_builder.hash_one(&key));
            let new_ideal = new_meta.ideal_slot(hash);
            let new_tag = new_meta.tag(hash);

            if new_ideal > write_pos {
                write_pos = new_ideal;
            }
            while unsafe { *new_slots.tags.add(write_pos) } & 0x80 == 0 {
                write_pos += 1;
            }
            unsafe {
                *new_slots.tags.add(write_pos) = new_tag;
                *new_slots.entries.add(write_pos) = MaybeUninit::new((key, value));
            }
            write_pos += 1;
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher> Drop for PoMap<K, V, H> {
    fn drop(&mut self) {
        // Slots::drop handles element + allocation cleanup.
    }
}

// ---------------------------------------------------------------------------
// Iterator types
// ---------------------------------------------------------------------------

/// Iterator over shared references to key-value pairs in deterministic order.
#[must_use]
pub struct Iter<'a, K: Key, V: Value> {
    tags: *const u8,
    entries: *const MaybeUninit<(K, V)>,
    index: usize,
    total_slots: usize,
    remaining: usize,
    _marker: PhantomData<&'a (K, V)>,
}

// SAFETY: Iter only reads shared references derived from an immutable borrow.
unsafe impl<K: Key + Sync, V: Value + Sync> Send for Iter<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Iter<'_, K, V> {}

impl<K: Key, V: Value> Clone for Iter<'_, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            tags: self.tags,
            entries: self.entries,
            index: self.index,
            total_slots: self.total_slots,
            remaining: self.remaining,
            _marker: PhantomData,
        }
    }
}

impl<'a, K: Key, V: Value> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.total_slots {
            let idx = self.index;
            self.index += 1;
            if unsafe { *self.tags.add(idx) } & 0x80 != 0 {
                continue;
            }
            self.remaining -= 1;
            let (k, v) = unsafe { &*(*self.entries.add(idx)).as_ptr() };
            return Some((k, v));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K: Key, V: Value> ExactSizeIterator for Iter<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: Key, V: Value> FusedIterator for Iter<'_, K, V> {}

/// Iterator over mutable references to key-value pairs in deterministic order.
#[must_use]
pub struct IterMut<'a, K: Key, V: Value> {
    tags: *const u8,
    entries: *mut MaybeUninit<(K, V)>,
    index: usize,
    total_slots: usize,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: IterMut yields &K and &mut V from a unique borrow of the map.
unsafe impl<K: Key + Send, V: Value + Send> Send for IterMut<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for IterMut<'_, K, V> {}

impl<'a, K: Key, V: Value> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.total_slots {
            let idx = self.index;
            self.index += 1;
            if unsafe { *self.tags.add(idx) } & 0x80 != 0 {
                continue;
            }
            self.remaining -= 1;
            let entry = unsafe { &mut *(*self.entries.add(idx)).as_mut_ptr() };
            return Some((&entry.0, &mut entry.1));
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K: Key, V: Value> ExactSizeIterator for IterMut<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: Key, V: Value> FusedIterator for IterMut<'_, K, V> {}

/// Iterator over shared references to keys in deterministic order.
#[must_use]
pub struct Keys<'a, K: Key, V: Value> {
    iter: Iter<'a, K, V>,
}

impl<K: Key, V: Value> Clone for Keys<'_, K, V> {
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

impl<K: Key, V: Value> ExactSizeIterator for Keys<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Key, V: Value> FusedIterator for Keys<'_, K, V> {}

/// Iterator over shared references to values in deterministic order.
#[must_use]
pub struct Values<'a, K: Key, V: Value> {
    iter: Iter<'a, K, V>,
}

impl<K: Key, V: Value> Clone for Values<'_, K, V> {
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

impl<K: Key, V: Value> ExactSizeIterator for Values<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Key, V: Value> FusedIterator for Values<'_, K, V> {}

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

impl<K: Key, V: Value> ExactSizeIterator for ValuesMut<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K: Key, V: Value> FusedIterator for ValuesMut<'_, K, V> {}

/// Owning iterator over key-value pairs in deterministic order.
#[must_use]
pub struct IntoIter<K: Key, V: Value> {
    slots: Slots<K, V>,
    index: usize,
    remaining: usize,
}

impl<K: Key, V: Value> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.slots.total_slots {
            let idx = self.index;
            self.index += 1;
            let tag = unsafe { *self.slots.tags.add(idx) };
            if tag & 0x80 != 0 {
                continue;
            }
            self.remaining -= 1;
            unsafe { *self.slots.tags.add(idx) = EMPTY };
            return Some(unsafe { (*self.slots.entries.add(idx)).assume_init_read() });
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
///
/// When dropped, any remaining entries are consumed and dropped.
#[must_use]
pub struct Drain<'a, K: Key, V: Value> {
    tags: *mut u8,
    entries: *mut MaybeUninit<(K, V)>,
    index: usize,
    total_slots: usize,
    remaining: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: Drain has exclusive access derived from &mut PoMap.
unsafe impl<K: Key + Send, V: Value + Send> Send for Drain<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Drain<'_, K, V> {}

impl<'a, K: Key, V: Value> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.total_slots {
            let idx = self.index;
            self.index += 1;
            let tag = unsafe { *self.tags.add(idx) };
            if tag & 0x80 != 0 {
                continue;
            }
            self.remaining -= 1;
            unsafe { *self.tags.add(idx) = EMPTY };
            return Some(unsafe { (*self.entries.add(idx)).assume_init_read() });
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K: Key, V: Value> ExactSizeIterator for Drain<'_, K, V> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K: Key, V: Value> FusedIterator for Drain<'_, K, V> {}

impl<K: Key, V: Value> Drop for Drain<'_, K, V> {
    fn drop(&mut self) {
        // Consume remaining entries so they are dropped.
        while self.index < self.total_slots {
            let idx = self.index;
            self.index += 1;
            let tag = unsafe { *self.tags.add(idx) };
            if tag & 0x80 != 0 {
                continue;
            }
            unsafe {
                *self.tags.add(idx) = EMPTY;
                ptr::drop_in_place((*self.entries.add(idx)).as_mut_ptr());
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

// ---------------------------------------------------------------------------
// IntoIterator impls
// ---------------------------------------------------------------------------

impl<'a, K: Key, V: Value, H: BuildHasher> IntoIterator for &'a PoMap<K, V, H> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V: Value, H: BuildHasher> IntoIterator for &'a mut PoMap<K, V, H> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V: Value, H: BuildHasher> IntoIterator for PoMap<K, V, H> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let total_slots = self.slots.total_slots;
        let remaining = self.len;
        // We need to prevent PoMap::drop from running (which would run Slots::drop),
        // but we need the Slots to live on in IntoIter. Use ManuallyDrop-like trick:
        // read out fields and forget self.
        let slots_ptr = self.slots.ptr;
        let tags = self.slots.tags;
        let entries = self.slots.entries;
        let layout = self.slots.layout;

        // Reconstruct a Slots that owns the allocation.
        let owned_slots = Slots {
            ptr: slots_ptr,
            total_slots,
            tags,
            entries,
            layout,
            _marker: PhantomData,
        };

        // Prevent double-free: forget the original.
        mem::forget(self);

        IntoIter {
            slots: owned_slots,
            index: 0,
            remaining,
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls: Clone, Debug, PartialEq, Eq, Index, Extend, FromIterator
// ---------------------------------------------------------------------------

impl<K: Key, V: Value, H: BuildHasher + Clone> Clone for PoMap<K, V, H> {
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
            meta: Meta {
                ideal_range: self.meta.ideal_range,
                slot_shift: self.meta.slot_shift,
                tag_shift: self.meta.tag_shift,
            },
            slots: new_slots,
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K: Key + fmt::Debug, V: Value + fmt::Debug, H: BuildHasher> fmt::Debug for PoMap<K, V, H> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (k, v) in self.iter() {
            map.entry(k, v);
        }
        map.finish()
    }
}

impl<K: Key, V: Value + PartialEq, H: BuildHasher> PartialEq for PoMap<K, V, H> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<K: Key, V: Value + Eq, H: BuildHasher> Eq for PoMap<K, V, H> {}

impl<K: Key, V: Value, H: BuildHasher> Index<&K> for PoMap<K, V, H> {
    type Output = V;

    #[inline]
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("key not found")
    }
}

impl<K: Key, V: Value, H: BuildHasher> Extend<(K, V)> for PoMap<K, V, H> {
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: BuildHasher> Extend<(&'a K, &'a V)> for PoMap<K, V, H> {
    #[inline]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key.clone(), value.clone());
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher + Default> FromIterator<(K, V)> for PoMap<K, V, H> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = PoMap::with_capacity_and_hasher(lower, H::default());
        map.extend(iter);
        map
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: BuildHasher + Default> FromIterator<(&'a K, &'a V)>
    for PoMap<K, V, H>
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (&'a K, &'a V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = PoMap::with_capacity_and_hasher(lower, H::default());
        map.extend(iter);
        map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHasher;
    use std::collections::HashMap;
    use std::hash::BuildHasherDefault;

    type TestMap = PoMap<u64, u64, BuildHasherDefault<AHasher>>;
    fn new_map() -> TestMap {
        PoMap::with_hasher(BuildHasherDefault::default())
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
    fn remove_basic() {
        let mut map = new_map();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn remove_entry_basic() {
        let mut map = new_map();
        map.insert(1, 10);
        assert_eq!(map.remove_entry(&1), Some((1, 10)));
        assert_eq!(map.remove_entry(&1), None);
        assert!(map.is_empty());
    }

    #[test]
    fn grow_basic() {
        let mut map = new_map();
        for i in 0..100u64 {
            map.insert(i, i * 10);
            for j in 0..=i {
                assert_eq!(
                    map.get(&j),
                    Some(&(j * 10)),
                    "missing key {} after inserting {} (len={})",
                    j,
                    i,
                    map.len()
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
                    assert_eq!(
                        map.insert(key, val),
                        expected.insert(key, val),
                        "insert mismatch for key {}",
                        key
                    );
                }
                1 => {
                    assert_eq!(
                        map.get(&key).copied(),
                        expected.get(&key).copied(),
                        "get mismatch for key {}",
                        key
                    );
                }
                _ => {
                    assert_eq!(
                        map.remove(&key),
                        expected.remove(&key),
                        "remove mismatch for key {}",
                        key
                    );
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
    fn resize_preserves() {
        let mut map = new_map();
        for i in 0..1000u64 {
            map.insert(i, i);
        }
        for i in 0..1000u64 {
            assert_eq!(map.get(&i), Some(&i), "missing key {} after resizes", i);
        }
    }

    // ---- New API tests ----

    #[test]
    fn contains_key_basic() {
        let mut map = new_map();
        map.insert(1, 10);
        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
    }

    #[test]
    fn get_key_value_basic() {
        let mut map = new_map();
        map.insert(42, 99);
        assert_eq!(map.get_key_value(&42), Some((&42, &99)));
        assert_eq!(map.get_key_value(&0), None);
    }

    #[test]
    fn get_mut_basic() {
        let mut map = new_map();
        map.insert(1, 10);
        *map.get_mut(&1).unwrap() = 20;
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn clear_basic() {
        let mut map = new_map();
        for i in 0..50u64 {
            map.insert(i, i);
        }
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        // Can still use the map after clear.
        map.insert(1, 100);
        assert_eq!(map.get(&1), Some(&100));
    }

    #[test]
    fn retain_basic() {
        let mut map = new_map();
        for i in 0..20u64 {
            map.insert(i, i * 10);
        }
        map.retain(|k, _| k % 2 == 0);
        assert_eq!(map.len(), 10);
        for i in 0..20u64 {
            if i % 2 == 0 {
                assert_eq!(map.get(&i), Some(&(i * 10)));
            } else {
                assert_eq!(map.get(&i), None);
            }
        }
    }

    #[test]
    fn drain_basic() {
        let mut map = new_map();
        for i in 0..10u64 {
            map.insert(i, i * 10);
        }
        let expected: Vec<(u64, u64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        let drained: Vec<(u64, u64)> = map.drain().collect();
        assert_eq!(drained, expected);
        assert!(map.is_empty());
        // Can still use the map.
        map.insert(99, 99);
        assert_eq!(map.get(&99), Some(&99));
    }

    #[test]
    fn drain_drop_clears_remaining() {
        let mut map = new_map();
        for i in 0..5u64 {
            map.insert(i, i);
        }
        {
            let mut drain = map.drain();
            let _ = drain.next();
        }
        assert!(map.is_empty());
    }

    #[test]
    fn iter_mut_basic() {
        let mut map = new_map();
        for i in 0..5u64 {
            map.insert(i, i);
        }
        for (_, v) in map.iter_mut() {
            *v += 100;
        }
        for i in 0..5u64 {
            assert_eq!(map.get(&i), Some(&(i + 100)));
        }
    }

    #[test]
    fn into_iter_basic() {
        let mut map = new_map();
        for i in 0..5u64 {
            map.insert(i, i * 10);
        }
        let items: Vec<(u64, u64)> = map.into_iter().collect();
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn into_keys_into_values() {
        let mut map = new_map();
        for i in 0..5u64 {
            map.insert(i, i * 10);
        }
        let keys: Vec<u64> = map.clone().into_keys().collect();
        let values: Vec<u64> = map.into_values().collect();
        assert_eq!(keys.len(), 5);
        assert_eq!(values.len(), 5);
    }

    #[test]
    fn values_mut_basic() {
        let mut map = new_map();
        map.insert(1, 10);
        map.insert(2, 20);
        for v in map.values_mut() {
            *v *= 2;
        }
        assert_eq!(map.get(&1), Some(&20));
        assert_eq!(map.get(&2), Some(&40));
    }

    #[test]
    fn shrink_to_basic() {
        let mut map = new_map();
        for i in 0..5u64 {
            map.insert(i, i);
        }
        let old_cap = map.capacity();
        // Reserve a lot, then shrink.
        map.reserve(1000);
        assert!(map.capacity() > old_cap);
        map.shrink_to(5);
        // All entries still accessible.
        for i in 0..5u64 {
            assert_eq!(map.get(&i), Some(&i), "missing key {} after shrink", i);
        }
    }

    #[test]
    fn shrink_to_fit_basic() {
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(1024);
        for i in 0..10u64 {
            map.insert(i, i);
        }
        map.shrink_to_fit();
        for i in 0..10u64 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn reserve_basic() {
        let mut map = new_map();
        map.reserve(1000);
        assert!(map.capacity() >= 1000);
        for i in 0..1000u64 {
            map.insert(i, i);
        }
        for i in 0..1000u64 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn try_reserve_basic() {
        let mut map = new_map();
        assert!(map.try_reserve(100).is_ok());
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn index_trait() {
        let mut map = new_map();
        map.insert(5, 50);
        assert_eq!(map[&5], 50);
    }

    #[test]
    #[should_panic(expected = "key not found")]
    fn index_trait_missing() {
        let map = new_map();
        let _ = map[&1];
    }

    #[test]
    fn extend_basic() {
        let mut map = new_map();
        map.extend([(1u64, 10u64), (2, 20), (3, 30)]);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&20));
    }

    #[test]
    fn from_iterator() {
        let map: TestMap = [(1u64, 10u64), (2, 20)].into_iter().collect();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn debug_trait() {
        let mut map = new_map();
        map.insert(1, 10);
        let s = format!("{:?}", map);
        assert!(s.contains("1"));
        assert!(s.contains("10"));
    }

    #[test]
    fn partial_eq_trait() {
        let mut a = new_map();
        let mut b = new_map();
        for i in 0..10u64 {
            a.insert(i, i);
            b.insert(i, i);
        }
        assert_eq!(a, b);
        b.insert(99, 99);
        assert_ne!(a, b);
    }

    #[test]
    fn clone_basic() {
        let mut map = new_map();
        for i in 0..50u64 {
            map.insert(i, i * 10);
        }
        let cloned = map.clone();
        assert_eq!(map, cloned);
    }

    #[test]
    fn default_trait() {
        let map: PoMap<u64, u64> = PoMap::default();
        assert!(map.is_empty());
    }

    #[test]
    fn hasher_accessor() {
        let map: PoMap<u64, u64> = PoMap::new();
        let _ = map.hasher();
    }

    #[test]
    fn exact_size_iterators() {
        let mut map = new_map();
        for i in 0..10u64 {
            map.insert(i, i);
        }
        assert_eq!(map.iter().len(), 10);
        assert_eq!(map.keys().len(), 10);
        assert_eq!(map.values().len(), 10);
    }
}
