//! Prefix-ordered hash map with an inline-hash AoS layout.
//!
//! Layout: `[entries: (u64_hash, K, V) × N]` — a single flat array. Each entry
//! maps to an ideal slot derived from its hash prefix; entries are globally
//! sorted by hash, displacing rightward into the nearest vacancy on collision.
//! The inline hash shares a cache line with its (K, V), so a probe is one
//! memory access: scan from the ideal slot, filter on exact `u64` compares,
//! terminate on the first hash greater than the target (which doubles as the
//! vacancy check, since EMPTY is `u64::MAX`).
//!
//! This buys deterministic iteration order (by hash, regardless of insertion
//! order) and class-leading read performance, at the cost of an
//! order-maintenance tax on writes (runs shift to stay sorted) and a larger
//! per-slot footprint (8-byte inline hash).
//!
//! The resize repack preserves inter-run gaps via cursor spacing when growing
//! (bump the write cursor past each vacant slot in the old map), leaving ideal
//! slots open in the new map so post-grow inserts land directly. Growth
//! happens at 62.5% load; `with_capacity(n)` provisions so that `n` inserts
//! trigger zero grows.

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

const MIN_IDEAL_RANGE: usize = 16;
const EMPTY_HASH: u64 = u64::MAX;

/// Grow when `len` reaches `ideal_range * LOAD_NUM / LOAD_DEN` (62.5%). A lower
/// load factor keeps runs short, which keeps insert shifts and remove backshift
/// chains short. Mostly free in memory because `ideal_range` is a power of two.
const LOAD_NUM: usize = 5;
const LOAD_DEN: usize = 8;

/// Smallest `ideal_range` (power of two) that holds `capacity` entries without
/// growing — i.e. whose grow threshold strictly exceeds `capacity`.
#[inline]
fn ideal_range_for(capacity: usize) -> usize {
    // Need ideal_range * LOAD_NUM / LOAD_DEN > capacity.
    let needed = capacity * LOAD_DEN / LOAD_NUM + 1;
    needed.next_power_of_two().max(MIN_IDEAL_RANGE)
}

/// Checked variant of [`ideal_range_for`] for `try_reserve`.
#[inline]
fn try_ideal_range_for(capacity: usize) -> Result<usize, TryReserveError> {
    let needed = capacity
        .checked_mul(LOAD_DEN)
        .ok_or(TryReserveError::CapacityOverflow)?
        / LOAD_NUM
        + 1;
    Ok(needed
        .checked_next_power_of_two()
        .ok_or(TryReserveError::CapacityOverflow)?
        .max(MIN_IDEAL_RANGE))
}

#[inline(always)]
const fn encode_hash(h: u64) -> u64 {
    // saturating_sub(1) yields at most u64::MAX-1, so the result can never
    // collide with EMPTY_HASH (u64::MAX) — no extra guard needed.
    h.saturating_sub(1)
}

/// Padding slots beyond `ideal_range` to absorb displacement at the tail.
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
///
/// Note there is no `Ord` bound: the map is ordered by *hash*, never by key
/// comparison — no ordering operation on `K` exists anywhere in the layout,
/// probe, or resize paths (`==` is used only for the final match).
pub trait Key: Hash + Eq + Clone {}
impl<K: Hash + Eq + Clone> Key for K {}

/// Marker trait for values stored in a [`PoMap`].
pub trait Value: Clone {}
impl<V: Clone> Value for V {}

/// A slot's contents. `repr(C)` guarantees the hash lives at offset 0 so the
/// raw `u64` reads in `hash_at` (and the 0xFF vacancy memset) are layout-safe
/// for any K/V — a plain tuple's repr(Rust) field order is unspecified.
#[repr(C)]
struct Entry<K, V> {
    hash: u64,
    key: K,
    value: V,
}

// ---------------------------------------------------------------------------
// Slots: the raw allocation holding (hash, K, V) entries
// ---------------------------------------------------------------------------

struct Slots<K: Key, V: Value> {
    ptr: NonNull<u8>,
    total_slots: usize,
    entries: *mut MaybeUninit<Entry<K, V>>,
    layout: Layout,
    _marker: PhantomData<(K, V)>,
}

// SAFETY: the raw pointers are derived from a private allocation and only
// accessed through &self / &mut self.
unsafe impl<K: Key + Send, V: Value + Send> Send for Slots<K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Slots<K, V> {}

impl<K: Key, V: Value> Slots<K, V> {
    fn new(total_slots: usize) -> Self {
        match Self::try_new(total_slots) {
            Ok(slots) => slots,
            Err(TryReserveError::AllocError { layout }) => handle_alloc_error(layout),
            Err(TryReserveError::CapacityOverflow) => panic!("PoMap capacity overflow"),
        }
    }

    fn try_new(total_slots: usize) -> Result<Self, TryReserveError> {
        let layout = Layout::array::<MaybeUninit<Entry<K, V>>>(total_slots)
            .map_err(|_| TryReserveError::CapacityOverflow)?
            .pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(TryReserveError::AllocError { layout })?;

        let entries = ptr.as_ptr() as *mut MaybeUninit<Entry<K, V>>;
        // EMPTY_HASH is u64::MAX = all 0xFF bytes, so one memset over the whole
        // allocation marks every slot's hash vacant in a single pass. The K/V
        // bytes are clobbered too but are MaybeUninit and never read while
        // hash == EMPTY.
        unsafe {
            ptr::write_bytes(ptr.as_ptr(), 0xFF, layout.size());
        }

        Ok(Self {
            ptr,
            total_slots,
            entries,
            layout,
            _marker: PhantomData,
        })
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

// ---------------------------------------------------------------------------
// Meta: hash → slot mapping
// ---------------------------------------------------------------------------

struct Meta {
    /// Number of ideal slot positions (power of 2).
    ideal_range: usize,
    /// hash >> slot_shift = ideal_slot index.
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
        // slot_shift is always < 64: ideal_range >= MIN_IDEAL_RANGE (16) is a
        // power of two, so slot_bits >= 4 and slot_shift = 64 - slot_bits <= 60.
        (hash >> self.slot_shift) as usize
    }
}

// ---------------------------------------------------------------------------
// PoMap
// ---------------------------------------------------------------------------

/// A prefix-ordered hash map with deterministic iteration order.
///
/// Entries are stored in a flat array sorted by hash, giving deterministic
/// iteration regardless of insertion order and cache-friendly lookups (the
/// hash lives inline with its key-value pair).
///
/// `GROWTH` is the multiplier applied to `ideal_range` on grow and **must be a
/// power of two**: the slot mapping shifts by `trailing_zeros(ideal_range)`, so
/// a non-power-of-two factor silently degrades the table to a handful of ideal
/// slots and O(n²) inserts (enforced at compile time). Higher = fewer rebuilds
/// (~f/(f-1)·n total entry moves) but more slack memory. Measured with u64
/// keys/values vs hashbrown: GROWTH=4 → insert_allocate ~1.02× (parity), avg
/// ~2.75× memory (worst ~6.25× right after a grow); GROWTH=2 → ~1.48× inserts,
/// avg ~2.03× memory (worst ~3.37×). No other operation is affected by GROWTH.
pub struct PoMap<K: Key, V: Value, H: BuildHasher = PoMapBuildHasher, const GROWTH: usize = 4> {
    len: usize,
    grow_threshold: usize,
    meta: Meta,
    slots: Slots<K, V>,
    hash_builder: H,
}

impl<K: Key, V: Value, const GROWTH: usize> PoMap<K, V, PoMapBuildHasher, GROWTH> {
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

impl<K: Key, V: Value, H: BuildHasher + Default, const GROWTH: usize> Default
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn default() -> Self {
        Self::with_hasher(H::default())
    }
}

impl<K: Key, V: Value, H: BuildHasher, const GROWTH: usize> PoMap<K, V, H, GROWTH> {
    /// Compile-time guard: a non-power-of-two growth factor breaks the
    /// power-of-two `ideal_range` invariant that `ideal_slot` relies on.
    const GROWTH_VALID: () = assert!(
        GROWTH.is_power_of_two() && GROWTH >= 2,
        "PoMap GROWTH must be a power of two >= 2"
    );

    /// Creates an empty `PoMap` which will use the given hash builder.
    #[inline]
    pub fn with_hasher(hash_builder: H) -> Self {
        Self::with_capacity_and_hasher(MIN_IDEAL_RANGE, hash_builder)
    }

    /// Creates an empty `PoMap` with the specified capacity and hash builder.
    ///
    /// The map will hold at least `capacity` entries without reallocating.
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: H) -> Self {
        let () = Self::GROWTH_VALID;
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
        self.grow_threshold
    }

    /// Returns a reference to the value corresponding to `key`.
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            // Hit-biased order: test equality before the terminator — most
            // lookups hit at or near the ideal slot, so the common path takes
            // one branch instead of two.
            if stored == hash {
                let Entry { key: k, value: v, .. } = unsafe { &*(*entries.add(pos)).as_ptr() };
                if k == key {
                    return Some(v);
                }
            } else if stored > hash {
                return None; // also catches EMPTY_HASH (u64::MAX)
            }
            pos += 1;
        }
    }

    /// Returns a mutable reference to the value corresponding to `key`.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let entry = unsafe { &mut *(*entries.add(pos)).as_mut_ptr() };
                if entry.key == *key {
                    return Some(&mut entry.value);
                }
            }
            pos += 1;
        }
    }

    /// Returns a reference to the key-value pair corresponding to `key`.
    #[inline]
    pub fn get_key_value(&self, key: &K) -> Option<(&K, &V)> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let Entry { key: k, value: v, .. } = unsafe { &*(*entries.add(pos)).as_ptr() };
                if k == key {
                    return Some((k, v));
                }
            }
            pos += 1;
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
        if self.len >= self.grow_threshold {
            self.grow();
        }

        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let ideal = self.meta.ideal_slot(hash);

        // Fast path: ideal slot vacant → direct write, no scan needed.
        if self.slots.hash_at(ideal) == EMPTY_HASH {
            unsafe {
                *self.slots.entries.add(ideal) = MaybeUninit::new(Entry { hash, key, value });
            }
            self.len += 1;
            return None;
        }

        self.insert_slow(key, value, hash, ideal)
    }

    #[inline(never)]
    fn insert_slow(&mut self, key: K, value: V, hash: u64, ideal: usize) -> Option<V> {
        let entries = self.slots.entries;
        let mut pos = ideal;
        loop {
            let stored = self.slots.hash_at(pos);
            if stored == EMPTY_HASH {
                unsafe {
                    *entries.add(pos) = MaybeUninit::new(Entry { hash, key, value });
                }
                self.len += 1;
                return None;
            }
            if stored == hash {
                let entry = unsafe { &*(*entries.add(pos)).as_ptr() };
                if entry.key == key {
                    let e = unsafe { &mut *(*entries.add(pos)).as_mut_ptr() };
                    return Some(mem::replace(&mut e.value, value));
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

        // (hash, key) belongs at `pos`: shift the run right into the nearest
        // vacancy with a single memmove, then write at `pos`.
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
            *entries.add(pos) = MaybeUninit::new(Entry { hash, key, value });
        }
        self.len += 1;
        None
    }

    /// Removes the entry for `key`, returning the value if present.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the key-value pair if present.
    #[inline]
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let entry = unsafe { &*(*entries.add(pos)).as_ptr() };
                if entry.key == *key {
                    let Entry { key: k, value: v, .. } = unsafe { (*entries.add(pos)).assume_init_read() };
                    self.backshift(pos);
                    self.len -= 1;
                    return Some((k, v));
                }
            }
            pos += 1;
        }
    }

    /// Backshifts the trailing run of displaced entries left by one to fill
    /// the hole at `pos`, then marks the run's last slot vacant.
    #[inline]
    fn backshift(&mut self, pos: usize) {
        let entries = self.slots.entries;
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
    }

    /// Clears the map, removing all key-value pairs but keeping the allocation.
    #[inline]
    pub fn clear(&mut self) {
        for i in 0..self.slots.total_slots {
            if self.slots.hash_at(i) != EMPTY_HASH {
                unsafe { ptr::drop_in_place((*self.slots.entries.add(i)).as_mut_ptr()) };
            }
        }
        unsafe {
            ptr::write_bytes(self.slots.ptr.as_ptr(), 0xFF, self.slots.layout.size());
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
        let entries = self.slots.entries;
        let total = self.slots.total_slots;
        let mut i = 0;
        while i < total {
            if self.slots.hash_at(i) == EMPTY_HASH {
                i += 1;
                continue;
            }
            let entry = unsafe { &mut *(*entries.add(i)).as_mut_ptr() };
            if f(&entry.key, &mut entry.value) {
                i += 1;
                continue;
            }
            unsafe { ptr::drop_in_place((*entries.add(i)).as_mut_ptr()) };
            self.backshift(i);
            self.len -= 1;
            // Do NOT advance i — the next entry may have shifted into this slot.
        }
    }

    /// Returns an iterator over key-value pairs in deterministic (hash) order.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            mask: 0,
            mask_base: 0,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable iterator over key-value pairs in deterministic order.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            mask: 0,
            mask_base: 0,
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
            entries: self.slots.entries,
            index: 0,
            total_slots: self.slots.total_slots,
            remaining: self.len,
            _marker: PhantomData,
        };
        self.len = 0;
        drain
    }

    /// Shrinks the capacity of the map as much as possible while keeping room
    /// for at least `min_capacity` entries.
    ///
    /// The resulting capacity may be larger than `min_capacity` due to
    /// power-of-two sizing. No-op if already at or below the target.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let target = self.len.max(min_capacity);
        let needed_ideal = ideal_range_for(target);
        if needed_ideal >= self.meta.ideal_range {
            return;
        }
        // Sparse source: plain `max(cursor, ideal)` repack (no per-vacant bump).
        self.rebuild(needed_ideal, false);
    }

    /// Shrinks the capacity of the map as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(self.len);
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows or the allocator fails.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        if needed <= self.capacity() {
            return;
        }
        let needed_ideal = ideal_range_for(needed);
        if needed_ideal > self.meta.ideal_range {
            // Possibly sparse source — no per-vacant cursor bumps.
            self.rebuild(needed_ideal, false);
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
        let needed_ideal = try_ideal_range_for(needed)?;
        if needed_ideal > self.meta.ideal_range {
            let new_total = needed_ideal
                .checked_add(padding_for(needed_ideal))
                .ok_or(TryReserveError::CapacityOverflow)?;
            let new_slots = Slots::try_new(new_total)?;
            self.repack_into(needed_ideal, new_slots, false);
        }
        Ok(())
    }

    fn grow(&mut self) {
        // Near-full source: preserve inter-run gaps with per-vacant cursor bumps.
        self.rebuild(self.meta.ideal_range * GROWTH, true);
    }

    /// Repacks live entries into a fresh allocation sized for `new_ideal_range`.
    ///
    /// Each entry lands at `max(cursor, ideal_slot)` so entries only move
    /// forward. When `bump_gaps` is set, the cursor also advances by one for
    /// every vacant slot in the old map, reproducing the inter-run spacing in
    /// the new map so future inserts at those ideal slots are direct. Only safe
    /// to bump when the source is near-full (grow); on a sparse source it would
    /// race the cursor far past every window.
    fn rebuild(&mut self, new_ideal_range: usize, bump_gaps: bool) {
        let new_total = new_ideal_range + padding_for(new_ideal_range);
        // Fully memset-initialized target, then a second pass writing the ~60%
        // occupied slots. Single-pass "write each slot exactly once" variants
        // (region-fill per gap; inline 8-byte gap stores) were tried and measured
        // 35-44% SLOWER: the upfront memset streams complete cache lines with no
        // read-for-ownership, which beats any partial-line gap-filling pattern.
        let new_slots = Slots::new(new_total);
        self.repack_into(new_ideal_range, new_slots, bump_gaps);
    }

    fn repack_into(&mut self, new_ideal_range: usize, new_slots: Slots<K, V>, bump_gaps: bool) {
        let new_meta = Meta::new(new_ideal_range);

        let mut cursor = 0usize;
        for i in 0..self.slots.total_slots {
            let h = self.slots.hash_at(i);
            if h == EMPTY_HASH {
                if bump_gaps {
                    cursor += 1;
                }
                continue;
            }
            // Move the entry out. The old Slots is mem::forget-en below (its
            // Drop never runs), so there is no need to clear the moved-from slot.
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

// ---------------------------------------------------------------------------
// Iterator types
// ---------------------------------------------------------------------------

/// Iterator over shared references to key-value pairs in deterministic order.
#[must_use]
pub struct Iter<'a, K: Key, V: Value> {
    entries: *const MaybeUninit<Entry<K, V>>,
    /// Next slot the occupancy refill will scan.
    index: usize,
    total_slots: usize,
    remaining: usize,
    /// Occupancy bits for slots `[mask_base, mask_base + 64)`, consumed LSB-first.
    mask: u64,
    mask_base: usize,
    _marker: PhantomData<&'a (K, V)>,
}

// SAFETY: Iter only reads shared references derived from an immutable borrow.
unsafe impl<K: Key + Sync, V: Value + Sync> Send for Iter<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Iter<'_, K, V> {}

impl<K: Key, V: Value> Clone for Iter<'_, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            entries: self.entries,
            index: self.index,
            total_slots: self.total_slots,
            remaining: self.remaining,
            mask: self.mask,
            mask_base: self.mask_base,
            _marker: PhantomData,
        }
    }
}

impl<'a, K: Key, V: Value> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Occupancy at random slots makes a per-slot skip branch unpredictable
        // (~50% miss at typical density), which dominates iteration cost.
        // Instead, refill a 64-slot occupancy mask with a branchless pass
        // (cmp → set → or; the loop trip is predictable), then pop set bits.
        // `remaining == 0` also stops without scanning trailing empties.
        if self.remaining == 0 {
            return None;
        }
        while self.mask == 0 {
            let n = (self.total_slots - self.index).min(64);
            let mut m = 0u64;
            if n == 64 {
                for j in 0..64 {
                    let occ =
                        (unsafe { *(self.entries.add(self.index + j) as *const u64) }
                            != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            } else {
                for j in 0..n {
                    let occ =
                        (unsafe { *(self.entries.add(self.index + j) as *const u64) }
                            != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            }
            self.mask = m;
            self.mask_base = self.index;
            self.index += n;
        }
        let bit = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1;
        self.remaining -= 1;
        let idx = self.mask_base + bit;
        let Entry { key: k, value: v, .. } = unsafe { &*(*self.entries.add(idx)).as_ptr() };
        Some((k, v))
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
    entries: *mut MaybeUninit<Entry<K, V>>,
    /// Next slot the occupancy refill will scan.
    index: usize,
    total_slots: usize,
    remaining: usize,
    /// Occupancy bits for slots `[mask_base, mask_base + 64)`, consumed LSB-first.
    mask: u64,
    mask_base: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: IterMut yields &K and &mut V from a unique borrow of the map.
unsafe impl<K: Key + Send, V: Value + Send> Send for IterMut<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for IterMut<'_, K, V> {}

impl<'a, K: Key, V: Value> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Same branchless 64-slot occupancy-mask scheme as `Iter::next`.
        if self.remaining == 0 {
            return None;
        }
        while self.mask == 0 {
            let n = (self.total_slots - self.index).min(64);
            let mut m = 0u64;
            if n == 64 {
                for j in 0..64 {
                    let occ =
                        (unsafe { *(self.entries.add(self.index + j) as *const u64) }
                            != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            } else {
                for j in 0..n {
                    let occ =
                        (unsafe { *(self.entries.add(self.index + j) as *const u64) }
                            != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            }
            self.mask = m;
            self.mask_base = self.index;
            self.index += n;
        }
        let bit = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1;
        self.remaining -= 1;
        let idx = self.mask_base + bit;
        let entry = unsafe { &mut *(*self.entries.add(idx)).as_mut_ptr() };
        Some((&entry.key, &mut entry.value))
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
            if self.slots.hash_at(idx) == EMPTY_HASH {
                continue;
            }
            self.remaining -= 1;
            // Mark consumed so Slots::drop doesn't double-drop it.
            let Entry { key: k, value: v, .. } =
                unsafe { (*self.slots.entries.add(idx)).assume_init_read() };
            unsafe {
                *(self.slots.entries.add(idx) as *mut u64) = EMPTY_HASH;
            }
            return Some((k, v));
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
    entries: *mut MaybeUninit<Entry<K, V>>,
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
            if unsafe { *(self.entries.add(idx) as *const u64) } == EMPTY_HASH {
                continue;
            }
            self.remaining -= 1;
            let Entry { key: k, value: v, .. } =
                unsafe { (*self.entries.add(idx)).assume_init_read() };
            unsafe {
                *(self.entries.add(idx) as *mut u64) = EMPTY_HASH;
            }
            return Some((k, v));
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
            if unsafe { *(self.entries.add(idx) as *const u64) } == EMPTY_HASH {
                continue;
            }
            unsafe {
                ptr::drop_in_place((*self.entries.add(idx)).as_mut_ptr());
                *(self.entries.add(idx) as *mut u64) = EMPTY_HASH;
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

impl<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> IntoIterator
    for &'a PoMap<K, V, H, GROWTH>
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> IntoIterator
    for &'a mut PoMap<K, V, H, GROWTH>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K: Key, V: Value, H: BuildHasher, const GROWTH: usize> IntoIterator
    for PoMap<K, V, H, GROWTH>
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let Self {
            len,
            slots,
            ..
        } = self;
        IntoIter {
            slots,
            index: 0,
            remaining: len,
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls: Clone, Debug, PartialEq, Eq, Index, Extend, FromIterator
// ---------------------------------------------------------------------------

impl<K: Key, V: Value, H: BuildHasher + Clone, const GROWTH: usize> Clone
    for PoMap<K, V, H, GROWTH>
{
    fn clone(&self) -> Self {
        let new_slots = Slots::new(self.slots.total_slots);
        for i in 0..self.slots.total_slots {
            if self.slots.hash_at(i) != EMPTY_HASH {
                unsafe {
                    let e = &*(*self.slots.entries.add(i)).as_ptr();
                    *new_slots.entries.add(i) = MaybeUninit::new(Entry {
                        hash: e.hash,
                        key: e.key.clone(),
                        value: e.value.clone(),
                    });
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

impl<K: Key + fmt::Debug, V: Value + fmt::Debug, H: BuildHasher, const GROWTH: usize> fmt::Debug
    for PoMap<K, V, H, GROWTH>
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

impl<K: Key, V: Value + PartialEq, H: BuildHasher, const GROWTH: usize> PartialEq
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<K: Key, V: Value + Eq, H: BuildHasher, const GROWTH: usize> Eq for PoMap<K, V, H, GROWTH> {}

impl<K: Key, V: Value, H: BuildHasher, const GROWTH: usize> Index<&K> for PoMap<K, V, H, GROWTH> {
    type Output = V;

    #[inline]
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("key not found")
    }
}

impl<K: Key, V: Value, H: BuildHasher, const GROWTH: usize> Extend<(K, V)>
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: BuildHasher, const GROWTH: usize> Extend<(&'a K, &'a V)>
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key.clone(), value.clone());
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher + Default, const GROWTH: usize> FromIterator<(K, V)>
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = PoMap::with_capacity_and_hasher(lower, H::default());
        map.extend(iter);
        map
    }
}

impl<'a, K: Key + 'a, V: Value + 'a, H: BuildHasher + Default, const GROWTH: usize>
    FromIterator<(&'a K, &'a V)> for PoMap<K, V, H, GROWTH>
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

    fn fuzz_against_hashmap<const GROWTH: usize>() {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xDEAD);
        let mut m: PoMap<u64, u64, BuildHasherDefault<AHasher>, GROWTH> =
            PoMap::with_hasher(BuildHasherDefault::default());
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
    fn grow_large_growth2() {
        let mut map: PoMap<u64, u64, BuildHasherDefault<AHasher>, 2> =
            PoMap::with_hasher(BuildHasherDefault::default());
        for i in 0..10_000u64 {
            map.insert(i, i);
        }
        for i in 0..10_000u64 {
            assert_eq!(map.get(&i), Some(&i), "missing {}", i);
        }
        assert_eq!(map.len(), 10_000);
    }

    #[test]
    fn matches_hashmap() {
        fuzz_against_hashmap::<4>();
    }

    #[test]
    fn matches_hashmap_growth2() {
        fuzz_against_hashmap::<2>();
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
    fn retain_large() {
        let mut map = new_map();
        for i in 0..5000u64 {
            map.insert(i, i);
        }
        map.retain(|k, _| k % 3 == 0);
        for i in 0..5000u64 {
            if i % 3 == 0 {
                assert_eq!(map.get(&i), Some(&i));
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
    fn into_iter_partial_drop() {
        let mut map = new_map();
        for i in 0..100u64 {
            map.insert(i, i);
        }
        let mut iter = map.into_iter();
        let _ = iter.next();
        let _ = iter.next();
        drop(iter); // remaining entries must be dropped without leaks/UB
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
    fn shrink_after_removes() {
        let mut map = new_map();
        for i in 0..1000u64 {
            map.insert(i, i);
        }
        for i in 0..900u64 {
            map.remove(&i);
        }
        map.shrink_to(0);
        for i in 900..1000u64 {
            assert_eq!(map.get(&i), Some(&i));
        }
        for i in 0..900u64 {
            assert_eq!(map.get(&i), None);
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
    fn with_capacity_no_grow() {
        // with_capacity(n) must allow n inserts with zero rebuilds: capacity()
        // strictly exceeds n-1 inserts' trigger point.
        let mut map: PoMap<u64, u64> = PoMap::with_capacity(1000);
        assert!(map.capacity() >= 1000);
        for i in 0..1000u64 {
            map.insert(i, i);
        }
        assert_eq!(map.len(), 1000);
    }

    #[test]
    fn try_reserve_basic() {
        let mut map = new_map();
        assert!(map.try_reserve(100).is_ok());
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn try_reserve_overflow() {
        let mut map = new_map();
        assert_eq!(
            map.try_reserve(usize::MAX),
            Err(TryReserveError::CapacityOverflow)
        );
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

    #[test]
    fn string_keys() {
        let mut map: PoMap<String, String> = PoMap::new();
        for i in 0..500 {
            map.insert(format!("key-{i}"), format!("value-{i}"));
        }
        for i in 0..500 {
            assert_eq!(map.get(&format!("key-{i}")), Some(&format!("value-{i}")));
        }
        map.retain(|k, _| !k.ends_with('7'));
        for i in 0..500 {
            let expect_present = i % 10 != 7;
            assert_eq!(
                map.contains_key(&format!("key-{i}")),
                expect_present,
                "key-{i}"
            );
        }
        map.clear();
        assert!(map.is_empty());
    }
}
