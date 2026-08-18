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
    borrow::Borrow,
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
/// The `Ord` bound exists for exactly one purpose: **canonicalizing 64-bit
/// hash collisions**. Entries are ordered by hash; when two distinct keys
/// collide on their full encoded hash (probability ≈ n²/2⁶⁵ — never on any
/// non-adversarial workload), the tie is broken by key order so that the
/// array layout, iteration order, `Eq`, `Ord`, and `Hash` of the *map itself*
/// are pure functions of its contents. A key comparison therefore executes
/// only on a full-hash collision during insert — never on any probe, lookup,
/// remove, or resize path.
pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

/// Marker trait for values stored in a [`PoMap`].
pub trait Value: Clone {}
impl<V: Clone> Value for V {}

/// A slot's contents. `repr(C)` guarantees the hash lives at offset 0 so the
/// raw `u64` reads in `hash_at` (and the 0xFF vacancy memset) are layout-safe
/// for any K/V — a plain tuple's repr(Rust) field order is unspecified.
#[repr(C)]
struct SlotEntry<K, V> {
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
    entries: *mut MaybeUninit<SlotEntry<K, V>>,
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
        let layout = Layout::array::<MaybeUninit<SlotEntry<K, V>>>(total_slots)
            .map_err(|_| TryReserveError::CapacityOverflow)?
            .pad_to_align();

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(TryReserveError::AllocError { layout })?;

        let entries = ptr.as_ptr() as *mut MaybeUninit<SlotEntry<K, V>>;
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

    /// Reads the raw hash word of slot `i`.
    ///
    /// SAFETY CONTRACT (the "trailing vacant" invariant): the final slot of
    /// every table is permanently vacant — enforced by `insert_slow`'s two
    /// landing guards and `repack_into`'s cursor bound — so every forward
    /// probe scan terminates at or before it (`EMPTY_HASH` is `u64::MAX`, the
    /// maximum, so both `stored > hash` and `== EMPTY` cutoffs fire there).
    /// This is what lets the probe loops omit bounds checks entirely.
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
    ///
    /// The key may be any borrowed form of the map's key type, with matching
    /// `Hash` and `Eq` (e.g. query a `String`-keyed map with `&str`).
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            // Hit-biased order: test equality before the terminator — most
            // lookups hit at or near the ideal slot, so the common path takes
            // one branch instead of two.
            if stored == hash {
                let SlotEntry { key: k, value: v, .. } = unsafe { &*(*entries.add(pos)).as_ptr() };
                if k.borrow() == key {
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
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
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
                if entry.key.borrow() == key {
                    return Some(&mut entry.value);
                }
            }
            pos += 1;
        }
    }

    /// Returns a reference to the key-value pair corresponding to `key`.
    #[inline]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = encode_hash(self.hash_builder.hash_one(key));
        let entries = self.slots.entries;
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            if stored > hash {
                return None;
            }
            if stored == hash {
                let SlotEntry { key: k, value: v, .. } = unsafe { &*(*entries.add(pos)).as_ptr() };
                if k.borrow() == key {
                    return Some((k, v));
                }
            }
            pos += 1;
        }
    }

    /// Returns `true` if the map contains the given key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
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
                *self.slots.entries.add(ideal) = MaybeUninit::new(SlotEntry { hash, key, value });
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
                // Never occupy the final slot: probe loops rely on it staying
                // vacant as their branch-free scan terminator.
                if pos + 1 >= self.slots.total_slots {
                    self.grow();
                    return self.insert(key, value);
                }
                unsafe {
                    *entries.add(pos) = MaybeUninit::new(SlotEntry { hash, key, value });
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
                // Full 64-bit hash collision between distinct keys (~n²/2⁶⁵):
                // canonicalize the tie by key order so layout and iteration
                // are functions of contents alone (required for the map-level
                // Eq/Ord/Hash impls). This compare runs only on collisions.
                if entry.key > key {
                    break; // insert before the larger colliding key
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
        // vacancy with a single memmove, then write at `pos`. The final slot
        // is reserved (never shifted into) so probes stay bounds-check-free.
        let mut empty_pos = pos + 1;
        loop {
            if empty_pos + 1 >= self.slots.total_slots {
                self.grow();
                return self.insert(key, value);
            }
            if self.slots.hash_at(empty_pos) == EMPTY_HASH {
                break;
            }
            empty_pos += 1;
        }

        let shift = empty_pos - pos;
        unsafe {
            ptr::copy(entries.add(pos), entries.add(pos + 1), shift);
            *entries.add(pos) = MaybeUninit::new(SlotEntry { hash, key, value });
        }
        self.len += 1;
        None
    }

    /// Gets the entry for `key` for in-place manipulation.
    ///
    /// ```
    /// use pomap::PoMap;
    /// let mut counts: PoMap<&str, u32> = PoMap::new();
    /// for word in ["a", "b", "a"] {
    ///     *counts.entry(word).or_insert(0) += 1;
    /// }
    /// assert_eq!(counts["a"], 2);
    /// ```
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, H, GROWTH> {
        let hash = encode_hash(self.hash_builder.hash_one(&key));
        let mut pos = self.meta.ideal_slot(hash);
        loop {
            let stored = self.slots.hash_at(pos);
            if stored == hash {
                let entry = unsafe { &*(*self.slots.entries.add(pos)).as_ptr() };
                if entry.key == key {
                    return Entry::Occupied(OccupiedEntry { map: self, pos });
                }
            } else if stored > hash {
                return Entry::Vacant(VacantEntry { map: self, key });
            }
            pos += 1;
        }
    }

    /// Returns the first key-value pair in iteration (hash) order.
    ///
    /// Because iteration order is canonical, this is a *deterministic*
    /// "arbitrary element": the same map contents always yield the same
    /// first entry, regardless of insertion history.
    #[inline]
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        if self.len == 0 {
            return None;
        }
        let mut i = 0;
        while self.slots.hash_at(i) == EMPTY_HASH {
            i += 1;
        }
        let SlotEntry { key: k, value: v, .. } =
            unsafe { &*(*self.slots.entries.add(i)).as_ptr() };
        Some((k, v))
    }

    /// Returns the last key-value pair in iteration (hash) order.
    #[inline]
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        if self.len == 0 {
            return None;
        }
        let mut i = self.slots.total_slots - 1;
        while self.slots.hash_at(i) == EMPTY_HASH {
            i -= 1;
        }
        let SlotEntry { key: k, value: v, .. } =
            unsafe { &*(*self.slots.entries.add(i)).as_ptr() };
        Some((k, v))
    }

    /// Removes and returns the first key-value pair in iteration (hash)
    /// order. Repeated `pop_first` drains the map in canonical order —
    /// a reproducible work-queue regardless of how the map was built.
    #[inline]
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        if self.len == 0 {
            return None;
        }
        let mut i = 0;
        while self.slots.hash_at(i) == EMPTY_HASH {
            i += 1;
        }
        let SlotEntry { key: k, value: v, .. } =
            unsafe { (*self.slots.entries.add(i)).assume_init_read() };
        self.backshift(i);
        self.len -= 1;
        Some((k, v))
    }

    /// Removes and returns the last key-value pair in iteration (hash) order.
    #[inline]
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        if self.len == 0 {
            return None;
        }
        let mut i = self.slots.total_slots - 1;
        while self.slots.hash_at(i) == EMPTY_HASH {
            i -= 1;
        }
        let SlotEntry { key: k, value: v, .. } =
            unsafe { (*self.slots.entries.add(i)).assume_init_read() };
        self.backshift(i);
        self.len -= 1;
        Some((k, v))
    }

    /// Removes the entry for `key`, returning the value if present.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the key-value pair if present.
    #[inline]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
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
                if entry.key.borrow() == key {
                    let SlotEntry { key: k, value: v, .. } = unsafe { (*entries.add(pos)).assume_init_read() };
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
            mask: 0,
            mask_base: 0,
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

    /// Repacks the map into its minimal table, producing a **canonical
    /// representation**: after `compact()`, every entry position, every gap,
    /// and the allocation size are a pure function of the map's *contents* —
    /// two maps with equal contents are structurally identical after
    /// compaction, regardless of the insertion/removal/growth history that
    /// produced them. (Byte-for-byte identity additionally requires the
    /// payload types' in-memory representation to be canonical — true for
    /// primitives; types with padding or heap pointers canonicalize
    /// structurally, not bytewise.)
    ///
    /// This is on-demand history independence: iteration order is canonical
    /// at *all* times; the live layout is deliberately history-dependent
    /// (grow-time cursor spacing trades layout canonicity for insert
    /// performance), and `compact()` restores representation canonicity.
    /// Unlike [`shrink_to_fit`](Self::shrink_to_fit), it repacks even when
    /// the table is already minimally sized.
    pub fn compact(&mut self) {
        self.rebuild(ideal_range_for(self.len), false);
    }

    // -----------------------------------------------------------------------
    // Set algebra: streaming merges over two hash-sorted tables
    // -----------------------------------------------------------------------

    /// Returns a new map with every key from `self` and `other`; on keys
    /// present in both, **`other`'s value wins** (matching `extend`
    /// semantics).
    ///
    /// Runs in **O(n + m)** as a streaming merge of the two hash-sorted
    /// tables — no re-hashing, no probing, no shifting — and the result is
    /// **compact-canonical** (identical bytes to building the same contents
    /// and calling [`compact`](Self::compact)).
    ///
    /// Like the map-level `Ord`/`Hash` impls, this requires both maps to use
    /// the same per-type-deterministic hasher (true for the defaults): the
    /// merge trusts the stored hashes.
    ///
    /// ```
    /// use pomap::PoMap;
    /// let a: PoMap<u32, &str> = [(1, "a"), (2, "a")].into_iter().collect();
    /// let b: PoMap<u32, &str> = [(2, "b"), (3, "b")].into_iter().collect();
    /// let u = a.union(&b);
    /// assert_eq!(u.len(), 3);
    /// assert_eq!(u[&2], "b"); // right-biased
    /// ```
    pub fn union(&self, other: &Self) -> Self
    where
        H: Clone,
    {
        self.merge_build(other, MergeMode::Union)
    }

    /// Returns a new map with the keys present in **both** maps, with values
    /// taken from `self`. O(n + m) streaming merge; compact-canonical result.
    pub fn intersection(&self, other: &Self) -> Self
    where
        H: Clone,
    {
        self.merge_build(other, MergeMode::Intersection)
    }

    /// Returns a new map with the keys of `self` that are **not** in `other`.
    /// O(n + m) streaming merge; compact-canonical result.
    pub fn difference(&self, other: &Self) -> Self
    where
        H: Clone,
    {
        self.merge_build(other, MergeMode::Difference)
    }

    /// Returns a new map with the keys present in **exactly one** of the two
    /// maps. O(n + m) streaming merge; compact-canonical result.
    pub fn symmetric_difference(&self, other: &Self) -> Self
    where
        H: Clone,
    {
        self.merge_build(other, MergeMode::SymmetricDifference)
    }

    /// Moves all entries of `other` into `self`, leaving `other` empty. On
    /// duplicate keys `other`'s value wins ([`BTreeMap::append`] parity).
    /// Implemented as a streaming [`union`](Self::union) rebuild.
    ///
    /// [`BTreeMap::append`]: alloc::collections::BTreeMap::append
    pub fn append(&mut self, other: &mut Self)
    where
        H: Clone,
    {
        *self = self.union(other);
        other.clear();
    }

    /// Core of the set operations: a two-pass sorted merge. Pass 1 counts the
    /// result; pass 2 places cloned entries with the canonical cursor walk
    /// (retrying at a doubled table on tail overflow, like `rebuild`), so the
    /// output is exactly what `compact()` would produce for those contents.
    fn merge_build(&self, other: &Self, mode: MergeMode) -> Self
    where
        H: Clone,
    {
        // Pass 1: count emissions.
        let mut count = 0usize;
        self.merge_walk(other, |_, item| {
            if mode.pick::<K, V>(&item).is_some() {
                count += 1;
            }
        });

        // Pass 2 (with doubling retry): place emissions in canonical order.
        let mut target = ideal_range_for(count);
        'retry: loop {
            let new_total = target
                .checked_add(padding_for(target))
                .expect("PoMap capacity overflow");
            let new_slots = Slots::<K, V>::new(new_total);
            let new_meta = Meta::new(target);
            let last_usable = new_total - 1;
            let mut cursor = 0usize;
            let mut placed = 0usize;
            let mut overflow = false;
            self.merge_walk(other, |hash, item| {
                if overflow {
                    return;
                }
                let Some(entry) = mode.pick::<K, V>(&item) else {
                    return;
                };
                cursor = cursor.max(new_meta.ideal_slot(hash));
                if cursor >= last_usable {
                    overflow = true;
                    return;
                }
                unsafe {
                    *new_slots.entries.add(cursor) = MaybeUninit::new(SlotEntry {
                        hash,
                        key: entry.key.clone(),
                        value: entry.value.clone(),
                    });
                }
                cursor += 1;
                placed += 1;
            });
            if overflow {
                // `new_slots` drops normally: it owns the clones placed so far.
                target = target.checked_mul(2).expect("PoMap capacity overflow");
                continue 'retry;
            }
            debug_assert_eq!(placed, count);
            return Self {
                len: count,
                grow_threshold: target * LOAD_NUM / LOAD_DEN,
                meta: new_meta,
                slots: new_slots,
                hash_builder: self.hash_builder.clone(),
            };
        }
    }

    /// Walks the two hash-sorted tables in lockstep, invoking `f` once per
    /// merged key in canonical (hash, key) order with the entry (or entries)
    /// holding it. Occupied slots are located with the same branchless
    /// 64-slot occupancy masks as `Iter` (§ the per-slot skip branch is
    /// unpredictable at typical densities and would dominate the merge).
    fn merge_walk<'m>(
        &'m self,
        other: &'m Self,
        mut f: impl FnMut(u64, MergeItem<'m, K, V>),
    ) {
        let mut a = OccCursor::new(&self.slots);
        let mut b = OccCursor::new(&other.slots);
        loop {
            match (a.peek(), b.peek()) {
                (None, None) => return,
                (Some((ha, x)), None) => {
                    f(ha, MergeItem::Left(x));
                    a.pop();
                }
                (None, Some((hb, y))) => {
                    f(hb, MergeItem::Right(y));
                    b.pop();
                }
                (Some((ha, x)), Some((hb, y))) => {
                    use core::cmp::Ordering::*;
                    match ha.cmp(&hb).then_with(|| x.key.cmp(&y.key)) {
                        Less => {
                            f(ha, MergeItem::Left(x));
                            a.pop();
                        }
                        Greater => {
                            f(hb, MergeItem::Right(y));
                            b.pop();
                        }
                        Equal => {
                            f(ha, MergeItem::Both(x, y));
                            a.pop();
                            b.pop();
                        }
                    }
                }
            }
        }
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
            let mut target = needed_ideal;
            loop {
                let new_total = target
                    .checked_add(padding_for(target))
                    .ok_or(TryReserveError::CapacityOverflow)?;
                let new_slots = Slots::try_new(new_total)?;
                if self.repack_into(target, new_slots, false) {
                    break;
                }
                target = target
                    .checked_mul(2)
                    .ok_or(TryReserveError::CapacityOverflow)?;
            }
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
        // Retry with a doubled target if the repack would overflow the table
        // (possible only under extreme hash clustering — e.g. an adversarial
        // or degenerate hasher piling entries at the top of the range). The
        // retry threshold is a pure function of the ordered contents and the
        // target size, so `compact()`'s canonical-representation guarantee is
        // preserved: equal contents always settle in the same table.
        let mut target = new_ideal_range;
        loop {
            let new_total = target
                .checked_add(padding_for(target))
                .expect("PoMap capacity overflow");
            // Fully memset-initialized target, then a second pass writing the
            // ~60% occupied slots. Single-pass "write each slot exactly once"
            // variants (region-fill per gap; inline 8-byte gap stores) were
            // tried and measured 35-44% SLOWER: the upfront memset streams
            // complete cache lines with no read-for-ownership, which beats any
            // partial-line gap-filling pattern.
            let new_slots = Slots::new(new_total);
            if self.repack_into(target, new_slots, bump_gaps) {
                return;
            }
            target = target.checked_mul(2).expect("PoMap capacity overflow");
        }
    }

    /// Repacks live entries into `new_slots`. Returns `true` on success
    /// (`new_slots` installed, old allocation freed) or `false` if the repack
    /// would violate the trailing-vacant invariant (`new_slots` discarded, the
    /// map untouched — the caller retries with a larger target).
    ///
    /// Failure cleanup is sound because entries are moved as *bitwise copies*
    /// while the old slots' bytes stay intact: on success the OLD allocation
    /// is freed raw (its `Drop` never runs), on failure the NEW one is —
    /// either way exactly one table ever owns (and eventually drops) the
    /// entries.
    fn repack_into(&mut self, new_ideal_range: usize, new_slots: Slots<K, V>, bump_gaps: bool) -> bool {
        let new_meta = Meta::new(new_ideal_range);
        // Keep the final slot vacant: probe loops rely on it as a
        // branch-free scan terminator (see `hash_at`).
        let last_usable = new_slots.total_slots - 1;

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
            if cursor >= last_usable {
                // Would occupy (or pass) the reserved final slot: discard the
                // new allocation WITHOUT running entry drops — the bitwise
                // copies written so far are still owned by the old table.
                unsafe { dealloc(new_slots.ptr.as_ptr(), new_slots.layout) };
                mem::forget(new_slots);
                return false;
            }
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
        true
    }
}

// ---------------------------------------------------------------------------
// Set-algebra support types
// ---------------------------------------------------------------------------

/// Which set operation a merge performs.
#[derive(Copy, Clone)]
enum MergeMode {
    Union,
    Intersection,
    Difference,
    SymmetricDifference,
}

/// Branchless cursor over a table's occupied slots in position (= canonical)
/// order: refills a 64-slot occupancy mask with a predictable cmp/set/or pass,
/// then pops set bits — the same scheme as `Iter::next`.
struct OccCursor<'m, K: Key, V: Value> {
    slots: &'m Slots<K, V>,
    mask: u64,
    base: usize,
    /// Next slot the refill will scan.
    scan: usize,
}

impl<'m, K: Key, V: Value> OccCursor<'m, K, V> {
    #[inline]
    fn new(slots: &'m Slots<K, V>) -> Self {
        Self { slots, mask: 0, base: 0, scan: 0 }
    }

    /// Returns the front occupied slot's (hash, entry) without consuming it.
    #[inline]
    fn peek(&mut self) -> Option<(u64, &'m SlotEntry<K, V>)> {
        while self.mask == 0 {
            if self.scan >= self.slots.total_slots {
                return None;
            }
            let n = (self.slots.total_slots - self.scan).min(64);
            let mut m = 0u64;
            if n == 64 {
                for j in 0..64 {
                    let occ = (self.slots.hash_at(self.scan + j) != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            } else {
                for j in 0..n {
                    let occ = (self.slots.hash_at(self.scan + j) != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            }
            self.mask = m;
            self.base = self.scan;
            self.scan += n;
        }
        let idx = self.base + self.mask.trailing_zeros() as usize;
        let entry = unsafe { &*(*self.slots.entries.add(idx)).as_ptr() };
        Some((self.slots.hash_at(idx), entry))
    }

    /// Consumes the front occupied slot.
    #[inline]
    fn pop(&mut self) {
        self.mask &= self.mask - 1;
    }
}

/// One merged element: present on the left, the right, or both sides.
enum MergeItem<'m, K: Key, V: Value> {
    Left(&'m SlotEntry<K, V>),
    Right(&'m SlotEntry<K, V>),
    Both(&'m SlotEntry<K, V>, &'m SlotEntry<K, V>),
}

impl MergeMode {
    /// Selects which entry (if any) this mode emits for a merged element.
    /// Union is right-biased on `Both` (matching `extend`); Intersection
    /// takes the left value.
    #[inline]
    fn pick<'m, K: Key, V: Value>(
        self,
        item: &MergeItem<'m, K, V>,
    ) -> Option<&'m SlotEntry<K, V>> {
        match (self, item) {
            (MergeMode::Union, MergeItem::Left(x)) => Some(x),
            (MergeMode::Union, MergeItem::Right(y)) => Some(y),
            (MergeMode::Union, MergeItem::Both(_, y)) => Some(y),
            (MergeMode::Intersection, MergeItem::Both(x, _)) => Some(x),
            (MergeMode::Intersection, _) => None,
            (MergeMode::Difference, MergeItem::Left(x)) => Some(x),
            (MergeMode::Difference, _) => None,
            (MergeMode::SymmetricDifference, MergeItem::Left(x)) => Some(x),
            (MergeMode::SymmetricDifference, MergeItem::Right(y)) => Some(y),
            (MergeMode::SymmetricDifference, MergeItem::Both(..)) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry API
// ---------------------------------------------------------------------------

/// A view into a single map slot, occupied or vacant, returned by
/// [`PoMap::entry`].
pub enum Entry<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> {
    /// The key is present.
    Occupied(OccupiedEntry<'a, K, V, H, GROWTH>),
    /// The key is absent.
    Vacant(VacantEntry<'a, K, V, H, GROWTH>),
}

/// A view into an occupied map slot.
pub struct OccupiedEntry<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> {
    map: &'a mut PoMap<K, V, H, GROWTH>,
    pos: usize,
}

/// A view into a vacant map slot, holding the key that would be inserted.
pub struct VacantEntry<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> {
    map: &'a mut PoMap<K, V, H, GROWTH>,
    key: K,
}

impl<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize> Entry<'a, K, V, H, GROWTH> {
    /// Inserts `default` if vacant; returns a mutable reference to the value.
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// Inserts the result of `default()` if vacant; returns a mutable
    /// reference to the value.
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default()),
        }
    }

    /// Inserts `V::default()` if vacant; returns a mutable reference.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Applies `f` to the value if occupied, then returns the entry.
    #[inline]
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(ref mut o) = self {
            f(o.get_mut());
        }
        self
    }

    /// Returns a reference to the entry's key.
    #[inline]
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(o) => o.key(),
            Entry::Vacant(v) => v.key(),
        }
    }
}

impl<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize>
    OccupiedEntry<'a, K, V, H, GROWTH>
{
    /// Returns a reference to the key.
    #[inline]
    pub fn key(&self) -> &K {
        let entry = unsafe { &*(*self.map.slots.entries.add(self.pos)).as_ptr() };
        &entry.key
    }

    /// Returns a reference to the value.
    #[inline]
    pub fn get(&self) -> &V {
        let entry = unsafe { &*(*self.map.slots.entries.add(self.pos)).as_ptr() };
        &entry.value
    }

    /// Returns a mutable reference to the value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        let entry = unsafe { &mut *(*self.map.slots.entries.add(self.pos)).as_mut_ptr() };
        &mut entry.value
    }

    /// Converts the entry into a mutable reference tied to the map's lifetime.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        let entry = unsafe { &mut *(*self.map.slots.entries.add(self.pos)).as_mut_ptr() };
        &mut entry.value
    }

    /// Replaces the value, returning the old one.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Removes the entry, returning the value.
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Removes the entry, returning the key-value pair.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let SlotEntry { key: k, value: v, .. } =
            unsafe { (*self.map.slots.entries.add(self.pos)).assume_init_read() };
        self.map.backshift(self.pos);
        self.map.len -= 1;
        (k, v)
    }
}

impl<'a, K: Key, V: Value, H: BuildHasher, const GROWTH: usize>
    VacantEntry<'a, K, V, H, GROWTH>
{
    /// Returns a reference to the key that would be inserted.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Returns ownership of the key without inserting.
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }

    /// Inserts the value, returning a mutable reference to it.
    ///
    /// The insert may grow or shift the table, so the slot is re-located
    /// afterward (the key is cloned for the re-lookup; `Key: Clone`).
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let key = self.key.clone();
        self.map.insert(self.key, value);
        self.map
            .get_mut(&key)
            .expect("vacant entry insert must succeed")
    }
}

// ---------------------------------------------------------------------------
// Iterator types
// ---------------------------------------------------------------------------

/// Iterator over shared references to key-value pairs in deterministic order.
#[must_use]
pub struct Iter<'a, K: Key, V: Value> {
    entries: *const MaybeUninit<SlotEntry<K, V>>,
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
        let SlotEntry { key: k, value: v, .. } = unsafe { &*(*self.entries.add(idx)).as_ptr() };
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
    entries: *mut MaybeUninit<SlotEntry<K, V>>,
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
    /// Next slot the occupancy refill will scan.
    index: usize,
    remaining: usize,
    /// Occupancy bits for slots `[mask_base, mask_base + 64)`, consumed LSB-first.
    mask: u64,
    mask_base: usize,
}

impl<K: Key, V: Value> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Same branchless 64-slot occupancy-mask scheme as `Iter::next`.
        // Consumed slots are marked EMPTY so `Slots::drop` (which this
        // iterator owns) doesn't double-drop them.
        if self.remaining == 0 {
            return None;
        }
        while self.mask == 0 {
            let n = (self.slots.total_slots - self.index).min(64);
            let mut m = 0u64;
            if n == 64 {
                for j in 0..64 {
                    let occ = (self.slots.hash_at(self.index + j) != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            } else {
                for j in 0..n {
                    let occ = (self.slots.hash_at(self.index + j) != EMPTY_HASH) as u64;
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
        let SlotEntry { key: k, value: v, .. } =
            unsafe { (*self.slots.entries.add(idx)).assume_init_read() };
        unsafe {
            *(self.slots.entries.add(idx) as *mut u64) = EMPTY_HASH;
        }
        Some((k, v))
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
    entries: *mut MaybeUninit<SlotEntry<K, V>>,
    /// Next slot the occupancy refill will scan.
    index: usize,
    total_slots: usize,
    remaining: usize,
    /// Occupancy bits for slots `[mask_base, mask_base + 64)`, consumed LSB-first.
    mask: u64,
    mask_base: usize,
    _marker: PhantomData<&'a mut (K, V)>,
}

// SAFETY: Drain has exclusive access derived from &mut PoMap.
unsafe impl<K: Key + Send, V: Value + Send> Send for Drain<'_, K, V> {}
unsafe impl<K: Key + Sync, V: Value + Sync> Sync for Drain<'_, K, V> {}

impl<K: Key, V: Value> Drain<'_, K, V> {
    /// Locates the next occupied slot (shared by `next` and `drop`), marking
    /// nothing; returns its index.
    #[inline]
    fn next_occupied(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        while self.mask == 0 {
            let n = (self.total_slots - self.index).min(64);
            let mut m = 0u64;
            if n == 64 {
                for j in 0..64 {
                    let occ = (unsafe { *(self.entries.add(self.index + j) as *const u64) }
                        != EMPTY_HASH) as u64;
                    m |= occ << j;
                }
            } else {
                for j in 0..n {
                    let occ = (unsafe { *(self.entries.add(self.index + j) as *const u64) }
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
        Some(self.mask_base + bit)
    }
}

impl<'a, K: Key, V: Value> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.next_occupied()?;
        let SlotEntry { key: k, value: v, .. } =
            unsafe { (*self.entries.add(idx)).assume_init_read() };
        unsafe {
            *(self.entries.add(idx) as *mut u64) = EMPTY_HASH;
        }
        Some((k, v))
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
        // Consume and drop any remaining entries, marking their slots vacant.
        while let Some(idx) = self.next_occupied() {
            unsafe {
                ptr::drop_in_place((*self.entries.add(idx)).as_mut_ptr());
                *(self.entries.add(idx) as *mut u64) = EMPTY_HASH;
            }
        }
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
            mask: 0,
            mask_base: 0,
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
                    *new_slots.entries.add(i) = MaybeUninit::new(SlotEntry {
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

    /// Reuses the existing allocation when the geometries match (total slot
    /// count determines `ideal_range` bijectively), avoiding a
    /// dealloc/alloc/memset cycle for repeated clone-into patterns.
    fn clone_from(&mut self, source: &Self) {
        if self.slots.total_slots != source.slots.total_slots {
            *self = source.clone();
            return;
        }
        // Drop our entries and re-vacate every slot, then mirror the source
        // slot-for-slot (identical layout ⟹ the result is bit-equivalent to
        // `source.clone()`).
        self.clear();
        for i in 0..source.slots.total_slots {
            if source.slots.hash_at(i) != EMPTY_HASH {
                unsafe {
                    let e = &*(*source.slots.entries.add(i)).as_ptr();
                    *self.slots.entries.add(i) = MaybeUninit::new(SlotEntry {
                        hash: e.hash,
                        key: e.key.clone(),
                        value: e.value.clone(),
                    });
                }
            }
        }
        self.len = source.len;
        self.grow_threshold = source.grow_threshold;
        self.hash_builder = source.hash_builder.clone();
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

/// Lexicographic comparison over the canonical (hash, key) iteration order.
///
/// Lawful because iteration order is a pure function of the map's contents:
/// entries are ordered by hash, with full-hash collisions canonicalized by key
/// order at insert. **Requires a per-type-deterministic hasher** (two maps of
/// the same type must agree on every key's hash — true for
/// [`PoMapBuildHasher`] and `BuildHasherDefault`, not for randomly seeded
/// states): with per-instance hasher seeds, maps of the same type would
/// disagree on iteration order and these impls would not be a lawful total
/// order.
impl<K: Key, V: Value + PartialOrd, H: BuildHasher, const GROWTH: usize> PartialOrd
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

/// See the [`PartialOrd`] impl for the canonical-order and hasher requirements.
impl<K: Key, V: Value + Ord, H: BuildHasher, const GROWTH: usize> Ord for PoMap<K, V, H, GROWTH> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

/// Hashes the map's contents in canonical iteration order (length-prefixed),
/// so equal maps hash equally regardless of insertion history. Same
/// per-type-deterministic-hasher requirement as the [`PartialOrd`] impl.
/// This makes `PoMap` usable as a key in hash maps (including other `PoMap`s)
/// and in hashed sets — something unordered hash maps cannot lawfully offer.
impl<K: Key, V: Value + Hash, H: BuildHasher, const GROWTH: usize> Hash
    for PoMap<K, V, H, GROWTH>
{
    #[inline]
    fn hash<S: core::hash::Hasher>(&self, state: &mut S) {
        state.write_usize(self.len);
        for (k, v) in self.iter() {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone, const GROWTH: usize>
    core::ops::BitOr<&PoMap<K, V, H, GROWTH>> for &PoMap<K, V, H, GROWTH>
{
    type Output = PoMap<K, V, H, GROWTH>;

    /// [`union`](PoMap::union): `&a | &b` (right-biased on duplicate keys).
    #[inline]
    fn bitor(self, rhs: &PoMap<K, V, H, GROWTH>) -> Self::Output {
        self.union(rhs)
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone, const GROWTH: usize>
    core::ops::BitAnd<&PoMap<K, V, H, GROWTH>> for &PoMap<K, V, H, GROWTH>
{
    type Output = PoMap<K, V, H, GROWTH>;

    /// [`intersection`](PoMap::intersection): `&a & &b` (values from the left).
    #[inline]
    fn bitand(self, rhs: &PoMap<K, V, H, GROWTH>) -> Self::Output {
        self.intersection(rhs)
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone, const GROWTH: usize>
    core::ops::Sub<&PoMap<K, V, H, GROWTH>> for &PoMap<K, V, H, GROWTH>
{
    type Output = PoMap<K, V, H, GROWTH>;

    /// [`difference`](PoMap::difference): `&a - &b`.
    #[inline]
    fn sub(self, rhs: &PoMap<K, V, H, GROWTH>) -> Self::Output {
        self.difference(rhs)
    }
}

impl<K: Key, V: Value, H: BuildHasher + Clone, const GROWTH: usize>
    core::ops::BitXor<&PoMap<K, V, H, GROWTH>> for &PoMap<K, V, H, GROWTH>
{
    type Output = PoMap<K, V, H, GROWTH>;

    /// [`symmetric_difference`](PoMap::symmetric_difference): `&a ^ &b`.
    #[inline]
    fn bitxor(self, rhs: &PoMap<K, V, H, GROWTH>) -> Self::Output {
        self.symmetric_difference(rhs)
    }
}

impl<K: Key, V: Value, H: BuildHasher, Q, const GROWTH: usize> Index<&Q>
    for PoMap<K, V, H, GROWTH>
where
    K: Borrow<Q>,
    Q: Hash + Eq + ?Sized,
{
    type Output = V;

    /// # Panics
    ///
    /// Panics if the key is not present in the map.
    #[inline]
    fn index(&self, key: &Q) -> &Self::Output {
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
    /// Builds by insertion. (A sort-then-place bulk build was implemented and
    /// measured 4-7x SLOWER for cheap keys — the comparison sort of fat
    /// tuples costs more than hashing + low-load insertion; see
    /// docs/findings.md §6.2. Call [`PoMap::compact`] afterward if a
    /// canonical representation is needed.)
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
// Serde (optional)
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use serde::de::{MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serializes entries in canonical (hash, key) iteration order — equal
    /// maps produce **identical serialized output**, regardless of the
    /// construction history (the determinism contract applied to the wire).
    impl<K, V, H, const GROWTH: usize> Serialize for PoMap<K, V, H, GROWTH>
    where
        K: Key + Serialize,
        V: Value + Serialize,
        H: BuildHasher,
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(self.len()))?;
            for (k, v) in self.iter() {
                map.serialize_entry(k, v)?;
            }
            map.end()
        }
    }

    impl<'de, K, V, H, const GROWTH: usize> Deserialize<'de> for PoMap<K, V, H, GROWTH>
    where
        K: Key + Deserialize<'de>,
        V: Value + Deserialize<'de>,
        H: BuildHasher + Default,
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct PoMapVisitor<K, V, H, const GROWTH: usize>(
                PhantomData<(K, V, H)>,
            );

            impl<'de, K, V, H, const GROWTH: usize> Visitor<'de> for PoMapVisitor<K, V, H, GROWTH>
            where
                K: Key + Deserialize<'de>,
                V: Value + Deserialize<'de>,
                H: BuildHasher + Default,
            {
                type Value = PoMap<K, V, H, GROWTH>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a map")
                }

                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut access: A,
                ) -> Result<Self::Value, A::Error> {
                    let mut map = PoMap::with_capacity_and_hasher(
                        access.size_hint().unwrap_or(0),
                        H::default(),
                    );
                    while let Some((k, v)) = access.next_entry()? {
                        map.insert(k, v);
                    }
                    Ok(map)
                }
            }

            deserializer.deserialize_map(PoMapVisitor(PhantomData))
        }
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

    /// Hasher that collides everything: every key hashes to the same value,
    /// forcing the full-hash-collision tie path on every insert.
    #[derive(Clone, Default)]
    struct ColliderBuildHasher;
    struct ColliderHasher;
    impl core::hash::Hasher for ColliderHasher {
        fn finish(&self) -> u64 {
            0xDEAD_BEEF
        }
        fn write(&mut self, _: &[u8]) {}
    }
    impl BuildHasher for ColliderBuildHasher {
        type Hasher = ColliderHasher;
        fn build_hasher(&self) -> ColliderHasher {
            ColliderHasher
        }
    }

    /// Under total hash collision, ties must be canonicalized by key order:
    /// iteration order, Eq, Ord, and Hash are functions of contents alone,
    /// independent of insertion history.
    #[test]
    fn collision_ties_are_canonical() {
        let keys: [u64; 7] = [42, 3, 99, 7, 55, 1, 88];
        let mut a: PoMap<u64, u64, ColliderBuildHasher> =
            PoMap::with_hasher(ColliderBuildHasher);
        for &k in &keys {
            a.insert(k, k * 10);
        }
        let mut b: PoMap<u64, u64, ColliderBuildHasher> =
            PoMap::with_hasher(ColliderBuildHasher);
        for &k in keys.iter().rev() {
            b.insert(k, k * 10);
        }

        // Iteration is key-sorted within the collision run, for both maps.
        let seq_a: alloc::vec::Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        let seq_b: alloc::vec::Vec<u64> = b.iter().map(|(k, _)| *k).collect();
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        assert_eq!(seq_a, sorted);
        assert_eq!(seq_b, sorted);

        // Content-equal maps built in opposite orders are ==, Ordering::Equal,
        // and hash identically.
        assert_eq!(a, b);
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);
        let hash_of = |m: &PoMap<u64, u64, ColliderBuildHasher>| {
            let mut h = ahash::AHasher::default();
            m.hash(&mut h);
            core::hash::Hasher::finish(&h)
        };
        assert_eq!(hash_of(&a), hash_of(&b));

        // Removing from the middle of a collision run keeps canonical order.
        a.remove(&55);
        b.remove(&55);
        assert_eq!(a, b);
        let seq: alloc::vec::Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        let mut expect = sorted.clone();
        expect.retain(|&k| k != 55);
        assert_eq!(seq, expect);
    }

    /// After `compact()`, two content-equal maps built by wildly different
    /// histories are byte-identical (u64 payloads: no padding, no pointers).
    #[test]
    fn compact_produces_canonical_bytes() {
        // History A: build-from-empty in ascending order, with transient
        // entries inserted and removed along the way (growth path exercised).
        let mut a: PoMap<u64, u64> = PoMap::new();
        for k in 0..500u64 {
            a.insert(k, k * 3);
            if k % 7 == 0 {
                a.insert(1_000_000 + k, 1); // transient
            }
        }
        for k in 0..500u64 {
            if k % 7 == 0 {
                a.remove(&(1_000_000 + k));
            }
        }

        // History B: pre-provisioned table (different initial geometry),
        // descending insertion order, its own transient churn.
        let mut b: PoMap<u64, u64> = PoMap::with_capacity(4096);
        for k in (0..500u64).rev() {
            b.insert(k, k * 3);
        }
        for k in 0..50u64 {
            b.insert(2_000_000 + k, 9);
        }
        for k in 0..50u64 {
            b.remove(&(2_000_000 + k));
        }

        assert_eq!(a, b);
        a.compact();
        b.compact();
        assert_eq!(a, b, "compact must preserve contents");

        // Structural identity: same geometry...
        assert_eq!(a.slots.total_slots, b.slots.total_slots);
        assert_eq!(a.slots.layout.size(), b.slots.layout.size());
        // ...and byte identity of the full allocations.
        let bytes_a =
            unsafe { core::slice::from_raw_parts(a.slots.ptr.as_ptr(), a.slots.layout.size()) };
        let bytes_b =
            unsafe { core::slice::from_raw_parts(b.slots.ptr.as_ptr(), b.slots.layout.size()) };
        assert_eq!(bytes_a, bytes_b, "compacted representations must be canonical");

        // And a third, freshly built + compacted map agrees too.
        let mut c: PoMap<u64, u64> = PoMap::new();
        for k in (0..500u64).step_by(2).chain((0..500u64).skip(1).step_by(2)) {
            c.insert(k, k * 3);
        }
        c.compact();
        let bytes_c =
            unsafe { core::slice::from_raw_parts(c.slots.ptr.as_ptr(), c.slots.layout.size()) };
        assert_eq!(bytes_a, bytes_c);

        // Map still fully functional after compaction.
        for k in 0..500u64 {
            assert_eq!(a.get(&k), Some(&(k * 3)));
        }
    }

    /// Adversarial hasher that clusters every key into the topmost few ideal
    /// slots, piling maximal-hash runs against the end of the table. This
    /// exercises the trailing-vacant invariant: the final slot must never be
    /// occupied (insert landing guards), repacks must retry on overflow, and
    /// probes for absent larger-hash keys must terminate in-bounds.
    #[derive(Clone, Default)]
    struct TailClusterBuildHasher;
    struct TailClusterHasher(u64);
    impl core::hash::Hasher for TailClusterHasher {
        fn finish(&self) -> u64 {
            // 16 distinct hash values, all within 16 of u64::MAX.
            u64::MAX - 16 + (self.0 & 15)
        }
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64);
            }
        }
    }
    impl BuildHasher for TailClusterBuildHasher {
        type Hasher = TailClusterHasher;
        fn build_hasher(&self) -> TailClusterHasher {
            TailClusterHasher(0)
        }
    }

    #[test]
    fn adversarial_tail_clustering_is_safe() {
        let mut m: PoMap<u64, u64, TailClusterBuildHasher> =
            PoMap::with_hasher(TailClusterBuildHasher);
        let mut reference = std::collections::HashMap::new();

        // Insert far more entries than any padding region holds, all clustered
        // at the top of the hash range; interleave removes and lookups.
        for k in 0..120u64 {
            m.insert(k, k * 7);
            reference.insert(k, k * 7);
            if k % 3 == 0 {
                m.remove(&(k / 2));
                reference.remove(&(k / 2));
            }
            // The final slot must remain vacant at all times.
            assert_eq!(
                m.slots.hash_at(m.slots.total_slots - 1),
                EMPTY_HASH,
                "trailing-vacant invariant violated at k={k}"
            );
            // Absent keys probing at/above the topmost run terminate safely.
            assert_eq!(m.get(&(1_000_000 + k)), None);
        }
        for (k, v) in &reference {
            assert_eq!(m.get(k), Some(v));
        }
        assert_eq!(m.len(), reference.len());

        // compact() under clustering: the minimal target can't hold the tail
        // run, so the repack retry loop must engage — and stay canonical.
        let mut m2: PoMap<u64, u64, TailClusterBuildHasher> =
            PoMap::with_hasher(TailClusterBuildHasher);
        for (k, v) in &reference {
            m2.insert(*k, *v);
        }
        m.compact();
        m2.compact();
        assert_eq!(m.slots.hash_at(m.slots.total_slots - 1), EMPTY_HASH);
        assert_eq!(m.slots.total_slots, m2.slots.total_slots);
        let ba = unsafe {
            core::slice::from_raw_parts(m.slots.ptr.as_ptr(), m.slots.layout.size())
        };
        let bb = unsafe {
            core::slice::from_raw_parts(m2.slots.ptr.as_ptr(), m2.slots.layout.size())
        };
        assert_eq!(ba, bb, "compact() must stay canonical under the retry path");
        for (k, v) in &reference {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn borrowed_key_lookups() {
        let mut m: PoMap<std::string::String, u32> = PoMap::new();
        m.insert("alpha".into(), 1);
        m.insert("beta".into(), 2);
        assert_eq!(m.get("alpha"), Some(&1));
        assert!(m.contains_key("beta"));
        assert_eq!(m["beta"], 2);
        assert_eq!(m.get_key_value("alpha").map(|(k, _)| k.as_str()), Some("alpha"));
        *m.get_mut("beta").unwrap() = 20;
        assert_eq!(m.remove("beta"), Some(20));
        assert_eq!(m.get("beta"), None);
    }

    #[test]
    fn entry_api() {
        let mut m: PoMap<u64, u64> = PoMap::new();
        assert_eq!(*m.entry(1).or_insert(10), 10);
        assert_eq!(*m.entry(1).or_insert(99), 10); // occupied: keeps existing
        *m.entry(1).or_insert(0) += 5;
        assert_eq!(m[&1], 15);
        m.entry(2).and_modify(|v| *v += 1).or_insert(100);
        assert_eq!(m[&2], 100);
        m.entry(2).and_modify(|v| *v += 1).or_insert(100);
        assert_eq!(m[&2], 101);
        assert_eq!(*m.entry(3).or_default(), 0);
        match m.entry(1) {
            Entry::Occupied(o) => {
                assert_eq!(*o.key(), 1);
                assert_eq!(o.remove(), 15);
            }
            Entry::Vacant(_) => panic!("expected occupied"),
        }
        assert_eq!(m.get(&1), None);
        match m.entry(4) {
            Entry::Vacant(v) => {
                assert_eq!(v.into_key(), 4);
            }
            Entry::Occupied(_) => panic!("expected vacant"),
        }
        assert_eq!(m.get(&4), None);
        // Entry under growth: fill enough to trigger a grow via entries.
        for k in 10..200u64 {
            *m.entry(k).or_insert(k) += 1;
        }
        for k in 10..200u64 {
            assert_eq!(m[&k], k + 1);
        }
    }

    #[test]
    fn first_last_pop_deterministic() {
        let mk = |order: &[u64]| {
            let mut m: PoMap<u64, u64> = PoMap::new();
            for &k in order {
                m.insert(k, k);
            }
            m
        };
        let a = mk(&[1, 2, 3, 4, 5]);
        let b = mk(&[5, 3, 1, 4, 2]);
        assert_eq!(a.first_key_value(), b.first_key_value());
        assert_eq!(a.last_key_value(), b.last_key_value());

        // pop_first drains both maps in the identical canonical order.
        let (mut a, mut b) = (a, b);
        let da: alloc::vec::Vec<_> = core::iter::from_fn(|| a.pop_first()).collect();
        let db: alloc::vec::Vec<_> = core::iter::from_fn(|| b.pop_first()).collect();
        assert_eq!(da, db);
        assert_eq!(da.len(), 5);
        assert!(a.is_empty());

        // pop_last is the reverse drain.
        let mut c = mk(&[1, 2, 3, 4, 5]);
        let dc: alloc::vec::Vec<_> = core::iter::from_fn(|| c.pop_last()).collect();
        let mut rev = da.clone();
        rev.reverse();
        assert_eq!(dc, rev);
    }

    #[test]
    fn set_algebra_streaming_merges() {
        let mk = |pairs: &[(u64, u64)]| -> PoMap<u64, u64> {
            pairs.iter().copied().collect()
        };
        let a = mk(&[(1, 10), (2, 10), (3, 10), (5, 10)]);
        let b = mk(&[(2, 20), (3, 20), (4, 20), (6, 20)]);

        let u = a.union(&b);
        assert_eq!(u.len(), 6);
        assert_eq!(u[&1], 10);
        assert_eq!(u[&2], 20); // right-biased
        assert_eq!(u[&3], 20);
        assert_eq!(u[&4], 20);

        let i = a.intersection(&b);
        assert_eq!(i.len(), 2);
        assert_eq!(i[&2], 10); // values from the left
        assert_eq!(i[&3], 10);
        assert_eq!(i.get(&1), None);

        let d = a.difference(&b);
        assert_eq!(d.len(), 2);
        assert!(d.contains_key(&1) && d.contains_key(&5));

        let x = a.symmetric_difference(&b);
        assert_eq!(x.len(), 4);
        assert!(x.contains_key(&1) && x.contains_key(&4));
        assert!(!x.contains_key(&2) && !x.contains_key(&3));

        // Operators mirror the methods.
        assert_eq!(&a | &b, u);
        assert_eq!(&a & &b, i);
        assert_eq!(&a - &b, d);
        assert_eq!(&a ^ &b, x);

        // Identities on edge cases.
        let e: PoMap<u64, u64> = PoMap::new();
        assert_eq!(a.union(&e), a);
        assert_eq!(e.union(&a), a);
        assert_eq!(a.intersection(&e).len(), 0);
        assert_eq!(a.difference(&e), a);

        // append: right-biased move-union, other emptied.
        let mut a2 = mk(&[(1, 10), (2, 10)]);
        let mut b2 = mk(&[(2, 20), (3, 20)]);
        a2.append(&mut b2);
        assert!(b2.is_empty());
        assert_eq!(a2.len(), 3);
        assert_eq!(a2[&2], 20);
    }

    /// Merge outputs are compact-canonical: union(a, b) is byte-identical to
    /// building the same contents by insertion and calling compact().
    #[test]
    fn union_is_compact_canonical() {
        let mut a: PoMap<u64, u64> = PoMap::new();
        let mut b: PoMap<u64, u64> = PoMap::new();
        for k in 0..300u64 {
            if k % 2 == 0 {
                a.insert(k, k);
            }
            if k % 3 == 0 {
                b.insert(k, k + 1000);
            }
        }
        let u = a.union(&b);

        let mut expect: PoMap<u64, u64> = PoMap::new();
        for k in 0..300u64 {
            if k % 2 == 0 {
                expect.insert(k, k);
            }
            if k % 3 == 0 {
                expect.insert(k, k + 1000); // later wins = right bias
            }
        }
        expect.compact();

        assert_eq!(u, expect);
        assert_eq!(u.slots.total_slots, expect.slots.total_slots);
        let bu = unsafe {
            core::slice::from_raw_parts(u.slots.ptr.as_ptr(), u.slots.layout.size())
        };
        let be = unsafe {
            core::slice::from_raw_parts(expect.slots.ptr.as_ptr(), expect.slots.layout.size())
        };
        assert_eq!(bu, be, "merge outputs must be compact-canonical");
    }

    /// Set algebra under total hash collision: the (hash, key) tie order makes
    /// the merge walk see strictly increasing sequences on both sides.
    #[test]
    fn set_algebra_under_collisions() {
        let mk = |keys: &[u64]| -> PoMap<u64, u64, ColliderBuildHasher> {
            let mut m = PoMap::with_hasher(ColliderBuildHasher);
            for &k in keys {
                m.insert(k, k);
            }
            m
        };
        let a = mk(&[5, 1, 9, 3]);
        let b = mk(&[3, 7, 1]);
        let u = a.union(&b);
        assert_eq!(u.len(), 5);
        let keys: alloc::vec::Vec<u64> = u.keys().copied().collect();
        assert_eq!(keys, alloc::vec![1, 3, 5, 7, 9]); // canonical tie order
        let i = a.intersection(&b);
        assert_eq!(i.len(), 2);
        assert!(i.contains_key(&1) && i.contains_key(&3));
        let x = a.symmetric_difference(&b);
        assert_eq!(x.len(), 3);
    }

    /// Bulk from_iter: HashMap duplicate semantics (last wins), canonical
    /// output bytes, and correctness under total hash collision.
    #[test]
    fn bulk_from_iter() {
        // Later duplicates win, matching insertion semantics.
        let m: PoMap<u64, u64> = [(1, 10), (2, 20), (1, 11), (3, 30), (2, 22)]
            .into_iter()
            .collect();
        assert_eq!(m.len(), 3);
        assert_eq!(m[&1], 11);
        assert_eq!(m[&2], 22);

        // Equal to insert-building; canonical after compact() on both.
        let mut bulk: PoMap<u64, u64> = (0..500u64).map(|k| (k, k * 3)).collect();
        let mut inserted: PoMap<u64, u64> = PoMap::new();
        for k in 0..500u64 {
            inserted.insert(k, k * 3);
        }
        assert_eq!(bulk, inserted);
        bulk.compact();
        inserted.compact();
        let ba = unsafe {
            core::slice::from_raw_parts(bulk.slots.ptr.as_ptr(), bulk.slots.layout.size())
        };
        let bb = unsafe {
            core::slice::from_raw_parts(
                inserted.slots.ptr.as_ptr(),
                inserted.slots.layout.size(),
            )
        };
        assert_eq!(ba, bb);

        // Under total collision: dedup + tie order still correct.
        let c: PoMap<u64, u64, ColliderBuildHasher> =
            [(9, 1), (1, 1), (9, 2), (5, 1)].into_iter().collect();
        assert_eq!(c.len(), 3);
        assert_eq!(c[&9], 2); // last duplicate wins
        let keys: alloc::vec::Vec<u64> = c.keys().copied().collect();
        assert_eq!(keys, alloc::vec![1, 5, 9]);

        // Empty and by-ref forms.
        let e: PoMap<u64, u64> = core::iter::empty::<(u64, u64)>().collect();
        assert!(e.is_empty());
        let r: PoMap<u64, u64> = m.iter().collect();
        assert_eq!(r, m);
    }

    /// Serde round-trip + canonical wire form (equal maps, different
    /// histories, identical JSON).
    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_and_canonical_wire() {
        let mut a: PoMap<u64, u64> = PoMap::new();
        let mut b: PoMap<u64, u64> = PoMap::with_capacity(1024);
        for k in 0..100u64 {
            a.insert(k, k * 2);
        }
        for k in (0..100u64).rev() {
            b.insert(k, k * 2);
        }
        let ja = serde_json::to_string(&a).unwrap();
        let jb = serde_json::to_string(&b).unwrap();
        assert_eq!(ja, jb, "equal maps must serialize identically");
        let back: PoMap<u64, u64> = serde_json::from_str(&ja).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn clone_from_reuses_allocation() {
        let src: PoMap<u64, u64> = (0..200u64).map(|k| (k, k)).collect();
        // Same geometry: allocation must be reused and contents mirrored.
        let mut dst: PoMap<u64, u64> = (1000..1200u64).map(|k| (k, k)).collect();
        assert_eq!(dst.slots.total_slots, src.slots.total_slots);
        let ptr_before = dst.slots.ptr.as_ptr();
        dst.clone_from(&src);
        assert_eq!(dst.slots.ptr.as_ptr(), ptr_before, "allocation must be reused");
        assert_eq!(dst, src);
        // Slot-for-slot mirror = bitwise-equal tables for u64 payloads.
        let ba = unsafe {
            core::slice::from_raw_parts(dst.slots.ptr.as_ptr(), dst.slots.layout.size())
        };
        let bb = unsafe {
            core::slice::from_raw_parts(src.slots.ptr.as_ptr(), src.slots.layout.size())
        };
        assert_eq!(ba, bb);
        // Different geometry: falls back to a fresh clone.
        let big: PoMap<u64, u64> = (0..50_000u64).map(|k| (k, k)).collect();
        dst.clone_from(&big);
        assert_eq!(dst, big);
    }

    /// Ord over maps is a lawful total order on content-distinct maps.
    #[test]
    fn map_ord_laws() {
        use core::cmp::Ordering;
        let mk = |pairs: &[(u64, u64)]| {
            let mut m: PoMap<u64, u64> = PoMap::new();
            for &(k, v) in pairs {
                m.insert(k, v);
            }
            m
        };
        let x = mk(&[(1, 1), (2, 2)]);
        let y = mk(&[(1, 1), (2, 3)]); // differs in one value
        let z = mk(&[(1, 1)]);

        assert_eq!(x.cmp(&x), Ordering::Equal);
        assert_eq!(x.cmp(&y), y.cmp(&x).reverse());
        assert_eq!(x.cmp(&z), z.cmp(&x).reverse());
        // transitivity over the three distinct maps, whatever the direction:
        let mut v = [&x, &y, &z];
        v.sort();
        assert!(v[0].cmp(v[1]) != Ordering::Greater && v[1].cmp(v[2]) != Ordering::Greater);
        assert!(v[0].cmp(v[2]) != Ordering::Greater);
    }
}
