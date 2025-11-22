use core::{cmp::Ordering, hash::Hash};

/// Minimum capacity we will allow for PoMap
pub const MIN_CAPACITY: usize = 16;

/// Maximum number of slots to linearly scan starting at the ideal slot.
/// We design the layout so that `[ideal_slot, ideal_slot + MAX_SCAN)` is
/// always in-bounds for the backing Vec.
pub const MAX_SCAN: usize = 4;

/// Number of bits in the hashcode
const HASH_BITS: usize = 64; // we use a 64-bit hashcode

pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

pub trait Value: Clone {}
impl<V: Clone> Value for V {}

#[derive(Hash, Clone, Debug)]
enum Slot<K: Key, V: Value> {
    Vacant,
    Occupied { hash: u64, key: K, value: V },
}

#[derive(Clone)]
pub struct PoMap<K: Key, V: Value> {
    meta: PoMapMeta,
    slots: Vec<Slot<K, V>>,
}

impl<K: Key, V: Value> PoMap<K, V> {
    /// Create a new [`PoMap`] with _at least_ the given capacity.
    ///
    /// Note that the actual internal capacity will always be scaled up to the next power of
    /// two (if not already a power of two) plus [`MAX_SCAN`].
    ///
    /// Also note that the minimum capacity is [`MIN_CAPACITY`].
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let (meta, vec_capacity) = PoMapMeta::new(capacity);
        let mut slots = Vec::with_capacity(vec_capacity);
        slots.resize_with(vec_capacity, || Slot::Vacant);
        Self { meta, slots }
    }

    /// Create a new [`PoMap`] with [`MIN_CAPACITY`] + [`MAX_SCAN`] internal capacity.
    #[inline(always)]
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    /// Hot path for `get` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline(always)]
    pub fn get_with_hash(&self, hash: u64, key: &K) -> Option<&V> {
        let ideal_slot = self.meta.ideal_slot(hash);
        let scan_end = ideal_slot + MAX_SCAN;

        for slot in &self.slots[ideal_slot..scan_end] {
            let Slot::Occupied {
                hash: slot_hash,
                key: slot_key,
                value,
            } = slot
            else {
                break;
            };

            // we guarantee the slots are sorted by hash
            if *slot_hash > hash {
                break;
            }
            if *slot_hash == hash && slot_key == key {
                return Some(value);
            }
        }
        None
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
    /// for the backing `Vec<Slot<..>>`.
    #[inline(always)]
    const fn new(requested: usize) -> (Self, usize) {
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
