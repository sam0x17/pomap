use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};
use std::hash::DefaultHasher;

/// Minimum capacity we will allow for PoMap
const MIN_CAPACITY: usize = 16;

/// Maximum number of slots to linearly scan starting at the ideal slot.
/// We design the layout so that `[ideal_slot, ideal_slot + MAX_SCAN)` is
/// always in-bounds for the backing Vec.
const MAX_SCAN: usize = 4;

/// Number of bits in the hashcode
const HASH_BITS: usize = 64; // we use a 64-bit hashcode

const GROWTH_FACTOR: usize = 2;

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
pub struct PoMap<K: Key, V: Value, H: Hasher + Default = DefaultHasher> {
    meta: PoMapMeta,
    slots: Vec<Slot<K, V>>,
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
        let mut slots = Vec::with_capacity(vec_capacity);
        slots.resize_with(vec_capacity, || Slot::Vacant);
        Self {
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

    /// Hot path for `get` that can be used when the caller already has the hash and doesn't
    /// want to recompute it.
    #[inline]
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
                return None;
            };

            let slot_hash = *slot_hash;

            if slot_hash < hash {
                continue;
            }
            // we guarantee the slots are sorted by hash
            if slot_hash > hash {
                return None;
            }
            if slot_key == key {
                return Some(value);
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
        let hash = hasher.finish();
        self.get_with_hash(hash, key)
    }

    /// Inserts a key-value pair into the map, replacing any existing value.
    ///
    /// Computes the hash of the key and then delegates to [`Self::insert_with_hash`].
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        self.insert_with_hash(hash, key, value)
    }

    #[inline]
    pub fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        let key = key;
        let value = value;

        loop {
            let ideal_slot = self.meta.ideal_slot(hash);
            let scan_end = ideal_slot + MAX_SCAN;

            for idx in ideal_slot..scan_end {
                if matches!(&self.slots[idx], Slot::Vacant) {
                    self.slots[idx] = Slot::Occupied { hash, key, value };
                    return None;
                }

                let slot_hash = match &self.slots[idx] {
                    Slot::Occupied {
                        hash: slot_hash, ..
                    } => *slot_hash,
                    Slot::Vacant => unreachable!(),
                };

                if slot_hash < hash {
                    continue;
                }
                // we guarantee the slots are sorted by hash
                if slot_hash > hash {
                    // we need to insert before this slot
                    // first we need to make sure we have room to shift elements at all within our
                    // MAX_SCAN boundary. If there is at least one free spot past this slot and
                    // before scan_end, we can shift elements to make room.
                    if let Some(offset) = self.slots[idx + 1..scan_end]
                        .iter()
                        .position(|slot| matches!(slot, Slot::Vacant))
                    {
                        let vacant_idx = idx + 1 + offset;
                        // Shift elements right by one to make space for the new element.
                        self.slots[idx..=vacant_idx].rotate_right(1);
                        self.slots[idx] = Slot::Occupied { hash, key, value };
                        return None;
                    }
                    // No room in this window; grow and retry.
                    break;
                }
                if let Slot::Occupied {
                    key: slot_key,
                    value: slot_value,
                    ..
                } = &mut self.slots[idx]
                {
                    if *slot_key == key {
                        let old_value = core::mem::replace(slot_value, value);
                        return Some(old_value);
                    }
                }
            }

            // If we make it here, we either ran out of room in the scan window or the table is
            // saturated for this hash window. Grow and retry the insertion.
            self.reserve(self.slots.capacity() * GROWTH_FACTOR);
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
        let new_slots: Vec<Slot<K, V>> = Vec::with_capacity(new_vec_capacity);

        // re-insert all existing elements into the new vec in the same order, with spaces
        // added based on hash prefix / the new meta
        let old_slots = core::mem::replace(&mut self.slots, new_slots);
        let mut cursor = 0;
        for slot in old_slots.into_iter() {
            let Slot::Occupied { hash, .. } = &slot else {
                continue;
            };

            // calculate ideal slot in the new layout
            let ideal_slot = new_meta.ideal_slot(*hash);

            // advance cursor and fill in vacant slots as needed
            for _ in cursor..ideal_slot {
                self.slots.push(Slot::Vacant);
            }
            cursor += ideal_slot.saturating_sub(cursor) + 1;

            // insert the slot, we should be at or past the ideal slot now. Because the
            // previous layout was already valid and is strictly smaller than the new one, this
            // can never cause us to exceed MAX_SCAN slots past the ideal slot because we are
            // always gaining more room.
            self.slots.push(slot.clone());
        }

        // fill remaining capacity with vacant slots (minimally to account for MAX_SCAN, or a
        // sparse set of elements)
        self.slots.resize_with(new_vec_capacity, || Slot::Vacant);

        // apply the new meta
        self.meta = new_meta;

        debug_assert!(self.slots.capacity() == new_vec_capacity);

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
