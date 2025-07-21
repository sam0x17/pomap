use std::{
    cmp::Ordering,
    hash::{DefaultHasher, Hash, Hasher},
};

use wide::u64x4;

use crate::simd::{find_match_index, find_match_index_any_of_2};

pub mod simd;

const EMPTY: u64 = u64::MIN;
const BUCKET_LEN: usize = 4;
const NUM_SWAPS_ALLOWED: usize = 1;

#[cfg(feature = "compare-keys")]
trait Key: Hash + Eq + Clone + Ord {}
#[cfg(feature = "compare-keys")]
impl<K: Hash + Eq + Clone + Ord> Key for K {}
#[cfg(not(feature = "compare-keys"))]
trait Key: Hash + Eq + Clone {}
#[cfg(not(feature = "compare-keys"))]
impl<K: Hash + Eq + Clone> Key for K {}

trait Value: Clone {}
impl<V: Clone> Value for V {}

#[derive(Hash, Clone, Debug)]
struct Entry<K: Key, V: Value> {
    hash: u64,
    key: K,
    value: V,
}

#[derive(Clone)]
pub struct PoMap<K: Key, V: Value> {
    p_bits: u8, // Number of prefix bits to go from global -> slot index, also bucket size, also log2(capacity)
    len: usize,
    entries: Vec<Option<Entry<K, V>>>,
}

impl<K: Key, V: Value> Default for PoMap<K, V> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, V: Value> PoMap<K, V> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            p_bits: 0,
            len: 0,
            entries: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.p_bits = 0;
        self.len = 0;
        self.entries.clear();
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    #[inline(always)]
    pub fn with_capacity(mut capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        capacity = (capacity.max(2)).next_power_of_two();
        Self {
            p_bits: capacity.trailing_zeros() as u8,
            len: 0,
            entries: vec![None; capacity],
        }
    }

    #[inline(always)]
    fn calculate_hash(key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        //(hash >> 1) + 1 // clamp to [1, u64::MAX - 1] so we can use these as control values
        hash
    }

    #[inline(always)]
    fn bucket_index(&self, hash: u64) -> usize {
        // debug_assert!(self.p_bits > 0, "p_bits must be greater than 0");
        // println!("({} >> (64 - {})) as usize", hash, self.p_bits);
        (hash >> (64 - self.p_bits)) as usize
    }

    #[inline(always)]
    fn entry(&self, index: usize) -> Option<&Entry<K, V>> {
        // Safety: index is guaranteed to be within bounds of self.entries
        unsafe { self.entries.get_unchecked(index).as_ref() }
    }

    #[inline(always)]
    fn entry_mut(&mut self, index: usize) -> &mut Option<Entry<K, V>> {
        // Safety: index is guaranteed to be within bounds of self.entries
        unsafe { self.entries.get_unchecked_mut(index) }
    }

    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.len == 0 {
            return None;
        }
        let hash = Self::calculate_hash(key);
        let bucket_index = self.bucket_index(hash);
        #[cfg(feature = "binary-search")]
        if let Ok(index) = (&self.entries[bucket_index..(bucket_index + self.p_bits as usize)])
            .binary_search_by(|e| {
                match e {
                    #[cfg(not(feature = "compare-keys"))]
                    Some(entry) => entry.hash.cmp(&hash),
                    #[cfg(feature = "compare-keys")]
                    Some(entry) => entry.hash.cmp(&hash).then_with(|| entry.key.cmp(key)),
                    None => Ordering::Greater, // treat None as larger than any hash
                }
            })
        {
            return self.entries[bucket_index + index]
                .as_ref()
                .map(|e| &e.value);
        }
        #[cfg(not(feature = "binary-search"))]
        for i in bucket_index..(bucket_index + self.p_bits as usize) {
            let Some(entry) = self.entry(i) else {
                // no entry here, not found
                break;
            };
            match entry.hash.cmp(&hash) {
                #[cfg(not(feature = "compare-keys"))]
                Ordering::Equal => return Some(&entry.value),
                #[cfg(feature = "compare-keys")]
                Ordering::Equal if entry.key == *key => return Some(&entry.value),
                Ordering::Less => break, // hash exceeded (they are sorted), not found
                _ => continue,
            }
        }
        None
    }

    #[inline(always)]
    fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        if self.entries.is_empty() {
            debug_assert_eq!(self.len, 0);
            // map is empty
            // fill with 4 buckets of 4
            self.entries.append(&mut vec![None; 16]);
            self.p_bits = 4;
        }
        loop {
            let bucket_index = self.bucket_index(hash);
            #[cfg(not(feature = "binary-search"))]
            let start_index = bucket_index;
            let end_index = bucket_index + self.p_bits as usize;
            #[cfg(feature = "binary-search")]
            let start_index = match (&self.entries[bucket_index..end_index]).binary_search_by(|e| {
                match e {
                    #[cfg(not(feature = "compare-keys"))]
                    Some(entry) => entry.hash.cmp(&hash),
                    #[cfg(feature = "compare-keys")]
                    Some(entry) => entry.hash.cmp(&hash).then_with(|| entry.key.cmp(key)),
                    None => Ordering::Greater, // treat None as larger than any hash
                }
            }) {
                Ok(index) => {
                    // found an entry with the same hash
                    let entry_index = bucket_index + index;
                    let entry = self.entry_mut(entry_index);
                    #[cfg(not(feature = "compare-keys"))]
                    if let Some(existing_entry) = entry {
                        // overwrite existing value
                        let old_value = std::mem::replace(&mut existing_entry.value, value);
                        return Some(old_value);
                    }
                    #[cfg(feature = "compare-keys")]
                    if let Some(existing_entry) = entry {
                        if existing_entry.key == key {
                            // overwrite existing value
                            let old_value = std::mem::replace(&mut existing_entry.value, value);
                            return Some(old_value);
                        }
                    }
                    // we already know this entry is occupied, so the next possible insertion point
                    // is the next index
                    bucket_index + index + 1
                }
                Err(index) => bucket_index + index,
            };
            for i in start_index..end_index {
                match self.entry_mut(i) {
                    Some(entry) => {
                        if entry.hash > hash {
                            // this element is larger, so we need to insert here, see if we can
                            // swap elements out of the
                            // TODO: limit number of swaps here optionally
                            for j in i..end_index {
                                if self.entry(j).is_none() {
                                    // Found an empty slot at `j`. Shift elements from `i` to `j-1` one position to the right.
                                    for k in (i..j).rev() {
                                        self.entries[k + 1] = self.entries[k].clone();
                                    }
                                    // The slot at `i` is now free.
                                    *self.entry_mut(i) = Some(Entry { hash, key, value });
                                    self.len += 1;
                                    return None;
                                }
                            }
                        }
                    }
                    None => {
                        // found an empty slot, insert here
                        *self.entry_mut(i) = Some(Entry { hash, key, value });
                        self.len += 1;
                        return None; // no previous value to return
                    }
                }
            }
            // must resize
            self.reserve(self.capacity() * 2);
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = Self::calculate_hash(&key);
        // println!("current state:\n{:#?}", self.hashes);
        // println!("[PoMap] Inserting key with hash: {}", hash);
        self.insert_with_hash(hash, key, value)
    }

    #[inline(always)]
    pub fn reserve(&mut self, capacity: usize) {
        if capacity <= self.capacity() {
            return; // no need to reserve more space
        }
        let new_capacity = (capacity.max(2)).next_power_of_two();
        if new_capacity > self.capacity() {
            // println!("Reserving new capacity: {}", new_capacity);
            let old_p_bits = self.p_bits;
            let old_capacity = self.capacity();
            self.p_bits = new_capacity.trailing_zeros() as u8;
            // the buckets themselves and their contents are already ordered by hash, so we can
            // directly copy into the new array just with different spacing based on the new
            // bucket boundaries
            let mut new_entries = vec![None; new_capacity];
            for bucket_index in 0..old_p_bits {
                let start_index = bucket_index * old_p_bits as usize;
                let end_index = start_index + old_p_bits as usize;
            }
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_insert() {
        let mut map = PoMap::new();
        assert_eq!(map.len(), 0);
        map.insert("key1".to_string(), 42);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&"key1".to_string()), Some(&42));
        map.insert("key2".to_string(), 84);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"key2".to_string()), Some(&84));
        map.insert("key1".to_string(), 100);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"key1".to_string()), Some(&100));
        map.insert("key3".to_string(), 200);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"key3".to_string()), Some(&200));
        map.insert("key4".to_string(), 300);
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&"key4".to_string()), Some(&300));
        map.insert("key5".to_string(), 400);
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&"key5".to_string()), Some(&400));
        map.insert("key6".to_string(), 500);
        assert_eq!(map.len(), 6);
        assert_eq!(map.get(&"key6".to_string()), Some(&500));
        map.insert("key7".to_string(), 600);
        assert_eq!(map.len(), 7);
        assert_eq!(map.get(&"key7".to_string()), Some(&600));
        map.insert("key8".to_string(), 700);
        assert_eq!(map.len(), 8);
        assert_eq!(map.get(&"key8".to_string()), Some(&700));
        map.insert("key9".to_string(), 800);
        assert_eq!(map.len(), 9);
        assert_eq!(map.get(&"key9".to_string()), Some(&800));
        map.insert("key10".to_string(), 900);
        assert_eq!(map.len(), 10);
        assert_eq!(map.get(&"key10".to_string()), Some(&900));
        map.insert("key11".to_string(), 1000);
        assert_eq!(map.len(), 11);
        assert_eq!(map.get(&"key11".to_string()), Some(&1000));
        map.insert("key12".to_string(), 1100);
        assert_eq!(map.len(), 12);
        assert_eq!(map.get(&"key12".to_string()), Some(&1100));
        map.insert("key13".to_string(), 1200);
        assert_eq!(map.len(), 13);
        assert_eq!(map.get(&"key13".to_string()), Some(&1200));
        map.insert("key14".to_string(), 1300);
        assert_eq!(map.len(), 14);
        assert_eq!(map.get(&"key14".to_string()), Some(&1300));
        map.insert("key15".to_string(), 1400);
        assert_eq!(map.len(), 15);
        assert_eq!(map.get(&"key15".to_string()), Some(&1400));
        map.insert("key16".to_string(), 1500);
        assert_eq!(map.len(), 16);
        assert_eq!(map.get(&"key16".to_string()), Some(&1500));
        // println!("state: {:#?}", map.hashes);
    }

    #[test]
    fn test_clear() {
        let mut map = PoMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);
        map.clear();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 0);
        assert_eq!(map.get(&"key1".to_string()), None);
        map.insert("key3".to_string(), 3);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&"key3".to_string()), Some(&3));
    }

    #[test]
    fn test_default() {
        let map: PoMap<String, i32> = PoMap::default();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 0);
    }

    #[test]
    fn test_get_nonexistent() {
        let mut map = PoMap::new();
        map.insert("key1".to_string(), 42);
        assert_eq!(map.get(&"key2".to_string()), None);
    }

    #[test]
    fn test_update() {
        let mut map = PoMap::new();
        map.insert("key1".to_string(), 42);
        let old_value = map.insert("key1".to_string(), 99);
        assert_eq!(map.len(), 1);
        assert_eq!(old_value, Some(42));
        assert_eq!(map.get(&"key1".to_string()), Some(&99));
    }

    // #[test]
    // fn test_remove() {
    //     let mut map = PoMap::new();
    //     map.insert("key1".to_string(), 1);
    //     map.insert("key2".to_string(), 2);
    //     assert_eq!(map.len(), 2);

    //     let removed = map.remove(&"key1".to_string());
    //     assert_eq!(removed, Some(1));
    //     assert_eq!(map.len(), 1);
    //     assert_eq!(map.get(&"key1".to_string()), None);
    //     assert_eq!(map.get(&"key2".to_string()), Some(&2));

    //     let non_existent = map.remove(&"key3".to_string());
    //     assert_eq!(non_existent, None);
    //     assert_eq!(map.len(), 1);
    // }

    #[test]
    fn test_resize_and_rehash() {
        let mut map = PoMap::with_capacity(16);
        assert_eq!(map.capacity(), 16);

        // Insert enough items to trigger a resize.
        // Load factor is 0.75, so 16 * 0.75 = 12. The 13th element will resize.
        for i in 0..12 {
            map.insert(i.to_string(), i);
        }
        assert_eq!(map.len(), 12);
        assert_eq!(map.capacity(), 64);

        map.insert("12".to_string(), 12);
        assert_eq!(map.len(), 13);
        assert_eq!(map.capacity(), 64);

        // Verify all old and new elements are present.
        for i in 0..13 {
            assert_eq!(map.get(&i.to_string()), Some(&i));
        }
    }

    // #[test]
    // fn test_many_insertions_and_removals() {
    //     let mut map = PoMap::new();
    //     let num_items = 1_000;

    //     for i in 0..num_items {
    //         map.insert(i, i * 2);
    //     }
    //     assert_eq!(map.len(), num_items);

    //     for i in (0..num_items).step_by(2) {
    //         assert_eq!(map.remove(&i), Some(i * 2));
    //     }
    //     assert_eq!(map.len(), num_items / 2);

    //     for i in 0..num_items {
    //         if i % 2 == 0 {
    //             assert_eq!(map.get(&i), None);
    //         } else {
    //             assert_eq!(map.get(&i), Some(&(i * 2)));
    //         }
    //     }
    // }

    #[derive(Debug, Clone, Eq)]
    struct CollidingKey {
        val: u64,
        hash: u64,
    }

    impl Hash for CollidingKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.hash.hash(state);
        }
    }

    impl PartialEq for CollidingKey {
        fn eq(&self, other: &Self) -> bool {
            self.val == other.val
        }
    }

    // #[test]
    // fn test_hash_collisions() {
    //     let mut map = PoMap::new();
    //     let key1 = CollidingKey { val: 1, hash: 123 };
    //     let key2 = CollidingKey { val: 2, hash: 123 };
    //     let key3 = CollidingKey { val: 3, hash: 123 };

    //     map.insert(key1.clone(), 10);
    //     map.insert(key2.clone(), 20);
    //     map.insert(key3.clone(), 30);

    //     assert_eq!(map.len(), 3);
    //     assert_eq!(map.get(&key1), Some(&10));
    //     assert_eq!(map.get(&key2), Some(&20));
    //     assert_eq!(map.get(&key3), Some(&30));

    //     // Test update on collision
    //     map.insert(key2.clone(), 25);
    //     assert_eq!(map.len(), 3);
    //     assert_eq!(map.get(&key2), Some(&25));

    //     // Test remove on collision
    //     assert_eq!(map.remove(&key1), Some(10));
    //     assert_eq!(map.len(), 2);
    //     assert_eq!(map.get(&key1), None);
    //     assert_eq!(map.get(&key2), Some(&25));
    //     assert_eq!(map.get(&key3), Some(&30));
    // }

    #[test]
    fn test_bucket_overflow_and_resize() {
        let mut map = PoMap::with_capacity(16);

        // Create keys that all map to the same bucket.
        // We can achieve this by crafting hashes. A key's bucket is `(hash >> (64 - p)) % num_buckets`.
        // With C=16, k=4, num_buckets=4, p=2.
        // We need `(hash >> 62) % 4` to be constant.
        let collision_hash_base: u64 = 1 << 62;

        for i in 0..4 {
            // These should all go to the same bucket.
            let key = CollidingKey {
                val: i,
                hash: collision_hash_base + i,
            };
            map.insert(key, i as i32);
        }
        assert_eq!(map.len(), 4);
        assert_eq!(map.capacity(), 16);

        // This should trigger a resize because the target bucket is full (len=k=4).
        let key = CollidingKey {
            val: 4,
            hash: collision_hash_base + 4,
        };
        map.insert(key, 4);
        assert_eq!(map.len(), 5);
        assert_eq!(map.capacity(), 16);

        // Check that all values are still there.
        for i in 0..5 {
            let key_to_check = CollidingKey {
                val: i,
                hash: collision_hash_base + i,
            };
            assert_eq!(map.get(&key_to_check), Some(&(i as i32)));
        }
    }

    // #[test]
    // fn test_random_insert_remove_comprehensive() {
    //     use std::collections::HashMap;

    //     let mut pomap = PoMap::new();
    //     let mut hashmap = HashMap::new();
    //     let num_items = 2_000;

    //     let mut items: Vec<(i32, i32)> = (0..num_items).map(|i| (i, i * 10)).collect();

    //     // Simple shuffle
    //     for i in 0..items.len() {
    //         let j = (i * 7) % items.len();
    //         items.swap(i, j);
    //     }

    //     // Insert all items
    //     for (k, v) in &items {
    //         pomap.insert(*k, *v);
    //         hashmap.insert(*k, *v);
    //         assert_eq!(pomap.len(), hashmap.len());
    //     }

    //     assert_eq!(pomap.len(), num_items as usize);
    //     for (k, v) in &items {
    //         assert_eq!(pomap.get(k), Some(v));
    //     }

    //     // Remove about half the items
    //     let mut removed_count = 0;
    //     for (i, (k, v)) in items.iter().enumerate() {
    //         if i % 2 == 0 {
    //             assert_eq!(pomap.remove(k), Some(*v));
    //             hashmap.remove(k);
    //             removed_count += 1;
    //         }
    //         assert_eq!(pomap.len(), hashmap.len());
    //     }

    //     assert_eq!(pomap.len(), (num_items - removed_count) as usize);

    //     // Check final state
    //     for (k, _) in &items {
    //         assert_eq!(pomap.get(k), hashmap.get(k));
    //     }
    // }

    #[test]
    fn test_zero_capacity() {
        let mut map: PoMap<i32, i32> = PoMap::with_capacity(0);
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 0);

        map.insert(1, 1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.capacity(), 16);
        assert_eq!(map.get(&1), Some(&1));
    }

    // #[test]
    // fn test_clone() {
    //     let mut map = PoMap::new();
    //     map.insert("a", 1);
    //     map.insert("b", 2);

    //     let mut clone = map.clone();
    //     assert_eq!(clone.len(), 2);
    //     assert_eq!(clone.get(&"a"), Some(&1));
    //     assert_eq!(clone.get(&"b"), Some(&2));

    //     // Modify the clone and check that the original is unaffected
    //     clone.insert("c", 3);
    //     clone.remove(&"a");
    //     assert_eq!(map.len(), 2);
    //     assert_eq!(map.get(&"a"), Some(&1));
    //     assert_eq!(map.get(&"c"), None);

    //     // Modify the original and check that the clone is unaffected
    //     map.insert("d", 4);
    //     assert_eq!(clone.len(), 2);
    //     assert_eq!(clone.get(&"d"), None);
    // }

    // #[test]
    // fn test_zero_sized_types() {
    //     // Test with ZST value
    //     let mut map_zst_val = PoMap::new();
    //     map_zst_val.insert(1, ());
    //     map_zst_val.insert(2, ());
    //     assert_eq!(map_zst_val.len(), 2);
    //     assert_eq!(map_zst_val.get(&1), Some(&()));
    //     assert_eq!(map_zst_val.remove(&1), Some(()));
    //     assert_eq!(map_zst_val.len(), 1);
    //     assert_eq!(map_zst_val.get(&2), Some(&()));

    //     // Test with ZST key
    //     // Since all keys are equal, it can only hold one item.
    //     let mut map_zst_key: PoMap<(), i32> = PoMap::new();
    //     assert_eq!(map_zst_key.insert((), 100), None);
    //     assert_eq!(map_zst_key.len(), 1);
    //     assert_eq!(map_zst_key.insert((), 200), Some(100));
    //     assert_eq!(map_zst_key.len(), 1);
    //     assert_eq!(map_zst_key.get(&()), Some(&200));
    // }

    // use std::sync::Arc;
    // use std::sync::atomic::{AtomicUsize, Ordering};

    // #[derive(Clone, Debug)]
    // struct DropCounter {
    //     id: i32,
    //     counter: Arc<AtomicUsize>,
    // }

    // impl Drop for DropCounter {
    //     fn drop(&mut self) {
    //         self.counter.fetch_add(1, Ordering::SeqCst);
    //     }
    // }

    // impl PartialEq for DropCounter {
    //     fn eq(&self, other: &Self) -> bool {
    //         self.id == other.id
    //     }
    // }

    // #[test]
    // fn test_drop_behavior() {
    //     let counter = Arc::new(AtomicUsize::new(0));

    //     {
    //         let mut map = PoMap::new();
    //         let make_droppable = |id| DropCounter {
    //             id,
    //             counter: Arc::clone(&counter),
    //         };

    //         // 1. Test drop on insert (overwrite)
    //         map.insert("a", make_droppable(1));
    //         assert_eq!(counter.load(Ordering::SeqCst), 0);
    //         map.insert("a", make_droppable(2)); // Overwrites key "a", should drop value 1
    //         assert_eq!(counter.load(Ordering::SeqCst), 1);

    //         // 2. Test drop on remove
    //         map.insert("b", make_droppable(3));
    //         assert_eq!(counter.load(Ordering::SeqCst), 1);
    //         map.remove(&"b"); // Removes key "b", should drop value 3
    //         assert_eq!(counter.load(Ordering::SeqCst), 2);

    //         // 3. Test drop on clear
    //         map.insert("c", make_droppable(4));
    //         map.insert("d", make_droppable(5));
    //         assert_eq!(counter.load(Ordering::SeqCst), 2);
    //         map.clear(); // Should drop values 2, 4, 5
    //         assert_eq!(counter.load(Ordering::SeqCst), 5);
    //         assert_eq!(map.len(), 0);

    //         // 4. Test drop on PoMap::drop
    //         map.insert("e", make_droppable(6));
    //         map.insert("f", make_droppable(7));
    //     } // map goes out of scope here

    //     // Should drop values 6, 7
    //     assert_eq!(counter.load(Ordering::SeqCst), 7);
    // }

    #[test]
    fn test_performance_vs_std_hashmap() {
        use std::collections::HashMap;
        use std::time::Instant;

        let num_items = 1_000_000;
        let mut items: Vec<(i32, i32)> = (0..num_items).map(|i| (i, i * 10)).collect();

        // Simple shuffle to make access less predictable
        for i in 0..items.len() {
            let j = (i * 13) % items.len();
            items.swap(i, j);
        }

        // --- PoMap Benchmark ---
        println!("\n--- Benchmarking PoMap ---");
        let mut pomap: PoMap<i32, i32> = PoMap::with_capacity(0);
        let start = Instant::now();
        for (k, v) in &items {
            pomap.insert(*k, *v);
        }
        let duration = start.elapsed();
        println!("PoMap insert {num_items} items: {duration:?}");

        let start = Instant::now();
        for (k, v) in &items {
            assert_eq!(pomap.get(k), Some(v));
        }
        let duration = start.elapsed();
        println!("PoMap get {num_items} items:    {duration:?}");

        let start = Instant::now();
        // for (k, v) in &items {
        //     assert_eq!(pomap.remove(k), Some(*v));
        // }
        let duration = start.elapsed();
        println!("PoMap remove {num_items} items: {duration:?}");

        // --- std::collections::HashMap Benchmark ---
        println!("\n--- Benchmarking std::collections::HashMap ---");
        let mut hashmap = HashMap::with_capacity(0);
        let start = Instant::now();
        for (k, v) in &items {
            hashmap.insert(*k, *v);
        }
        let duration = start.elapsed();
        println!("HashMap insert {num_items} items: {duration:?}");

        let start = Instant::now();
        for (k, v) in &items {
            assert_eq!(hashmap.get(k), Some(v));
        }
        let duration = start.elapsed();
        println!("HashMap get {num_items} items:    {duration:?}");

        let start = Instant::now();
        for (k, v) in &items {
            assert_eq!(hashmap.remove(k), Some(*v));
        }
        let duration = start.elapsed();
        println!("HashMap remove {num_items} items: {duration:?}");
    }
}
