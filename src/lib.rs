// use std::alloc::{self, Layout};
// use std::collections::hash_map::DefaultHasher;
// use std::collections::hash_set;
use std::hash::{Hash, Hasher};
// use std::ptr;
// use wide::u64x4;

mod simd;

#[derive(Hash, Clone, Debug, Default)]
enum Slot<K: Hash + Eq + Clone, V: Clone> {
    Bucket {
        p_bits: u8,
        len: u8,
    },
    Entry {
        hash: u64,
        key: K,
        value: V,
    },
    #[default]
    Empty,
}

pub struct PoMap<K: Hash + Eq + Clone, V: Clone> {
    p_bits: u8, // Number of prefix bits to go from global -> bucket
    len: usize,
    entries: Vec<Slot<K, V>>,
}

impl<K: Hash + Eq + Clone, V: Clone> PoMap<K, V> {
    pub fn new() -> Self {
        Self {
            p_bits: 0,
            len: 0,
            entries: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.p_bits = 0;
        self.len = 0;
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            p_bits: 0,
            len: 0,
            entries: Vec::with_capacity(capacity),
        }
    }
}

// #[derive(Clone, Debug, PartialEq, Eq, Default)]
// struct Bucket<K: Hash + Eq + Clone, V: Clone> {
//     hashes: u64x4,
//     entries: [Entry<K, V>; WIDE_LEN],
// }

// #[inline(always)]
// fn find_match_index(vec: u64x4, target: u64) -> Option<usize> {
//     // lane = 0xFFFF_FFFF_FFFF_FFFF if equal, else 0
//     let cmp = vec.cmp_eq(u64x4::splat(target));

//     // pull the top bit of each lane (now 0 or 1)
//     let hi_bits: u64x4 = cmp >> 63; // shift by *scalar* 63, OK

//     // convert to array so we can collapse to one byte
//     let a = hi_bits.to_array(); // [u64; 4], each element 0 or 1

//     // pack into a 4‑bit mask: bit i == 1 if lane i matched
//     let mask: u8 = (a[0] as u8) | ((a[1] as u8) << 1) | ((a[2] as u8) << 2) | ((a[3] as u8) << 3);

//     if mask == 0 {
//         None
//     } else {
//         Some(mask.trailing_zeros() as usize)
//     }
// }

// /// A Prefix-Ordered Hash Map (PoMap).
// ///
// /// This map uses a cache-conscious, array-based layout to provide fast lookups
// /// and an elegant, single-pass resize operation. It is implemented using manually
// /// allocated memory, giving it the same performance characteristics as `Vec`
// /// without requiring `K: Default` or `V: Default` trait bounds.
// ///
// /// The core idea is to partition the key space by the top `p` bits of their hash.
// /// These partitions are called buckets. Each bucket has a capacity of `k`, where
// /// `k` is `log(capacity)`. Within each bucket, entries are kept sorted by their
// /// full hash value, allowing for binary search during lookups and insertions.
// #[derive(Clone, Debug, PartialEq, Eq, Default)]
// pub struct PoMap<K: Hash + Eq + Clone, V: Clone> {
//     p: u8, // Number of prefix bits
//     len: usize,
//     buckets: Vec<Bucket<K, V>>,
// }

// impl<K: Hash + Eq + Clone, V: Clone> PoMap<K, V> {
//     /// Creates an empty `PoMap`.
//     ///
//     /// The map will not allocate until the first element is inserted.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map: PoMap<&str, i32> = PoMap::new();
//     /// ```
//     pub fn new() -> Self {
//         Self {
//             p: 0,
//             len: 0,
//             buckets: Vec::new(),
//         }
//     }

//     /// Clears the map, removing all key-value pairs while retaining the allocated capacity.
//     ///
//     /// # Time Complexity
//     ///
//     /// `O(N)` where N is the number of elements, because it must drop each element.
//     /// If the key and value types do not need to be dropped (e.g., are `Copy`),
//     /// the complexity is `O(1)` because it can zero out the index memory directly.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// map.insert("a", 1);
//     /// map.clear();
//     /// assert!(map.is_empty());
//     /// ```
//     pub fn clear(&mut self) {
//         self.buckets.clear();
//         self.p = 0;
//         self.len = 0;
//     }

//     /// Creates an empty `PoMap` with at least the specified capacity.
//     ///
//     /// The map will be able to hold at least `capacity` elements without
//     /// reallocating. If `capacity` is 0, the map will not allocate.
//     ///
//     /// The actual capacity will be the next power of two.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map: PoMap<&str, i32> = PoMap::with_capacity(10);
//     /// assert!(map.capacity() >= 10);
//     /// ```
//     pub fn with_capacity(capacity: usize) -> Self {
//         Self {
//             p: 0,
//             len: 0,
//             buckets: Vec::with_capacity(capacity),
//         }
//     }

//     #[inline(always)]
//     fn hash_key(key: &K) -> u64 {
//         let mut hasher = DefaultHasher::new();
//         key.hash(&mut hasher);
//         hasher.finish()
//     }

//     #[inline(always)]
//     fn get_bucket_idx(&self, hash: u64) -> usize {
//         let num_buckets = self.buckets.len();
//         if num_buckets == 0 {
//             return 0;
//         }
//         (hash >> (64 - self.p)) as usize % num_buckets
//     }

//     #[inline(always)]
//     fn find_key_in_bucket(&self, bucket: &Bucket<K, V>, key: &K, hash: u64) -> Option<usize> {
//         let Some(index) = simd::find_match_index(bucket.hashes, hash) else {
//             return None;
//         };
//         let entry = &bucket.entries[index];
//         if &entry.key == key { Some(index) } else { None }
//     }

//     /// Returns the number of elements in the map.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// assert_eq!(map.len(), 0);
//     /// map.insert("a", 1);
//     /// assert_eq!(map.len(), 1);
//     /// ```
//     pub fn len(&self) -> usize {
//         self.len
//     }

//     /// Returns `true` if the map contains no elements.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// assert!(map.is_empty());
//     /// map.insert("a", 1);
//     /// assert!(!map.is_empty());
//     /// ```
//     pub fn is_empty(&self) -> bool {
//         self.len == 0
//     }

//     /// Returns the number of elements the map can hold without reallocating.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let map: PoMap<i32, i32> = PoMap::with_capacity(10);
//     /// assert!(map.capacity() >= 10);
//     /// ```
//     pub fn capacity(&self) -> usize {
//         self.buckets.capacity() * WIDE_LEN
//     }

//     /// Inserts a key-value pair into the map.
//     ///
//     /// If the map did not have this key present, `None` is returned.
//     ///
//     /// If the map did have this key present, the value is updated, and the old
//     /// value is returned. The key is not updated, though; this matters for
//     /// types that can be `==` without being identical.
//     ///
//     /// # Time Complexity
//     ///
//     /// The average time complexity is `O(log N)` where N is the number of elements in the map.
//     /// This is because finding the insertion point involves a binary search within a bucket
//     /// of size `log(N)`, followed by shifting elements within that bucket.
//     ///
//     /// The worst-case complexity is also `O(log N)`, unless there are many hash collisions,
//     /// in which case the linear scan for the correct key can dominate.
//     ///
//     /// When insertion triggers a resize, the complexity is `O(N)`, but this is amortized
//     /// to `O(log N)` over many insertions.
//     ///
//     /// It is worth noting that the resize operation is particularly efficient and runs in a
//     /// single linear scan.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// assert_eq!(map.insert("a", 1), None);
//     /// assert!(!map.is_empty());
//     ///
//     /// map.insert("a", 2);
//     /// assert_eq!(map.insert("a", 3), Some(2));
//     /// assert_eq!(map.get(&"a"), Some(&3));
//     /// ```
//     pub fn insert(&mut self, key: K, value: V) -> Option<V> {
//         // --- Resize Logic ---
//         // Stage 1: Check if we need to resize due to the load factor.
//         if self.capacity == 0 || self.len + 1 > (self.capacity as f64 * LOAD_FACTOR) as usize {
//             let new_capacity = if self.capacity == 0 {
//                 16
//             } else {
//                 self.capacity * 2
//             };
//             self.resize(new_capacity);
//         }

//         // Stage 2: Check if the target bucket is full. This requires re-calculating
//         // the hash and bucket index in case the first resize happened.
//         let hash = Self::hash_key(&key);
//         let mut bucket_idx = self.get_bucket_idx(hash);

//         let mut index_base = bucket_idx * (self.k + 1);
//         let mut len = unsafe { *self.indices.add(index_base) } as usize;

//         if len >= self.k {
//             self.resize(self.capacity * 2);
//             // After this resize, we MUST re-calculate all layout-dependent variables.
//             bucket_idx = self.get_bucket_idx(hash);
//             index_base = bucket_idx * (self.k + 1);
//         }
//         // --- End Resize Logic ---

//         // Now that we've resized (if necessary), we can safely proceed.
//         if let Some((abs_idx, _)) = self.find_key_in_bucket(bucket_idx, &key, hash) {
//             unsafe {
//                 let entry_ptr = self.data.add(abs_idx);
//                 // Safety: entry_ptr is valid and points to an existing element.
//                 // ptr::replace swaps the value and returns the old one, which is what we want.
//                 let old_value = ptr::replace(&mut (*entry_ptr).value, value);
//                 return Some(old_value);
//             }
//         }

//         // Re-fetch len as it might have changed if we resized due to bucket overflow.
//         len = unsafe { *self.indices.add(index_base) } as usize;
//         let data_base = bucket_idx * self.k;
//         let rel_data_idx = len as u8;
//         unsafe {
//             self.data
//                 .add(data_base + rel_data_idx as usize)
//                 .write(Entry { key, value, hash });
//         }

//         let index_slice_start = unsafe { self.indices.add(index_base + 1) };
//         let insert_pos = unsafe {
//             let slice = std::slice::from_raw_parts(index_slice_start, len);
//             match slice.binary_search_by_key(&hash, |rel_idx| {
//                 (*self.data.add(data_base + *rel_idx as usize)).hash
//             }) {
//                 Ok(pos) => pos,
//                 Err(pos) => pos,
//             }
//         };

//         let insert_offset = index_base + 1 + insert_pos;
//         unsafe {
//             let slice_ptr = self.indices.add(insert_offset);
//             ptr::copy(slice_ptr, slice_ptr.add(1), len - insert_pos);
//             *slice_ptr = rel_data_idx;
//         }
//         unsafe {
//             *self.indices.add(index_base) += 1; // Increment length
//         }
//         self.len += 1;
//         None
//     }

//     /// Returns a reference to the value corresponding to the key.
//     ///
//     /// # Time Complexity
//     ///
//     /// The average time complexity is `O(log(log N))` where N is the number of elements.
//     /// This is because lookup is a binary search within a bucket of size `log(N)`.
//     ///
//     /// The worst-case complexity due to hash collisions is `O(log N)`, as it may require
//     /// scanning the entire bucket.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// map.insert("a", 1);
//     /// assert_eq!(map.get(&"a"), Some(&1));
//     /// assert_eq!(map.get(&"b"), None);
//     /// ```
//     pub fn get(&self, key: &K) -> Option<&V> {
//         if self.is_empty() {
//             return None;
//         }
//         let hash = Self::hash_key(key);
//         let bucket_idx = self.get_bucket_idx(hash);

//         self.find_key_in_bucket(bucket_idx, key, hash)
//             .map(|(abs_idx, _)| unsafe { &(*self.data.add(abs_idx)).value })
//     }

//     /// Removes a key from the map, returning the value at the key if the key
//     /// was previously in the map.
//     ///
//     /// # Time Complexity
//     ///
//     /// The average and worst-case time complexity is `O(log N)` where N is the number of elements.
//     /// This is because `remove` must find the element (a `O(log(log N))` binary search) and then
//     /// compact the data and index arrays by shifting elements, which takes `O(log N)` time.
//     ///
//     /// # Examples
//     ///
//     /// ```
//     /// use pomap::PoMap;
//     /// let mut map = PoMap::new();
//     /// map.insert("a", 1);
//     /// assert_eq!(map.remove(&"a"), Some(1));
//     /// assert_eq!(map.remove(&"a"), None);
//     /// ```
//     pub fn remove(&mut self, key: &K) -> Option<V> {
//         if self.is_empty() {
//             return None;
//         }
//         let hash = Self::hash_key(key);
//         let bucket_idx = self.get_bucket_idx(hash);

//         let (abs_idx_to_remove, index_pos_to_remove) =
//             self.find_key_in_bucket(bucket_idx, key, hash)?;

//         // Read the entire entry out. This leaves the slot logically uninitialized.
//         // The key will be dropped at the end of this function, and we return the value.
//         let entry_to_remove = unsafe { ptr::read(self.data.add(abs_idx_to_remove)) };
//         let value = entry_to_remove.value;

//         let data_base = bucket_idx * self.k;
//         let index_base = bucket_idx * (self.k + 1);
//         let len = unsafe { *self.indices.add(index_base) } as usize;
//         let rel_idx_to_remove = (abs_idx_to_remove - data_base) as u8;

//         // Shift data elements to the left to fill the gap.
//         for i in abs_idx_to_remove..(data_base + len - 1) {
//             unsafe {
//                 let src = self.data.add(i + 1);
//                 let dst = self.data.add(i);
//                 ptr::copy_nonoverlapping(src, dst, 1);
//             }
//         }

//         // Update indices that pointed to shifted elements.
//         // Any index greater than the one we removed needs to be decremented.
//         let index_slice_start = unsafe { self.indices.add(index_base + 1) };
//         for i in 0..len {
//             let rel_idx_ptr = unsafe { index_slice_start.add(i) };
//             let rel_idx = unsafe { *rel_idx_ptr };
//             if rel_idx > rel_idx_to_remove {
//                 unsafe {
//                     *rel_idx_ptr -= 1;
//                 }
//             }
//         }

//         // Now, remove the index from the sorted index list
//         let index_slice_start_for_remove = unsafe { index_slice_start.add(index_pos_to_remove) };
//         unsafe {
//             ptr::copy(
//                 index_slice_start_for_remove.add(1),
//                 index_slice_start_for_remove,
//                 len - index_pos_to_remove - 1,
//             );
//         }

//         unsafe {
//             *self.indices.add(index_base) -= 1; // Decrement length
//         }
//         self.len -= 1;
//         Some(value)
//     }

//     fn resize(&mut self, new_capacity: usize) {
//         let (new_k, new_p, new_num_buckets) = Self::calculate_layout(new_capacity);

//         let new_data_layout = Layout::array::<Entry<K, V>>(new_capacity).unwrap();
//         let new_data = unsafe { alloc::alloc(new_data_layout) as *mut Entry<K, V> };

//         let new_indices_layout = Layout::array::<u8>(new_num_buckets * (new_k + 1)).unwrap();
//         let new_indices = unsafe { alloc::alloc_zeroed(new_indices_layout) };

//         if new_data.is_null() || new_indices.is_null() {
//             panic!("Failed to allocate memory for PoMap resize");
//         }

//         let old_num_buckets = self.num_buckets;
//         let old_data = self.data;
//         let old_indices = self.indices;
//         let old_capacity = self.capacity;
//         let old_k = self.k;

//         self.capacity = new_capacity;
//         self.k = new_k;
//         self.p = new_p;
//         self.num_buckets = new_num_buckets;
//         self.data = new_data;
//         self.indices = new_indices;

//         if old_capacity > 0 {
//             for old_bucket_idx in 0..old_num_buckets {
//                 let old_data_base = old_bucket_idx * old_k;
//                 let old_index_base = old_bucket_idx * (old_k + 1);
//                 let old_len = unsafe { *old_indices.add(old_index_base) } as usize;

//                 let old_index_slice_start = unsafe { old_indices.add(old_index_base + 1) };

//                 for i in 0..old_len {
//                     let rel_old_idx = unsafe { *old_index_slice_start.add(i) } as usize;
//                     let abs_old_idx = old_data_base + rel_old_idx;
//                     let element = unsafe { ptr::read(old_data.add(abs_old_idx)) };
//                     let new_bucket_idx = self.get_bucket_idx(element.hash);

//                     let new_data_base = new_bucket_idx * self.k;
//                     let new_index_base = new_bucket_idx * (self.k + 1);
//                     let len = unsafe { *self.indices.add(new_index_base) } as usize;

//                     // Relative index is just current length (defragment buckets)
//                     let rel_data_idx = len as u8;
//                     unsafe {
//                         self.data
//                             .add(new_data_base + rel_data_idx as usize)
//                             .write(element);
//                         // append index directly since we are processing in hash order
//                         *self.indices.add(new_index_base + 1 + len) = rel_data_idx;
//                         *self.indices.add(new_index_base) += 1;
//                     }
//                 }
//             }

//             // Deallocate old memory
//             let old_data_layout = Layout::array::<Entry<K, V>>(old_capacity).unwrap();
//             unsafe { alloc::dealloc(old_data as *mut u8, old_data_layout) };

//             let old_indices_layout = Layout::array::<u8>(old_num_buckets * (old_k + 1)).unwrap();
//             unsafe { alloc::dealloc(old_indices, old_indices_layout) };
//         }
//     }
// }

// // --- Unit Tests ---
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_new_and_insert() {
//         let mut map = PoMap::new();
//         assert_eq!(map.len(), 0);
//         map.insert("key1".to_string(), 42);
//         assert_eq!(map.len(), 1);
//         assert_eq!(map.get(&"key1".to_string()), Some(&42));
//     }

//     #[test]
//     fn test_clear() {
//         let mut map = PoMap::new();
//         map.insert("key1".to_string(), 1);
//         map.insert("key2".to_string(), 2);
//         let capacity_before = map.capacity();
//         map.clear();
//         assert_eq!(map.len(), 0);
//         assert_eq!(map.capacity(), capacity_before); // Should not deallocate
//         assert_eq!(map.get(&"key1".to_string()), None);
//         map.insert("key3".to_string(), 3);
//         assert_eq!(map.len(), 1);
//         assert_eq!(map.get(&"key3".to_string()), Some(&3));
//     }

//     #[test]
//     fn test_default() {
//         let map: PoMap<String, i32> = PoMap::default();
//         assert_eq!(map.len(), 0);
//         assert_eq!(map.capacity(), 0);
//     }

//     #[test]
//     fn test_get_nonexistent() {
//         let mut map = PoMap::new();
//         map.insert("key1".to_string(), 42);
//         assert_eq!(map.get(&"key2".to_string()), None);
//     }

//     #[test]
//     fn test_update() {
//         let mut map = PoMap::new();
//         map.insert("key1".to_string(), 42);
//         let old_value = map.insert("key1".to_string(), 99);
//         assert_eq!(map.len(), 1);
//         assert_eq!(old_value, Some(42));
//         assert_eq!(map.get(&"key1".to_string()), Some(&99));
//     }

//     #[test]
//     fn test_remove() {
//         let mut map = PoMap::new();
//         map.insert("key1".to_string(), 1);
//         map.insert("key2".to_string(), 2);
//         assert_eq!(map.len(), 2);

//         let removed = map.remove(&"key1".to_string());
//         assert_eq!(removed, Some(1));
//         assert_eq!(map.len(), 1);
//         assert_eq!(map.get(&"key1".to_string()), None);
//         assert_eq!(map.get(&"key2".to_string()), Some(&2));

//         let non_existent = map.remove(&"key3".to_string());
//         assert_eq!(non_existent, None);
//         assert_eq!(map.len(), 1);
//     }

//     #[test]
//     fn test_resize_and_rehash() {
//         let mut map = PoMap::with_capacity(16);
//         assert_eq!(map.capacity(), 16);
//         assert_eq!(map.k, 4); // k = 16.ilog2()

//         // Insert enough items to trigger a resize.
//         // Load factor is 0.75, so 16 * 0.75 = 12. The 13th element will resize.
//         for i in 0..12 {
//             map.insert(i.to_string(), i);
//         }
//         assert_eq!(map.len(), 12);
//         assert_eq!(map.capacity(), 64);

//         map.insert("12".to_string(), 12);
//         assert_eq!(map.len(), 13);
//         assert_eq!(map.capacity(), 64);
//         assert_eq!(map.k, 6);

//         // Verify all old and new elements are present.
//         for i in 0..13 {
//             assert_eq!(map.get(&i.to_string()), Some(&i));
//         }
//     }

//     #[test]
//     fn test_many_insertions_and_removals() {
//         let mut map = PoMap::new();
//         let num_items = 1_000;

//         for i in 0..num_items {
//             map.insert(i, i * 2);
//         }
//         assert_eq!(map.len(), num_items);

//         for i in (0..num_items).step_by(2) {
//             assert_eq!(map.remove(&i), Some(i * 2));
//         }
//         assert_eq!(map.len(), num_items / 2);

//         for i in 0..num_items {
//             if i % 2 == 0 {
//                 assert_eq!(map.get(&i), None);
//             } else {
//                 assert_eq!(map.get(&i), Some(&(i * 2)));
//             }
//         }
//     }

//     #[derive(Debug, Clone, Eq)]
//     struct CollidingKey {
//         val: u64,
//         hash: u64,
//     }

//     impl Hash for CollidingKey {
//         fn hash<H: Hasher>(&self, state: &mut H) {
//             self.hash.hash(state);
//         }
//     }

//     impl PartialEq for CollidingKey {
//         fn eq(&self, other: &Self) -> bool {
//             self.val == other.val
//         }
//     }

//     #[test]
//     fn test_hash_collisions() {
//         let mut map = PoMap::new();
//         let key1 = CollidingKey { val: 1, hash: 123 };
//         let key2 = CollidingKey { val: 2, hash: 123 };
//         let key3 = CollidingKey { val: 3, hash: 123 };

//         map.insert(key1.clone(), 10);
//         map.insert(key2.clone(), 20);
//         map.insert(key3.clone(), 30);

//         assert_eq!(map.len(), 3);
//         assert_eq!(map.get(&key1), Some(&10));
//         assert_eq!(map.get(&key2), Some(&20));
//         assert_eq!(map.get(&key3), Some(&30));

//         // Test update on collision
//         map.insert(key2.clone(), 25);
//         assert_eq!(map.len(), 3);
//         assert_eq!(map.get(&key2), Some(&25));

//         // Test remove on collision
//         assert_eq!(map.remove(&key1), Some(10));
//         assert_eq!(map.len(), 2);
//         assert_eq!(map.get(&key1), None);
//         assert_eq!(map.get(&key2), Some(&25));
//         assert_eq!(map.get(&key3), Some(&30));
//     }

//     #[test]
//     fn test_bucket_overflow_and_resize() {
//         let mut map = PoMap::with_capacity(16);

//         // Create keys that all map to the same bucket.
//         // We can achieve this by crafting hashes. A key's bucket is `(hash >> (64 - p)) % num_buckets`.
//         // With C=16, k=4, num_buckets=4, p=2.
//         // We need `(hash >> 62) % 4` to be constant.
//         let collision_hash_base: u64 = 1 << 62;

//         for i in 0..4 {
//             // These should all go to the same bucket.
//             let key = CollidingKey {
//                 val: i,
//                 hash: collision_hash_base + i,
//             };
//             map.insert(key, i as i32);
//         }
//         assert_eq!(map.len(), 4);
//         assert_eq!(map.capacity(), 16);

//         // This should trigger a resize because the target bucket is full (len=k=4).
//         let key = CollidingKey {
//             val: 4,
//             hash: collision_hash_base + 4,
//         };
//         map.insert(key, 4);
//         assert_eq!(map.len(), 5);
//         assert_eq!(map.capacity(), 16);

//         // Check that all values are still there.
//         for i in 0..5 {
//             let key_to_check = CollidingKey {
//                 val: i,
//                 hash: collision_hash_base + i,
//             };
//             assert_eq!(map.get(&key_to_check), Some(&(i as i32)));
//         }
//     }

//     #[test]
//     fn test_random_insert_remove_comprehensive() {
//         use std::collections::HashMap;

//         let mut pomap = PoMap::new();
//         let mut hashmap = HashMap::new();
//         let num_items = 2_000;

//         let mut items: Vec<(i32, i32)> = (0..num_items).map(|i| (i, i * 10)).collect();

//         // Simple shuffle
//         for i in 0..items.len() {
//             let j = (i * 7) % items.len();
//             items.swap(i, j);
//         }

//         // Insert all items
//         for (k, v) in &items {
//             pomap.insert(*k, *v);
//             hashmap.insert(*k, *v);
//             assert_eq!(pomap.len(), hashmap.len());
//         }

//         assert_eq!(pomap.len(), num_items as usize);
//         for (k, v) in &items {
//             assert_eq!(pomap.get(k), Some(v));
//         }

//         // Remove about half the items
//         let mut removed_count = 0;
//         for (i, (k, v)) in items.iter().enumerate() {
//             if i % 2 == 0 {
//                 assert_eq!(pomap.remove(k), Some(*v));
//                 hashmap.remove(k);
//                 removed_count += 1;
//             }
//             assert_eq!(pomap.len(), hashmap.len());
//         }

//         assert_eq!(pomap.len(), (num_items - removed_count) as usize);

//         // Check final state
//         for (k, _) in &items {
//             assert_eq!(pomap.get(k), hashmap.get(k));
//         }
//     }

//     #[test]
//     fn test_zero_capacity() {
//         let mut map: PoMap<i32, i32> = PoMap::with_capacity(0);
//         assert_eq!(map.len(), 0);
//         assert_eq!(map.capacity(), 0);

//         map.insert(1, 1);
//         assert_eq!(map.len(), 1);
//         assert_eq!(map.capacity(), 16);
//         assert_eq!(map.get(&1), Some(&1));
//     }

//     #[test]
//     fn test_clone() {
//         let mut map = PoMap::new();
//         map.insert("a", 1);
//         map.insert("b", 2);

//         let mut clone = map.clone();
//         assert_eq!(clone.len(), 2);
//         assert_eq!(clone.get(&"a"), Some(&1));
//         assert_eq!(clone.get(&"b"), Some(&2));

//         // Modify the clone and check that the original is unaffected
//         clone.insert("c", 3);
//         clone.remove(&"a");
//         assert_eq!(map.len(), 2);
//         assert_eq!(map.get(&"a"), Some(&1));
//         assert_eq!(map.get(&"c"), None);

//         // Modify the original and check that the clone is unaffected
//         map.insert("d", 4);
//         assert_eq!(clone.len(), 2);
//         assert_eq!(clone.get(&"d"), None);
//     }

//     #[test]
//     fn test_zero_sized_types() {
//         // Test with ZST value
//         let mut map_zst_val = PoMap::new();
//         map_zst_val.insert(1, ());
//         map_zst_val.insert(2, ());
//         assert_eq!(map_zst_val.len(), 2);
//         assert_eq!(map_zst_val.get(&1), Some(&()));
//         assert_eq!(map_zst_val.remove(&1), Some(()));
//         assert_eq!(map_zst_val.len(), 1);
//         assert_eq!(map_zst_val.get(&2), Some(&()));

//         // Test with ZST key
//         // Since all keys are equal, it can only hold one item.
//         let mut map_zst_key: PoMap<(), i32> = PoMap::new();
//         assert_eq!(map_zst_key.insert((), 100), None);
//         assert_eq!(map_zst_key.len(), 1);
//         assert_eq!(map_zst_key.insert((), 200), Some(100));
//         assert_eq!(map_zst_key.len(), 1);
//         assert_eq!(map_zst_key.get(&()), Some(&200));
//     }

//     use std::sync::Arc;
//     use std::sync::atomic::{AtomicUsize, Ordering};

//     #[derive(Clone, Debug)]
//     struct DropCounter {
//         id: i32,
//         counter: Arc<AtomicUsize>,
//     }

//     impl Drop for DropCounter {
//         fn drop(&mut self) {
//             self.counter.fetch_add(1, Ordering::SeqCst);
//         }
//     }

//     impl PartialEq for DropCounter {
//         fn eq(&self, other: &Self) -> bool {
//             self.id == other.id
//         }
//     }

//     #[test]
//     fn test_drop_behavior() {
//         let counter = Arc::new(AtomicUsize::new(0));

//         {
//             let mut map = PoMap::new();
//             let make_droppable = |id| DropCounter {
//                 id,
//                 counter: Arc::clone(&counter),
//             };

//             // 1. Test drop on insert (overwrite)
//             map.insert("a", make_droppable(1));
//             assert_eq!(counter.load(Ordering::SeqCst), 0);
//             map.insert("a", make_droppable(2)); // Overwrites key "a", should drop value 1
//             assert_eq!(counter.load(Ordering::SeqCst), 1);

//             // 2. Test drop on remove
//             map.insert("b", make_droppable(3));
//             assert_eq!(counter.load(Ordering::SeqCst), 1);
//             map.remove(&"b"); // Removes key "b", should drop value 3
//             assert_eq!(counter.load(Ordering::SeqCst), 2);

//             // 3. Test drop on clear
//             map.insert("c", make_droppable(4));
//             map.insert("d", make_droppable(5));
//             assert_eq!(counter.load(Ordering::SeqCst), 2);
//             map.clear(); // Should drop values 2, 4, 5
//             assert_eq!(counter.load(Ordering::SeqCst), 5);
//             assert_eq!(map.len(), 0);

//             // 4. Test drop on PoMap::drop
//             map.insert("e", make_droppable(6));
//             map.insert("f", make_droppable(7));
//         } // map goes out of scope here

//         // Should drop values 6, 7
//         assert_eq!(counter.load(Ordering::SeqCst), 7);
//     }

//     #[test]
//     fn test_performance_vs_std_hashmap() {
//         use std::collections::HashMap;
//         use std::time::Instant;

//         let num_items = 1_000_000;
//         let mut items: Vec<(i32, i32)> = (0..num_items).map(|i| (i, i * 10)).collect();

//         // Simple shuffle to make access less predictable
//         for i in 0..items.len() {
//             let j = (i * 13) % items.len();
//             items.swap(i, j);
//         }

//         // --- PoMap Benchmark ---
//         println!("\n--- Benchmarking PoMap ---");
//         let mut pomap = PoMap::with_capacity(0);
//         let start = Instant::now();
//         for (k, v) in &items {
//             pomap.insert(*k, *v);
//         }
//         let duration = start.elapsed();
//         println!("PoMap insert {num_items} items: {duration:?}");

//         let start = Instant::now();
//         for (k, v) in &items {
//             assert_eq!(pomap.get(k), Some(v));
//         }
//         let duration = start.elapsed();
//         println!("PoMap get {num_items} items:    {duration:?}");

//         let start = Instant::now();
//         for (k, v) in &items {
//             assert_eq!(pomap.remove(k), Some(*v));
//         }
//         let duration = start.elapsed();
//         println!("PoMap remove {num_items} items: {duration:?}");

//         // --- std::collections::HashMap Benchmark ---
//         println!("\n--- Benchmarking std::collections::HashMap ---");
//         let mut hashmap = HashMap::with_capacity(0);
//         let start = Instant::now();
//         for (k, v) in &items {
//             hashmap.insert(*k, *v);
//         }
//         let duration = start.elapsed();
//         println!("HashMap insert {num_items} items: {duration:?}");

//         let start = Instant::now();
//         for (k, v) in &items {
//             assert_eq!(hashmap.get(k), Some(v));
//         }
//         let duration = start.elapsed();
//         println!("HashMap get {num_items} items:    {duration:?}");

//         let start = Instant::now();
//         for (k, v) in &items {
//             assert_eq!(hashmap.remove(k), Some(*v));
//         }
//         let duration = start.elapsed();
//         println!("HashMap remove {num_items} items: {duration:?}");
//     }
// }
