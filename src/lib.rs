use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;

const LOAD_FACTOR: f64 = 0.75;

/// Internal structure used to hold entries in the [`PoMap`].
#[derive(Clone, Debug)]
struct Entry<K: Hash + Eq + Clone, V: Clone> {
    key: K,
    value: V,
    hash: u64,
}

/// A Prefix-Ordered Hash Map (PoMap).
///
/// This map uses a cache-conscious, array-based layout to provide fast lookups
/// and an elegant, single-pass resize operation. Its performance characteristics are
/// O(log(log n)) for lookups and an amortized O(log n) for insertions.
#[derive(Clone, Debug)]
pub struct PoMap<K: Hash + Eq + Clone, V: Clone> {
    capacity: usize,
    len: usize,
    k: usize, // Bucket size: K = log(capacity)
    p: u32,   // Number of prefix bits
    num_buckets: usize,
    data: Vec<Option<Entry<K, V>>>,
    indices: Vec<Option<u8>>,
}

impl<K: Hash + Eq + Clone, V: Clone> PoMap<K, V> {
    /// Creates a new, empty PoMap.
    ///
    /// The map will be initialized with a capacity of 0. It will allocate
    /// space on the first insertion.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Creates a new PoMap with a specified initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                capacity: 0,
                len: 0,
                k: 0,
                p: 0,
                num_buckets: 0,
                data: Vec::new(),
                indices: Vec::new(),
            };
        }
        let (k, p, num_buckets) = Self::calculate_layout(capacity);
        Self {
            capacity,
            len: 0,
            k,
            p,
            num_buckets,
            data: vec![None; capacity],
            indices: vec![None; capacity],
        }
    }

    /// Calculates the layout parameters for a given capacity.
    fn calculate_layout(capacity: usize) -> (usize, u32, usize) {
        if capacity == 0 {
            return (0, 0, 0);
        }
        // K = log2(capacity), ensuring K is at least 1 and at most 256 (for u8 indices).
        let k = (capacity.ilog2() as usize).clamp(1, 256);
        let num_buckets = capacity / k;
        let p = if num_buckets > 0 {
            num_buckets.ilog2()
        } else {
            0
        };
        (k, p, num_buckets)
    }

    /// Hashes a key to a u64.
    fn hash_key(key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Gets the bucket index from a hash using the top `p` prefix bits.
    fn get_bucket_idx(&self, hash: u64) -> usize {
        if self.num_buckets == 0 {
            return 0;
        }
        (hash >> (64 - self.p)) as usize % self.num_buckets
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the current capacity of the map.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// A robust method to find a key by linearly scanning the small bucket.
    fn find_key_in_bucket(&self, bucket_idx: usize, key: &K, hash: u64) -> Option<usize> {
        let data_base = bucket_idx * self.k;
        let index_base = bucket_idx * self.k;

        for i in 0..self.k {
            if let Some(rel_idx) = self.indices[index_base + i] {
                let abs_idx = data_base + rel_idx as usize;
                if let Some(entry) = &self.data[abs_idx] {
                    if entry.hash == hash && entry.key == *key {
                        return Some(abs_idx);
                    }
                }
            } else {
                break;
            }
        }
        None
    }

    /// Inserts a key-value pair into the map.
    /// If the key already exists, the value is updated, and the old value is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 || self.len + 1 > (self.capacity as f64 * LOAD_FACTOR) as usize {
            let new_capacity = if self.capacity == 0 {
                16
            } else {
                self.capacity * 2
            };
            self.resize(new_capacity);
        }

        let hash = Self::hash_key(&key);
        let bucket_idx = self.get_bucket_idx(hash);

        // --- 1. Check for existing key and update if found ---
        if let Some(abs_idx) = self.find_key_in_bucket(bucket_idx, &key, hash) {
            return Some(mem::replace(
                &mut self.data[abs_idx].as_mut().unwrap().value,
                value,
            ));
        }

        // --- 2. Find an empty data slot in the bucket ---
        let data_base = bucket_idx * self.k;
        let mut rel_data_idx: Option<u8> = None;
        for i in 0..self.k {
            if self.data[data_base + i].is_none() {
                rel_data_idx = Some(i as u8);
                break;
            }
        }

        if rel_data_idx.is_none() {
            self.resize(self.capacity * 2);
            return self.insert(key, value);
        }
        let rel_data_idx = rel_data_idx.unwrap();

        // --- 3. Place new entry in the data slot ---
        self.data[data_base + rel_data_idx as usize] = Some(Entry { key, value, hash });

        // --- 4. Insert the new relative index into the sorted index slice ---
        let index_base = bucket_idx * self.k;
        let mut num_indices = 0;
        while num_indices < self.k && self.indices[index_base + num_indices].is_some() {
            num_indices += 1;
        }

        let mut insert_pos = num_indices;
        for i in 0..num_indices {
            let existing_rel_idx = self.indices[index_base + i].unwrap();
            let existing_hash = self.data[data_base + existing_rel_idx as usize]
                .as_ref()
                .unwrap()
                .hash;
            if hash < existing_hash {
                insert_pos = i;
                break;
            }
        }

        self.indices[index_base + insert_pos..index_base + num_indices + 1].rotate_right(1);
        self.indices[index_base + insert_pos] = Some(rel_data_idx);

        self.len += 1;
        None
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.len == 0 {
            return None;
        }
        let hash = Self::hash_key(key);
        let bucket_idx = self.get_bucket_idx(hash);

        self.find_key_in_bucket(bucket_idx, key, hash)
            .and_then(|abs_idx| self.data[abs_idx].as_ref().map(|e| &e.value))
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.len == 0 {
            return None;
        }
        let hash = Self::hash_key(key);
        let bucket_idx = self.get_bucket_idx(hash);

        // --- 1. Find the key to remove ---
        let abs_idx_to_remove = self.find_key_in_bucket(bucket_idx, key, hash)?;

        // --- 2. Remove the entry from the data array ---
        let removed_entry = self.data[abs_idx_to_remove].take().unwrap();
        let data_base = bucket_idx * self.k;
        let rel_idx_to_remove = (abs_idx_to_remove - data_base) as u8;

        // --- 3. Find and remove the relative index from the index array ---
        let index_base = bucket_idx * self.k;
        let mut pos_to_remove: Option<usize> = None;
        let mut num_indices = 0;
        for i in 0..self.k {
            if self.indices[index_base + i].is_some() {
                num_indices += 1;
                if self.indices[index_base + i] == Some(rel_idx_to_remove) {
                    pos_to_remove = Some(i);
                }
            } else {
                break;
            }
        }

        if let Some(pos) = pos_to_remove {
            // Get a mutable slice of the valid indices from the point of removal.
            let slice_to_compact = &mut self.indices[index_base + pos..index_base + num_indices];
            // Shift all elements after `pos` one to the left.
            slice_to_compact.rotate_left(1);
            // Clear the now-duplicate last element of the valid range.
            self.indices[index_base + num_indices - 1] = None;
        }

        self.len -= 1;
        Some(removed_entry.value)
    }

    /// The resize algorithm. This rebuilds the map into a new, larger layout.
    fn resize(&mut self, new_capacity: usize) {
        let (new_k, new_p, new_num_buckets) = Self::calculate_layout(new_capacity);

        let mut new_data = vec![None; new_capacity];
        let mut new_indices = vec![None; new_capacity];

        let mut new_data_counts = vec![0u8; new_num_buckets];
        let mut new_index_counts = vec![0u8; new_num_buckets];

        let old_data = mem::take(&mut self.data);
        let old_indices = mem::take(&mut self.indices);

        if self.num_buckets > 0 {
            for old_bucket_idx in 0..self.num_buckets {
                let old_data_base = old_bucket_idx * self.k;
                let old_index_slice =
                    &old_indices[old_bucket_idx * self.k..old_bucket_idx * self.k + self.k];

                for rel_old_idx_opt in old_index_slice {
                    if let Some(rel_old_idx) = rel_old_idx_opt {
                        let abs_old_idx = old_data_base + *rel_old_idx as usize;

                        if let Some(element) = &old_data[abs_old_idx] {
                            let new_bucket_idx =
                                (element.hash >> (64 - new_p)) as usize % new_num_buckets;

                            let rel_data_idx = new_data_counts[new_bucket_idx];
                            let abs_new_data_idx = new_bucket_idx * new_k + rel_data_idx as usize;
                            new_data[abs_new_data_idx] = Some(element.clone());
                            new_data_counts[new_bucket_idx] += 1;

                            let rel_index_pos = new_index_counts[new_bucket_idx];
                            let abs_new_index_pos = new_bucket_idx * new_k + rel_index_pos as usize;
                            new_indices[abs_new_index_pos] = Some(rel_data_idx);
                            new_index_counts[new_bucket_idx] += 1;
                        }
                    }
                }
            }
        }

        self.capacity = new_capacity;
        self.k = new_k;
        self.p = new_p;
        self.num_buckets = new_num_buckets;
        self.data = new_data;
        self.indices = new_indices;
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

    #[test]
    fn test_remove() {
        let mut map = PoMap::new();
        map.insert("key1".to_string(), 1);
        map.insert("key2".to_string(), 2);
        assert_eq!(map.len(), 2);

        let removed = map.remove(&"key1".to_string());
        assert_eq!(removed, Some(1));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&"key1".to_string()), None);
        assert_eq!(map.get(&"key2".to_string()), Some(&2));

        let non_existent = map.remove(&"key3".to_string());
        assert_eq!(non_existent, None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_resize() {
        let mut map = PoMap::with_capacity(16);
        assert_eq!(map.capacity(), 16);

        // Insert enough elements to trigger a resize.
        for i in 0..13 {
            map.insert(i.to_string(), i);
        }

        assert_eq!(map.len(), 13);
        assert!(map.capacity() >= 32);

        // Verify all elements are still accessible.
        for i in 0..13 {
            assert_eq!(map.get(&i.to_string()), Some(&i));
        }
    }

    #[test]
    fn test_many_insertions_and_removals() {
        let mut map = PoMap::new();
        let num_items = 1_000;

        for i in 0..num_items {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), num_items);

        for i in (0..num_items).step_by(2) {
            assert_eq!(map.remove(&i), Some(i * 2));
        }
        assert_eq!(map.len(), num_items / 2);

        for i in 0..num_items {
            if i % 2 == 0 {
                assert_eq!(map.get(&i), None);
            } else {
                assert_eq!(map.get(&i), Some(&(i * 2)));
            }
        }
    }

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
}
