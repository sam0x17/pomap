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

/// A Prefix-Ordered Hash Map (POHM) or just PoMap.
///
/// This map uses a cache-conscious, array-based layout to provide fast lookups
/// and an elegant, single-pass resize operation.
#[derive(Clone, Debug)]
pub struct PoMap<K: Hash + Eq + Clone, V: Clone> {
    capacity: usize,
    len: usize,
    k: usize, // Bucket size: K = log(capacity)
    p: u32,   // Number of prefix bits
    num_buckets: usize,
    // Data is stored in fixed-size blocks, one for each bucket.
    data: Vec<Option<Entry<K, V>>>,
    // Indices are u8 relative offsets into a data block.
    indices: Vec<Option<u8>>,
}

impl<K: Hash + Eq + Clone, V: Clone> PoMap<K, V> {
    /// Creates a new, empty PohmMap with an initial capacity.
    pub fn new() -> Self {
        Self::with_capacity(0) // Start with a default capacity
    }

    /// Creates a new PohmMap with a specified initial capacity.
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
        let p = num_buckets.ilog2();
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
        if self.p == 0 {
            return 0;
        }
        // Right shift to get the top `p` bits.
        (hash >> (64 - self.p)) as usize
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the current capacity of the map.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Inserts a key-value pair into the map.
    /// If the map did not have this key present, None is returned.
    /// If the map did have this key present, the value is updated,
    /// and the old value is returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if we need to grow the map before insertion.
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
        let data_base = bucket_idx * self.k;
        let index_base = bucket_idx * self.k;

        // --- 1. Check for existing key and update if found ---
        let index_slice = &self.indices[index_base..index_base + self.k];
        let search_result = index_slice.binary_search_by(|probe_rel_idx_opt| {
            if let Some(probe_rel_idx) = probe_rel_idx_opt {
                let probe_abs_idx = data_base + *probe_rel_idx as usize;
                self.data[probe_abs_idx].as_ref().unwrap().hash.cmp(&hash)
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if let Ok(found_idx_pos) = search_result {
            let rel_idx = self.indices[index_base + found_idx_pos].unwrap();
            let abs_idx = data_base + rel_idx as usize;
            if let Some(entry) = self.data[abs_idx].as_mut() {
                if entry.key == key {
                    return Some(mem::replace(&mut entry.value, value));
                }
            }
        }

        // --- 2. Find an empty data slot in the bucket ---
        let mut rel_data_idx: Option<u8> = None;
        for i in 0..self.k {
            if self.data[data_base + i].is_none() {
                rel_data_idx = Some(i as u8);
                break;
            }
        }

        // If no slot is found, the bucket is full. Resize and retry.
        if rel_data_idx.is_none() {
            self.resize(self.capacity * 2);
            return self.insert(key, value);
        }
        let rel_data_idx = rel_data_idx.unwrap();

        // --- 3. Place new entry in the data slot ---
        self.data[data_base + rel_data_idx as usize] = Some(Entry { key, value, hash });

        // --- 4. Insert the new relative index into the sorted index slice ---
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
        let data_base = bucket_idx * self.k;
        let index_base = bucket_idx * self.k;

        let index_slice = &self.indices[index_base..index_base + self.k];
        let search_result = index_slice.binary_search_by(|probe_rel_idx_opt| {
            if let Some(probe_rel_idx) = probe_rel_idx_opt {
                let probe_abs_idx = data_base + *probe_rel_idx as usize;
                self.data[probe_abs_idx].as_ref().unwrap().hash.cmp(&hash)
            } else {
                std::cmp::Ordering::Greater
            }
        });

        match search_result {
            Ok(found_idx_pos) => {
                let rel_idx = self.indices[index_base + found_idx_pos].unwrap();
                let abs_idx = data_base + rel_idx as usize;
                let entry = self.data[abs_idx].as_ref().unwrap();
                if &entry.key == key {
                    Some(&entry.value)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    /// The resize algorithm. This rebuilds the map into a new, larger layout.
    fn resize(&mut self, new_capacity: usize) {
        let (new_k, new_p, new_num_buckets) = Self::calculate_layout(new_capacity);

        let mut new_data = vec![None; new_capacity];
        let mut new_indices = vec![None; new_capacity];

        let mut new_data_counts = vec![0u8; new_num_buckets];
        let mut new_index_counts = vec![0u8; new_num_buckets];

        let old_data = mem::replace(&mut self.data, vec![]);
        let old_indices = mem::replace(&mut self.indices, vec![]);

        for old_bucket_idx in 0..self.num_buckets {
            let old_data_base = old_bucket_idx * self.k;
            let old_index_slice =
                &old_indices[old_bucket_idx * self.k..old_bucket_idx * self.k + self.k];

            for rel_old_idx_opt in old_index_slice {
                if let Some(rel_old_idx) = rel_old_idx_opt {
                    let abs_old_idx = old_data_base + *rel_old_idx as usize;

                    if let Some(element) = old_data[abs_old_idx].as_ref() {
                        let new_bucket_idx = (element.hash >> (64 - new_p)) as usize;

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
    fn test_new() {
        let map: PoMap<String, i32> = PoMap::new();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity, 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = PoMap::new();
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
    fn test_resize() {
        let mut map = PoMap::with_capacity(16);
        assert_eq!(map.capacity(), 16);

        // Insert enough elements to trigger a resize.
        // Load factor is 0.75, so 16 * 0.75 = 12. Inserting the 13th element should resize.
        for i in 0..13 {
            map.insert(i.to_string(), i);
        }

        assert_eq!(map.len(), 13);
        assert_eq!(map.capacity(), 64);

        // Verify all elements are still accessible.
        for i in 0..13 {
            assert_eq!(map.get(&i.to_string()), Some(&i));
        }
    }

    #[test]
    fn test_many_insertions() {
        let mut map = PoMap::new();
        let num_items = 1_000;

        for i in 0..num_items {
            map.insert(i, i * 2);
        }

        assert_eq!(map.len(), num_items);

        for i in 0..num_items {
            assert_eq!(map.get(&i), Some(&(i * 2)));
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
