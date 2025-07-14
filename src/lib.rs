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
/// and an elegant, single-pass resize operation.
#[derive(Clone, Debug)]
pub struct PoMap<K: Hash + Eq + Clone, V: Clone> {
    capacity: usize,
    len: usize,
    k: usize, // Bucket size: K = log(capacity)
    p: u32,   // Number of prefix bits
    num_buckets: usize,
    data: Vec<Option<Entry<K, V>>>,
    // Index buckets are length-prefixed. The first u8 is the length,
    // followed by K u8 relative offsets.
    indices: Vec<u8>,
}

impl<K: Hash + Eq + Clone, V: Clone> PoMap<K, V> {
    /// Creates a new, empty PoMap.
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
            // Each bucket gets K slots for indices + 1 slot for its length.
            indices: vec![0; num_buckets * (k + 1)],
        }
    }

    fn calculate_layout(capacity: usize) -> (usize, u32, usize) {
        if capacity == 0 {
            return (0, 0, 0);
        }
        // Clamp to 255 so the length prefix `k` can also be stored in a u8.
        let k = (capacity.ilog2() as usize).clamp(1, 255);
        let num_buckets = capacity / k;
        let p = if num_buckets > 0 {
            num_buckets.ilog2()
        } else {
            0
        };
        (k, p, num_buckets)
    }

    fn hash_key(key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn get_bucket_idx(&self, hash: u64) -> usize {
        if self.num_buckets == 0 {
            return 0;
        }
        (hash >> (64 - self.p)) as usize % self.num_buckets
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn find_key_in_bucket(&self, bucket_idx: usize, key: &K, hash: u64) -> Option<(usize, usize)> {
        let data_base = bucket_idx * self.k;
        let index_base = bucket_idx * (self.k + 1);
        let len = self.indices[index_base] as usize;

        let index_slice = &self.indices[index_base + 1..index_base + 1 + len];

        for (i, rel_idx) in index_slice.iter().enumerate() {
            let abs_idx = data_base + *rel_idx as usize;
            if let Some(entry) = &self.data[abs_idx] {
                if entry.hash == hash && entry.key == *key {
                    // Return absolute data index and position within the index slice
                    return Some((abs_idx, i));
                }
            }
        }
        None
    }

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

        if let Some((abs_idx, _)) = self.find_key_in_bucket(bucket_idx, &key, hash) {
            return Some(mem::replace(
                &mut self.data[abs_idx].as_mut().unwrap().value,
                value,
            ));
        }

        let data_base = bucket_idx * self.k;
        let index_base = bucket_idx * (self.k + 1);
        let len = self.indices[index_base] as usize;

        if len >= self.k {
            self.resize(self.capacity * 2);
            return self.insert(key, value);
        }

        let mut rel_data_idx: Option<u8> = None;
        for i in 0..self.k {
            if self.data[data_base + i].is_none() {
                rel_data_idx = Some(i as u8);
                break;
            }
        }
        let rel_data_idx = rel_data_idx.unwrap(); // Should always find a slot if len < k

        self.data[data_base + rel_data_idx as usize] = Some(Entry { key, value, hash });

        let index_slice = &self.indices[index_base + 1..index_base + 1 + len];
        let mut insert_pos = len;
        for (i, existing_rel_idx) in index_slice.iter().enumerate() {
            let existing_hash = self.data[data_base + *existing_rel_idx as usize]
                .as_ref()
                .unwrap()
                .hash;
            if hash < existing_hash {
                insert_pos = i;
                break;
            }
        }

        let insert_offset = index_base + 1 + insert_pos;
        self.indices[insert_offset..index_base + 1 + len + 1].rotate_right(1);
        self.indices[insert_offset] = rel_data_idx;
        self.indices[index_base] += 1; // Increment length
        self.len += 1;
        None
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        if self.len == 0 {
            return None;
        }
        let hash = Self::hash_key(key);
        let bucket_idx = self.get_bucket_idx(hash);

        self.find_key_in_bucket(bucket_idx, key, hash)
            .and_then(|(abs_idx, _)| self.data[abs_idx].as_ref().map(|e| &e.value))
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.len == 0 {
            return None;
        }
        let hash = Self::hash_key(key);
        let bucket_idx = self.get_bucket_idx(hash);

        let (abs_idx_to_remove, index_pos_to_remove) =
            self.find_key_in_bucket(bucket_idx, key, hash)?;

        let removed_entry = self.data[abs_idx_to_remove].take().unwrap();

        let index_base = bucket_idx * (self.k + 1);
        let len = self.indices[index_base] as usize;
        let index_slice_start = index_base + 1 + index_pos_to_remove;
        let index_slice_end = index_base + 1 + len;

        self.indices[index_slice_start..index_slice_end].rotate_left(1);

        self.indices[index_base] -= 1; // Decrement length
        self.len -= 1;
        Some(removed_entry.value)
    }

    fn resize(&mut self, new_capacity: usize) {
        let (new_k, new_p, new_num_buckets) = Self::calculate_layout(new_capacity);
        let mut new_data = vec![None; new_capacity];
        let mut new_indices = vec![0; new_num_buckets * (new_k + 1)];

        let old_data = mem::take(&mut self.data);
        let old_indices = mem::take(&mut self.indices);

        if self.num_buckets > 0 {
            for old_bucket_idx in 0..self.num_buckets {
                let old_data_base = old_bucket_idx * self.k;
                let old_index_base = old_bucket_idx * (self.k + 1);
                let old_len = old_indices[old_index_base] as usize;

                let old_index_slice =
                    &old_indices[old_index_base + 1..old_index_base + 1 + old_len];

                for rel_old_idx in old_index_slice {
                    let abs_old_idx = old_data_base + *rel_old_idx as usize;

                    if let Some(element) = &old_data[abs_old_idx] {
                        let new_bucket_idx =
                            (element.hash >> (64 - new_p)) as usize % new_num_buckets;
                        let new_data_base = new_bucket_idx * new_k;
                        let new_index_base = new_bucket_idx * (new_k + 1);

                        let rel_data_idx = new_indices[new_index_base]; // current len is next free slot
                        let abs_new_data_idx = new_data_base + rel_data_idx as usize;
                        new_data[abs_new_data_idx] = Some(element.clone());

                        let rel_index_pos = new_indices[new_index_base];
                        let abs_new_index_pos = new_index_base + 1 + rel_index_pos as usize;
                        new_indices[abs_new_index_pos] = rel_data_idx;
                        new_indices[new_index_base] += 1; // Increment len
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

        for i in 0..13 {
            map.insert(i.to_string(), i);
        }

        assert_eq!(map.len(), 13);
        assert!(map.capacity() >= 32);

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
