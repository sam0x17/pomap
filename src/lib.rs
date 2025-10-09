use std::{
    cmp::Ordering,
    hash::{DefaultHasher, Hash, Hasher},
};

const DEFAULT_CAPACITY: usize = 16;
const GROWTH_FACTOR: usize = 2;
const C: f64 = 1.5;

#[inline(always)]
pub fn num_buckets(n: usize) -> usize {
    (n as f64 / (C * (n as f64).log2())).max(1.0) as usize
}

#[inline(always)]
const fn bucket_id_from_bits(hash: u64, bucket_bits: u8) -> usize {
    let shift = 64 - bucket_bits as u32;
    let mask = (1usize << bucket_bits) - 1;
    ((hash >> shift) as usize) & mask
}

#[cfg(feature = "compare-keys")]
pub trait Key: Hash + Eq + Clone + Ord {}
#[cfg(feature = "compare-keys")]
impl<K: Hash + Eq + Clone + Ord> Key for K {}
#[cfg(not(feature = "compare-keys"))]
pub trait Key: Hash + Eq + Clone {}
#[cfg(not(feature = "compare-keys"))]
impl<K: Hash + Eq + Clone> Key for K {}

pub trait Value: Clone {}
impl<V: Clone> Value for V {}

#[derive(Hash, Clone, Debug)]
struct Entry<K: Key, V: Value> {
    hash: u64,
    key: K,
    value: V,
}

#[derive(Clone)]
pub struct PoMap<K: Key, V: Value> {
    bucket_bits: u8,
    bucket_len: usize,
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
            bucket_bits: 0,
            bucket_len: 0,
            len: 0,
            entries: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.bucket_bits = 0;
        self.bucket_len = 0;
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
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let mut map = Self::new();
        map.rebuild(capacity);
        map
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
    fn bucket_id_for(&self, hash: u64) -> usize {
        if self.entries.is_empty() {
            0
        } else {
            bucket_id_from_bits(hash, self.bucket_bits)
        }
    }

    #[inline(always)]
    fn bucket_bounds(&self, bucket_id: usize) -> (usize, usize) {
        let start = bucket_id * self.bucket_len;
        let end = start + self.bucket_len;
        (start, end)
    }

    #[inline(always)]
    fn compare_entry(entry: &Entry<K, V>, hash: u64, key: &K) -> Ordering {
        #[cfg(feature = "compare-keys")]
        {
            entry.hash.cmp(&hash).then_with(|| entry.key.cmp(key))
        }
        #[cfg(not(feature = "compare-keys"))]
        {
            let _ = key;
            entry.hash.cmp(&hash)
        }
    }

    #[inline(always)]
    fn entry_matches(entry: &Entry<K, V>, hash: u64, key: &K) -> bool {
        entry.hash == hash && entry.key == *key
    }

    #[inline(always)]
    fn rebuild(&mut self, min_capacity: usize) {
        let required = min_capacity.max(self.len.max(1));
        let mut total_slots = required.next_power_of_two();
        if total_slots < DEFAULT_CAPACITY {
            total_slots = DEFAULT_CAPACITY;
        }
        let mut bucket_count = num_buckets(total_slots).max(1);
        bucket_count = bucket_count.next_power_of_two();
        if bucket_count > total_slots {
            bucket_count = total_slots;
        }
        let bucket_bits = bucket_count.trailing_zeros() as u8;
        let bucket_len = total_slots / bucket_count;

        let mut new_entries = vec![None; total_slots];
        let mut bucket_filled = vec![0usize; bucket_count];
        for entry in std::mem::take(&mut self.entries).into_iter().flatten() {
            let bucket_id = bucket_id_from_bits(entry.hash, bucket_bits);
            let fill = &mut bucket_filled[bucket_id];
            debug_assert!(*fill < bucket_len, "bucket overflow during rebuild");
            let offset = bucket_id * bucket_len + *fill;
            new_entries[offset] = Some(entry);
            *fill += 1;
        }
        self.entries = new_entries;
        self.bucket_bits = bucket_bits;
        self.bucket_len = bucket_len;
    }

    #[inline(always)]
    fn growth_target(&self) -> usize {
        self.capacity()
            .max(DEFAULT_CAPACITY)
            .saturating_mul(GROWTH_FACTOR)
    }

    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.len == 0 || self.bucket_len == 0 {
            return None;
        }
        let hash = Self::calculate_hash(key);
        let bucket_id = self.bucket_id_for(hash);
        let (start, end) = self.bucket_bounds(bucket_id);
        if end > self.entries.len() {
            return None;
        }
        #[cfg(feature = "binary-search")]
        {
            if let Ok(index) = self.entries[start..end].binary_search_by(|slot| match slot {
                Some(entry) => Self::compare_entry(entry, hash, key),
                None => Ordering::Greater,
            }) {
                return self.entries[start + index]
                    .as_ref()
                    .map(|entry| &entry.value);
            }
        }
        #[cfg(not(feature = "binary-search"))]
        {
            for i in start..end {
                let Some(entry) = self.entries[i].as_ref() else {
                    break;
                };
                match Self::compare_entry(entry, hash, key) {
                    Ordering::Less => continue,
                    Ordering::Equal => return Some(&entry.value),
                    Ordering::Greater => break,
                }
            }
        }
        None
    }

    #[inline(always)]
    fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        if self.entries.is_empty() || self.bucket_len == 0 {
            debug_assert_eq!(self.len, 0);
            self.rebuild(DEFAULT_CAPACITY);
        }
        'outer: loop {
            if self.bucket_len == 0 || self.entries.is_empty() {
                self.rebuild(self.growth_target());
                continue 'outer;
            }
            if self.len >= self.capacity() {
                self.rebuild(self.growth_target());
                continue 'outer;
            }
            let bucket_id = self.bucket_id_for(hash);
            let (start, end) = self.bucket_bounds(bucket_id);
            debug_assert!(end <= self.entries.len(), "bucket bounds out of range");
            if self.entries[start..end].iter().all(|slot| slot.is_some()) {
                self.rebuild(self.growth_target());
                continue 'outer;
            }
            #[cfg(feature = "binary-search")]
            let scan_start = match self.entries[start..end].binary_search_by(|slot| match slot {
                Some(entry) => Self::compare_entry(entry, hash, &key),
                None => Ordering::Greater,
            }) {
                Ok(index) => {
                    let absolute = start + index;
                    if let Some(existing) = self.entries[absolute].as_mut() {
                        let old_value = std::mem::replace(&mut existing.value, value);
                        return Some(old_value);
                    } else {
                        self.entries[absolute] = Some(Entry { hash, key, value });
                        self.len += 1;
                        return None;
                    }
                }
                Err(index) => start + index,
            };
            #[cfg(not(feature = "binary-search"))]
            let scan_start = start;
            let mut first_free = None;
            for i in scan_start..end {
                match self.entries[i].as_ref() {
                    Some(existing) => match Self::compare_entry(existing, hash, &key) {
                        Ordering::Less => continue,
                        Ordering::Equal => {
                            let entry = self.entries[i].as_mut().expect("entry must exist");
                            let old_value = std::mem::replace(&mut entry.value, value);
                            return Some(old_value);
                        }
                        Ordering::Greater => {
                            let mut j = i;
                            while j < end && self.entries[j].is_some() {
                                j += 1;
                            }
                            if j == end {
                                self.rebuild(self.growth_target());
                                continue 'outer;
                            }
                            for k in (i..j).rev() {
                                let moved = self.entries[k].take();
                                self.entries[k + 1] = moved;
                            }
                            self.entries[i] = Some(Entry { hash, key, value });
                            self.len += 1;
                            return None;
                        }
                    },
                    None => {
                        first_free = Some(i);
                        break;
                    }
                }
            }
            if let Some(index) = first_free {
                self.entries[index] = Some(Entry { hash, key, value });
                self.len += 1;
                return None;
            }
            self.rebuild(self.growth_target());
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
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.len == 0 || self.bucket_len == 0 {
            return None;
        }
        let hash = Self::calculate_hash(key);
        let bucket_id = self.bucket_id_for(hash);
        let (start, end) = self.bucket_bounds(bucket_id);
        if end > self.entries.len() {
            return None;
        }

        #[cfg(feature = "binary-search")]
        let mut candidate = match self.entries[start..end].binary_search_by(|slot| match slot {
            Some(entry) => Self::compare_entry(entry, hash, key),
            None => Ordering::Greater,
        }) {
            Ok(index) => start + index,
            Err(_) => return None,
        };

        #[cfg(not(feature = "binary-search"))]
        let candidate = {
            let mut found = None;
            for i in start..end {
                match self.entries[i].as_ref() {
                    Some(entry) => match Self::compare_entry(entry, hash, key) {
                        Ordering::Less => continue,
                        Ordering::Equal => {
                            if Self::entry_matches(entry, hash, key) {
                                found = Some(i);
                                break;
                            }
                            continue;
                        }
                        Ordering::Greater => break,
                    },
                    None => break,
                }
            }
            match found {
                Some(index) => index,
                None => return None,
            }
        };

        #[cfg(feature = "binary-search")]
        {
            // If the located entry does not match the key (possible when compare-keys is off),
            // search neighboring entries with identical hash values.
            if !self.entries[candidate]
                .as_ref()
                .is_some_and(|entry| Self::entry_matches(entry, hash, key))
            {
                let mut search_index = candidate;
                while search_index > start {
                    search_index -= 1;
                    match self.entries[search_index].as_ref() {
                        Some(entry) if entry.hash == hash => {
                            if Self::entry_matches(entry, hash, key) {
                                candidate = search_index;
                                break;
                            }
                        }
                        Some(_) | None => break,
                    }
                }
                if !self.entries[candidate]
                    .as_ref()
                    .is_some_and(|entry| Self::entry_matches(entry, hash, key))
                {
                    search_index = candidate;
                    while search_index + 1 < end {
                        search_index += 1;
                        match self.entries[search_index].as_ref() {
                            Some(entry) if entry.hash == hash => {
                                if Self::entry_matches(entry, hash, key) {
                                    candidate = search_index;
                                    break;
                                }
                            }
                            Some(_) | None => break,
                        }
                    }
                }
                if !self.entries[candidate]
                    .as_ref()
                    .is_some_and(|entry| Self::entry_matches(entry, hash, key))
                {
                    return None;
                }
            }
        }

        let removed_entry = match self.entries[candidate].take() {
            Some(entry) => entry,
            None => return None,
        };

        for idx in (candidate + 1)..end {
            if let Some(moved) = self.entries[idx].take() {
                self.entries[idx - 1] = Some(moved);
            } else {
                break;
            }
        }

        self.len -= 1;
        Some(removed_entry.value)
    }

    #[inline(always)]
    pub fn reserve(&mut self, capacity: usize) {
        if capacity > self.capacity() {
            self.rebuild(capacity);
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
        let mut map = PoMap::with_capacity(DEFAULT_CAPACITY);
        assert_eq!(map.capacity(), DEFAULT_CAPACITY);

        // Insert enough items to trigger a resize.
        const NUM: usize = 15;
        for i in 0..NUM {
            map.insert(i.to_string(), i);
        }
        assert_eq!(map.len(), NUM);

        map.insert(NUM.to_string(), NUM);
        assert_eq!(map.len(), NUM + 1);
        assert_eq!(map.capacity(), DEFAULT_CAPACITY * GROWTH_FACTOR);

        // Verify all old and new elements are present.
        for i in 0..(NUM + 1) {
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

    impl PartialOrd for CollidingKey {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CollidingKey {
        fn cmp(&self, other: &Self) -> Ordering {
            self.val.cmp(&other.val)
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

    #[test]
    fn test_random_insert_remove_comprehensive() {
        use std::collections::HashMap;

        let mut pomap = PoMap::new();
        let mut hashmap = HashMap::new();
        let num_items = 2_000;

        let mut items: Vec<(i32, i32)> = (0..num_items).map(|i| (i, i * 10)).collect();

        // Simple shuffle
        for i in 0..items.len() {
            let j = (i * 7) % items.len();
            items.swap(i, j);
        }

        // Insert all items
        for (k, v) in &items {
            pomap.insert(*k, *v);
            hashmap.insert(*k, *v);
            assert_eq!(pomap.len(), hashmap.len());
        }

        assert_eq!(pomap.len(), num_items as usize);
        for (k, v) in &items {
            assert_eq!(pomap.get(k), Some(v));
        }

        // Remove about half the items
        let mut removed_count = 0;
        for (i, (k, v)) in items.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(pomap.remove(k), Some(*v));
                hashmap.remove(k);
                removed_count += 1;
            }
            assert_eq!(pomap.len(), hashmap.len());
        }

        assert_eq!(pomap.len(), (num_items - removed_count) as usize);

        // Check final state
        for (k, _) in &items {
            assert_eq!(pomap.get(k), hashmap.get(k));
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

    #[test]
    fn test_clone() {
        let mut map = PoMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut clone = map.clone();
        assert_eq!(clone.len(), 2);
        assert_eq!(clone.get(&"a"), Some(&1));
        assert_eq!(clone.get(&"b"), Some(&2));

        // Modify the clone and check that the original is unaffected
        clone.insert("c", 3);
        clone.remove(&"a");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"c"), None);

        // Modify the original and check that the clone is unaffected
        map.insert("d", 4);
        assert_eq!(clone.len(), 2);
        assert_eq!(clone.get(&"d"), None);
    }

    #[test]
    fn test_zero_sized_types() {
        // Test with ZST value
        let mut map_zst_val = PoMap::new();
        map_zst_val.insert(1, ());
        map_zst_val.insert(2, ());
        assert_eq!(map_zst_val.len(), 2);
        assert_eq!(map_zst_val.get(&1), Some(&()));
        assert_eq!(map_zst_val.remove(&1), Some(()));
        assert_eq!(map_zst_val.len(), 1);
        assert_eq!(map_zst_val.get(&2), Some(&()));

        // Test with ZST key
        // Since all keys are equal, it can only hold one item.
        let mut map_zst_key: PoMap<(), i32> = PoMap::new();
        assert_eq!(map_zst_key.insert((), 100), None);
        assert_eq!(map_zst_key.len(), 1);
        assert_eq!(map_zst_key.insert((), 200), Some(100));
        assert_eq!(map_zst_key.len(), 1);
        assert_eq!(map_zst_key.get(&()), Some(&200));
    }

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
        for (k, v) in &items {
            assert_eq!(pomap.remove(k), Some(*v));
        }
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
