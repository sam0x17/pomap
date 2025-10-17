use std::hash::{DefaultHasher, Hash, Hasher};

const DEFAULT_CAPACITY: usize = 16;
const GROWTH_FACTOR: f64 = 2.0;

#[inline(always)]
pub fn num_buckets(n: usize) -> usize {
    (n / (n).ilog2() as usize).max(1)
    // (n as f64).sqrt() as usize
}

#[inline(always)]
const fn grow_capacity(current: usize) -> usize {
    let res = (current as f64 * GROWTH_FACTOR).ceil() as usize;
    if res < 1 { 1 } else { res }
}

#[inline(always)]
const fn bucket_id_from_bits(hash: u64, bucket_bits: u8) -> usize {
    let shift = 64 - bucket_bits as u32;
    let mask = (1usize << bucket_bits) - 1;
    ((hash >> shift) as usize) & mask
}

#[inline(always)]
const fn bucket_slot_from_bits(hash: u64, bucket_bits: u8, bucket_len_bits: u8) -> usize {
    if bucket_len_bits == 0 {
        0
    } else {
        let shift = 64 - bucket_bits as u32 - bucket_len_bits as u32;
        let mask = (1usize << bucket_len_bits) - 1;
        ((hash >> shift) as usize) & mask
    }
}

pub trait Key: Hash + Eq + Clone + Ord {}
impl<K: Hash + Eq + Clone + Ord> Key for K {}

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
    bucket_len_bits: u8,
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
            bucket_len_bits: 0,
            bucket_len: 0,
            len: 0,
            entries: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.bucket_bits = 0;
        self.bucket_len_bits = 0;
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
    fn bucket_slot_offset(&self, hash: u64) -> usize {
        bucket_slot_from_bits(hash, self.bucket_bits, self.bucket_len_bits)
    }

    #[inline(always)]
    fn place_entry_in_slice(
        entries: &mut [Option<Entry<K, V>>],
        bucket_len: usize,
        bucket_bits: u8,
        bucket_len_bits: u8,
        bucket_id: usize,
        entry: Entry<K, V>,
    ) -> Result<(), Entry<K, V>> {
        if bucket_len == 0 {
            return Err(entry);
        }
        let start = bucket_id * bucket_len;
        let mask = bucket_len - 1;
        let mut slot = bucket_slot_from_bits(entry.hash, bucket_bits, bucket_len_bits);
        for _ in 0..bucket_len {
            let idx = start + slot;
            if entries[idx].is_none() {
                entries[idx] = Some(entry);
                return Ok(());
            }
            slot = (slot + 1) & mask;
        }
        Err(entry)
    }

    #[inline(always)]
    fn place_entry_in_bucket(&mut self, bucket_id: usize, entry: Entry<K, V>) -> bool {
        Self::place_entry_in_slice(
            &mut self.entries,
            self.bucket_len,
            self.bucket_bits,
            self.bucket_len_bits,
            bucket_id,
            entry,
        )
        .is_ok()
    }

    #[inline(always)]
    fn entry_matches(entry: &Entry<K, V>, hash: u64, key: &K) -> bool {
        entry.hash == hash && entry.key == *key
    }

    #[inline(always)]
    fn rebuild(&mut self, min_capacity: usize) {
        let mut old_entries: Vec<_> = std::mem::take(&mut self.entries)
            .into_iter()
            .flatten()
            .collect();

        let mut required = min_capacity.max(self.len.max(1));

        loop {
            let total_slots = required.next_power_of_two();
            let bucket_count = num_buckets(total_slots).next_power_of_two();
            let bucket_bits = bucket_count.trailing_zeros() as u8;
            let bucket_len = total_slots / bucket_count;
            let bucket_len_bits = bucket_len.trailing_zeros() as u8;
            debug_assert!(
                (bucket_bits as u32 + bucket_len_bits as u32) <= 64,
                "bucket and slot bits exceed hash width"
            );
            let mut new_entries = vec![None; total_slots];
            let mut success = true;

            while let Some(entry) = old_entries.pop() {
                let bucket_id = bucket_id_from_bits(entry.hash, bucket_bits);
                debug_assert!(
                    bucket_id < bucket_count,
                    "bucket_id out of range during rebuild"
                );
                match Self::place_entry_in_slice(
                    &mut new_entries,
                    bucket_len,
                    bucket_bits,
                    bucket_len_bits,
                    bucket_id,
                    entry,
                ) {
                    Ok(()) => {}
                    Err(entry) => {
                        success = false;
                        old_entries.push(entry);
                        break;
                    }
                }
            }

            if success && old_entries.is_empty() {
                self.entries = new_entries;
                self.bucket_bits = bucket_bits;
                self.bucket_len = bucket_len;
                self.bucket_len_bits = bucket_len_bits;
                return;
            }

            old_entries.extend(new_entries.into_iter().flatten());
            required = grow_capacity(total_slots);
        }
    }

    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.len == 0 || self.bucket_len == 0 {
            return None;
        }
        let hash = Self::calculate_hash(key);
        let bucket_id = self.bucket_id_for(hash);
        let (start, end) = self.bucket_bounds(bucket_id);
        debug_assert!(end <= self.entries.len(), "bucket bounds out of range");
        let mask = self.bucket_len - 1;
        let mut slot = self.bucket_slot_offset(hash);
        for _ in 0..self.bucket_len {
            let idx = start + slot;
            match self.entries[idx].as_ref() {
                Some(entry) => {
                    if Self::entry_matches(entry, hash, key) {
                        return Some(&entry.value);
                    }
                }
                None => return None,
            }
            slot = (slot + 1) & mask;
        }
        None
    }

    #[inline(always)]
    fn insert_with_hash(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        if self.entries.is_empty() || self.bucket_len == 0 {
            debug_assert_eq!(self.len, 0);
            self.rebuild(DEFAULT_CAPACITY);
        }
        loop {
            if self.bucket_len == 0 || self.entries.is_empty() {
                self.rebuild(grow_capacity(self.capacity()));
                continue;
            }
            if self.len >= self.capacity() {
                self.rebuild(grow_capacity(self.capacity()));
                continue;
            }
            let bucket_id = self.bucket_id_for(hash);
            let (start, end) = self.bucket_bounds(bucket_id);
            debug_assert!(end <= self.entries.len(), "bucket bounds out of range");
            let mut slot = self.bucket_slot_offset(hash);
            let mask = self.bucket_len - 1;
            for _ in 0..self.bucket_len {
                let idx = start + slot;
                if let Some(entry) = self.entries[idx].as_mut() {
                    if Self::entry_matches(entry, hash, &key) {
                        let old_value = std::mem::replace(&mut entry.value, value);
                        return Some(old_value);
                    }
                } else {
                    self.entries[idx] = Some(Entry { hash, key, value });
                    self.len += 1;
                    return None;
                }
                slot = (slot + 1) & mask;
            }
            self.rebuild(grow_capacity(self.capacity()));
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
        debug_assert!(end <= self.entries.len(), "bucket bounds out of range");
        let mask = self.bucket_len - 1;
        let mut slot = self.bucket_slot_offset(hash);
        let mut found_index = None;

        for _ in 0..self.bucket_len {
            let idx = start + slot;
            match self.entries[idx].as_ref() {
                Some(entry) => {
                    if Self::entry_matches(entry, hash, key) {
                        found_index = Some(idx);
                        break;
                    }
                }
                None => return None,
            }
            slot = (slot + 1) & mask;
        }

        let index = match found_index {
            Some(idx) => idx,
            None => return None,
        };

        let removed_entry = match self.entries[index].take() {
            Some(entry) => entry,
            None => return None,
        };

        let mut probe_slot = ((index - start) + 1) & mask;
        let mut to_reinsert = Vec::with_capacity(self.bucket_len.saturating_sub(1));
        for _ in 0..self.bucket_len - 1 {
            let idx = start + probe_slot;
            match self.entries[idx].take() {
                Some(entry) => {
                    to_reinsert.push(entry);
                }
                None => break,
            }
            probe_slot = (probe_slot + 1) & mask;
        }

        for entry in to_reinsert {
            debug_assert_eq!(
                bucket_id_from_bits(entry.hash, self.bucket_bits),
                bucket_id,
                "entry must stay within its bucket"
            );
            let placed = self.place_entry_in_bucket(bucket_id, entry);
            debug_assert!(placed, "reinsertion within bucket must succeed");
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
    use std::cmp::Ordering;

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
        assert_eq!(map.capacity(), grow_capacity(DEFAULT_CAPACITY));

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

        let num_items = 100_000;
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
