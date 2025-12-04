/// Minimal, PoMap-optimized bitset:
/// - 1 bit per slot
/// - tuned for very small ranges (MAX_SCAN = 16)
#[derive(Clone)]
pub struct Bitset {
    data: Vec<u64>,
    len_bits: usize,
}

const WORD_SHIFT: usize = 6;
const WORD_BITS: usize = 1 << WORD_SHIFT;
const WORD_MASK: usize = WORD_BITS - 1;
const RANGE_LIMIT: usize = super::pomap::MAX_SCAN; // max 64 on 32-bit, 128 on 64-bit

#[inline(always)]
const fn split_index(idx: usize) -> (usize, usize) {
    (idx >> WORD_SHIFT, idx & WORD_MASK)
}

impl Default for Bitset {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Bitset {
    /// Create an empty bitset.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            len_bits: 0,
        }
    }

    /// Create a bitset with space for `bits` bits, all initialized to 0.
    #[inline(always)]
    pub fn with_len(bits: usize) -> Self {
        let words = bits.div_ceil(WORD_BITS);
        Self {
            data: vec![0u64; words],
            len_bits: bits,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len_bits
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    /// Set bit `idx` to 1. Clamps out-of-range indices to a no-op.
    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        if idx < self.len_bits {
            unsafe { self.set_unchecked(idx) };
        }
    }

    /// Set bit `idx` to 1 without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `idx < self.len_bits`.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, idx: usize) {
        debug_assert!(idx < self.len_bits);
        let (word, bit) = split_index(idx);
        unsafe {
            *self.data.get_unchecked_mut(word) |= 1u64 << bit;
        }
    }

    /// Set bit `idx` to 0. Clamps out-of-range indices to a no-op.
    #[inline(always)]
    pub fn clear(&mut self, idx: usize) {
        if idx < self.len_bits {
            unsafe { self.clear_unchecked(idx) };
        }
    }

    /// Set bit `idx` to 0 without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `idx < self.len_bits`.
    #[inline(always)]
    pub unsafe fn clear_unchecked(&mut self, idx: usize) {
        debug_assert!(idx < self.len_bits);
        let (word, bit) = split_index(idx);
        let mask = !(1u64 << bit);
        unsafe {
            *self.data.get_unchecked_mut(word) &= mask;
        }
    }

    /// Set bit `idx` to `value`. Clamps out-of-range indices to a no-op.
    #[inline(always)]
    pub fn set_to(&mut self, idx: usize, value: bool) {
        if idx < self.len_bits {
            unsafe { self.set_to_unchecked(idx, value) };
        }
    }

    /// Set bit `idx` to `value` without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `idx < self.len_bits`.
    #[inline(always)]
    pub unsafe fn set_to_unchecked(&mut self, idx: usize, value: bool) {
        if value {
            unsafe { self.set_unchecked(idx) };
        } else {
            unsafe { self.clear_unchecked(idx) };
        }
    }

    /// Read bit `idx`. Out-of-range indices return false.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        if idx < self.len_bits {
            unsafe { self.get_unchecked(idx) }
        } else {
            false
        }
    }

    /// Read bit `idx` without bounds checks.
    ///
    /// # Safety
    /// Caller must ensure `idx < self.len_bits`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len_bits);
        let (word, bit) = split_index(idx);
        unsafe { ((*self.data.get_unchecked(word) >> bit) & 1) != 0 }
    }

    /// Clear all bits to 0, keeping capacity.
    #[inline(always)]
    pub fn clear_all(&mut self) {
        unsafe {
            core::ptr::write_bytes(self.data.as_mut_ptr(), 0, self.data.len());
        }
    }

    /// Returns true if *any* bit is 1 in [start, start+len), clamping to the
    /// available length and to RANGE_LIMIT (16).
    #[inline(always)]
    pub fn any_one_in_range(&self, start: usize, len: usize) -> bool {
        if let Some((s, l)) = clamp_range(start, len, self.len_bits) {
            let l = l.min(RANGE_LIMIT);
            if l == 0 {
                return false;
            }
            unsafe { self.any_one_in_range_unchecked(s, l) }
        } else {
            false
        }
    }

    /// Returns true if *any* bit is 1 in [start, start+len) without clamping.
    ///
    /// PoMap only calls this with len <= RANGE_LIMIT (16),
    /// so this is optimized for that and touches at most 2 words. Caller must
    /// uphold that bound.
    #[inline(always)]
    pub unsafe fn any_one_in_range_unchecked(&self, start: usize, len: usize) -> bool {
        debug_assert!(len <= RANGE_LIMIT, "this bitset is tuned for MAX_SCAN = 16");
        debug_assert!(start + len <= self.len_bits);
        debug_assert!(len > 0);

        let (word_idx, bit_off) = split_index(start);

        // Range fits entirely in one word.
        if bit_off + len <= WORD_BITS {
            let mask = ((1u64 << len) - 1) << bit_off;
            let word = unsafe { *self.data.get_unchecked(word_idx) };
            (word & mask) != 0
        } else {
            // Crosses a word boundary: part in word_idx, rest in word_idx+1.
            let bits_first = WORD_BITS - bit_off;
            let bits_second = len - bits_first;

            // First word: bits [bit_off .. 63]
            let mask_first = (!0u64) << bit_off;
            let first = unsafe { *self.data.get_unchecked(word_idx) };
            if (first & mask_first) != 0 {
                return true;
            }

            // Second word: bits [0 .. bits_second)
            // bits_second <= 16 and < 64, so shift is safe.
            let mask_second = (1u64 << bits_second) - 1;
            let second = unsafe { *self.data.get_unchecked(word_idx + 1) };
            (second & mask_second) != 0
        }
    }

    /// Find the first 0-bit (vacant slot) in [start, start+len), clamping to
    /// the available length and RANGE_LIMIT (16).
    #[inline(always)]
    pub fn first_zero_in_range(&self, start: usize, len: usize) -> Option<usize> {
        if let Some((s, l)) = clamp_range(start, len, self.len_bits) {
            let l = l.min(RANGE_LIMIT);
            if l == 0 {
                return None;
            }
            unsafe { self.first_zero_in_range_unchecked(s, l) }
        } else {
            None
        }
    }

    /// Find the first 0-bit (vacant slot) in [start, start+len) without clamping.
    ///
    /// Returns the *global* bit index if found, or None.
    ///
    /// Again, tuned for len <= MAX_SCAN (16); at most 2 word loads.
    #[inline(always)]
    pub unsafe fn first_zero_in_range_unchecked(&self, start: usize, len: usize) -> Option<usize> {
        debug_assert!(len <= RANGE_LIMIT, "this bitset is tuned for MAX_SCAN = 16");
        debug_assert!(start + len <= self.len_bits);
        debug_assert!(len > 0);

        let (word_idx, bit_off) = split_index(start);

        // Range fits entirely in one word.
        if bit_off + len <= WORD_BITS {
            let mask = ((1u64 << len) - 1) << bit_off;
            let zeros = (!unsafe { *self.data.get_unchecked(word_idx) }) & mask;
            if zeros != 0 {
                let tz = zeros.trailing_zeros() as usize;
                return Some((word_idx << WORD_SHIFT) + tz);
            }
            return None;
        }

        // Crosses word boundary.
        let bits_first = WORD_BITS - bit_off;
        let bits_second = len - bits_first;

        // First word: bits [bit_off .. 63]
        let mask_first = (!0u64) << bit_off;
        let zeros_first = (!unsafe { *self.data.get_unchecked(word_idx) }) & mask_first;
        if zeros_first != 0 {
            let tz = zeros_first.trailing_zeros() as usize;
            return Some((word_idx << WORD_SHIFT) + tz);
        }

        // Second word: bits [0 .. bits_second)
        let mask_second = (1u64 << bits_second) - 1;
        let zeros_second = (!unsafe { *self.data.get_unchecked(word_idx + 1) }) & mask_second;
        if zeros_second != 0 {
            let tz = zeros_second.trailing_zeros() as usize;
            return Some(((word_idx + 1) << WORD_SHIFT) + tz);
        }

        None
    }

    /// PoMap helper: search for the first 0-bit inside a "bucket" window of length `bucket_len`,
    /// starting at offset `start_slot` within that bucket, and wrapping around. Safe wrapper
    /// clamps to in-bounds ranges and RANGE_LIMIT.
    #[inline(always)]
    pub fn first_zero_in_bucket(
        &self,
        bucket_start: usize,
        bucket_len: usize,
        start_slot: usize,
    ) -> Option<usize> {
        let Some((bucket_start, raw_len)) = clamp_range(bucket_start, bucket_len, self.len_bits)
        else {
            return None;
        };
        let bucket_len = raw_len.min(RANGE_LIMIT);
        if bucket_len == 0 {
            return None;
        }
        if start_slot >= bucket_len {
            return None;
        }
        unsafe { self.first_zero_in_bucket_unchecked(bucket_start, bucket_len, start_slot) }
    }

    /// PoMap helper: search for the first 0-bit inside a "bucket" window of length `bucket_len`,
    /// starting at offset `start_slot` within that bucket, and wrapping around. Unsafe and
    /// unchecked; caller must guarantee bounds.
    ///
    /// - `bucket_start` is the global bit index of the bucket's first slot.
    /// - `bucket_len` will be MAX_SCAN (=16) for PoMap.
    /// - `start_slot` is 0..bucket_len (usually the probe offset).
    #[inline(always)]
    pub unsafe fn first_zero_in_bucket_unchecked(
        &self,
        bucket_start: usize,
        bucket_len: usize,
        start_slot: usize,
    ) -> Option<usize> {
        debug_assert!(
            bucket_len <= RANGE_LIMIT,
            "bucket_len should be <= MAX_SCAN"
        );
        debug_assert!(start_slot < bucket_len);
        debug_assert!(bucket_start + bucket_len <= self.len_bits);

        // First try [bucket_start + start_slot .. bucket_start + bucket_len)
        let first_len = bucket_len - start_slot;
        if let Some(idx) =
            unsafe { self.first_zero_in_range_unchecked(bucket_start + start_slot, first_len) }
        {
            return Some(idx);
        }

        // Then wrap to [bucket_start .. bucket_start + start_slot)
        if start_slot > 0 {
            unsafe { self.first_zero_in_range_unchecked(bucket_start, start_slot) }
        } else {
            None
        }
    }
}

#[inline(always)]
fn clamp_range(start: usize, len: usize, max_len: usize) -> Option<(usize, usize)> {
    if start >= max_len {
        return None;
    }
    let remaining = max_len - start;
    let clamped_len = len.min(remaining);
    if clamped_len == 0 {
        None
    } else {
        Some((start, clamped_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn new_and_with_len_establish_capacity() {
        let empty = Bitset::new();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.data.len(), 0);

        let single = Bitset::with_len(1);
        assert_eq!(single.len(), 1);
        assert!(!single.is_empty());
        assert_eq!(single.data.len(), 1);

        let full_word = Bitset::with_len(WORD_BITS);
        assert_eq!(full_word.data.len(), 1);

        let two_words = Bitset::with_len(WORD_BITS + 1);
        assert_eq!(two_words.data.len(), 2);
    }

    #[test]
    fn set_get_clear_roundtrip_edges() {
        let mut bits = Bitset::with_len((WORD_BITS * 2) + 2);
        let indices = [0, WORD_BITS - 1, WORD_BITS, bits.len() - 1];

        for &idx in &indices {
            bits.set(idx);
            assert!(bits.get(idx));
        }

        for &idx in &indices {
            bits.clear(idx);
            assert!(!bits.get(idx));
        }
    }

    #[test]
    fn clear_all_resets_words() {
        let mut bits = Bitset::with_len(WORD_BITS * 3);
        for i in (0..bits.len()).step_by(3) {
            bits.set(i);
        }

        bits.clear_all();
        assert!(bits.data.iter().all(|&w| w == 0));
        for i in 0..bits.len() {
            assert!(!bits.get(i));
        }
    }

    #[test]
    fn any_one_in_range_handles_single_and_cross_word() {
        let mut bits = Bitset::with_len(WORD_BITS * 2);
        bits.set(5);
        bits.set(WORD_BITS - 1);
        bits.set(WORD_BITS);
        bits.set(WORD_BITS + 3);

        assert!(!bits.any_one_in_range(0, 5));
        assert!(bits.any_one_in_range(5, 1));
        assert!(bits.any_one_in_range(4, 4));

        assert!(bits.any_one_in_range(WORD_BITS - 2, 4));
        assert!(bits.any_one_in_range(WORD_BITS - 1, 1));
        assert!(bits.any_one_in_range(WORD_BITS, 1));
        assert!(!bits.any_one_in_range(WORD_BITS + 1, 2));
        assert!(bits.any_one_in_range(WORD_BITS + 2, 2));
    }

    #[test]
    fn safe_methods_clamp_out_of_range() {
        let mut bits = Bitset::with_len(10);
        bits.set(15);
        bits.set_to(20, true);
        assert!(bits.data.iter().all(|&w| w == 0));
        assert!(!bits.get(15));

        bits.clear(25); // no panic, no effect
        assert!(!bits.get(9));
    }

    #[test]
    fn first_zero_in_range_single_and_cross_word() {
        let mut bits = Bitset::with_len(WORD_BITS * 2);
        assert_eq!(bits.first_zero_in_range(0, 8), Some(0));

        for i in 0..8 {
            bits.set(i);
        }
        assert_eq!(bits.first_zero_in_range(0, 8), None);
        assert_eq!(bits.first_zero_in_range(0, 9), Some(8));

        for i in (WORD_BITS - 8)..(WORD_BITS + 4) {
            bits.set(i);
        }
        bits.clear(WORD_BITS - 3);
        bits.clear(WORD_BITS + 1);

        assert_eq!(
            bits.first_zero_in_range(WORD_BITS - 8, 12),
            Some(WORD_BITS - 3)
        );

        bits.set(WORD_BITS - 3);
        assert_eq!(
            bits.first_zero_in_range(WORD_BITS - 8, 12),
            Some(WORD_BITS + 1)
        );
    }

    #[test]
    fn range_methods_clamp_len_and_bounds() {
        let mut bits = Bitset::with_len(WORD_BITS * 2);
        unsafe { bits.set_unchecked(WORD_BITS + 5) };

        // len clamps to RANGE_LIMIT
        assert!(bits.any_one_in_range(WORD_BITS, 100));
        // start beyond length returns false
        assert!(!bits.any_one_in_range(bits.len(), 1));
        assert_eq!(bits.first_zero_in_range(bits.len(), 4), None);
    }

    #[test]
    fn first_zero_in_bucket_wraps_when_needed() {
        let mut bits = Bitset::with_len(32);
        let bucket_start = 8;
        let bucket_len = 8;

        for i in 0..bucket_len {
            bits.set(bucket_start + i);
        }
        bits.clear(bucket_start + 2);

        assert_eq!(
            bits.first_zero_in_bucket(bucket_start, bucket_len, 5),
            Some(bucket_start + 2)
        );

        bits.set(bucket_start + 2);
        assert_eq!(bits.first_zero_in_bucket(bucket_start, bucket_len, 0), None);
    }

    #[test]
    fn bucket_clamps_to_available_bits() {
        let mut bits = Bitset::with_len(20);
        let bucket_start = 18; // only 2 bits available
        let bucket_len = 8;
        bits.set(bucket_start);

        // start_slot beyond clamped len returns None.
        assert_eq!(bits.first_zero_in_bucket(bucket_start, bucket_len, 5), None);

        // With valid start_slot, clamps bucket_len to remaining bits and finds the zero.
        assert_eq!(
            bits.first_zero_in_bucket(bucket_start, bucket_len, 1),
            Some(19)
        );
    }

    #[test]
    fn unsafe_variants_require_no_clamp() {
        let mut bits = Bitset::with_len(WORD_BITS);
        unsafe { bits.set_unchecked(3) };
        unsafe { bits.clear_unchecked(2) };
        assert!(unsafe { bits.get_unchecked(3) });
        assert!(!unsafe { bits.any_one_in_range_unchecked(0, 1) });
        assert!(unsafe { bits.any_one_in_range_unchecked(0, 4) });
        assert_eq!(unsafe { bits.first_zero_in_range_unchecked(0, 4) }, Some(0));
    }

    #[test]
    fn randomized_queries_match_bool_reference() {
        let len_bits = 256;
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);
        let mut bits = Bitset::with_len(len_bits);
        let mut reference = vec![false; len_bits];

        for _ in 0..500 {
            let idx = rng.random_range(0..len_bits);
            let val = rng.random_bool(0.5);
            bits.set_to(idx, val);
            reference[idx] = val;

            let start = rng.random_range(0..len_bits);
            let remaining = len_bits - start;
            let range_len = rng.random_range(0..=remaining.min(16));
            let slice = &reference[start..start + range_len];

            let expected_any = slice.iter().any(|&b| b);
            let expected_zero = slice.iter().position(|b| !b).map(|off| start + off);

            assert_eq!(bits.any_one_in_range(start, range_len), expected_any);
            assert_eq!(bits.first_zero_in_range(start, range_len), expected_zero);
        }
    }
}
