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

    /// Set bit `idx` to 1.
    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len_bits);
        let (word, bit) = split_index(idx);
        unsafe {
            *self.data.get_unchecked_mut(word) |= 1u64 << bit;
        }
    }

    /// Set bit `idx` to 0.
    #[inline(always)]
    pub fn clear(&mut self, idx: usize) {
        debug_assert!(idx < self.len_bits);
        let (word, bit) = split_index(idx);
        let mask = !(1u64 << bit);
        unsafe {
            *self.data.get_unchecked_mut(word) &= mask;
        }
    }

    /// Set bit `idx` to `value`.
    #[inline(always)]
    pub fn set_to(&mut self, idx: usize, value: bool) {
        if value {
            self.set(idx);
        } else {
            self.clear(idx);
        }
    }

    /// Read bit `idx`.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
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

    /// Returns true if *any* bit is 1 in [start, start+len).
    ///
    /// PoMap only calls this with len <= MAX_SCAN (16),
    /// so this is optimized for that and touches at most 2 words.
    #[inline(always)]
    pub fn any_one_in_range(&self, start: usize, len: usize) -> bool {
        if len == 0 {
            return false;
        }
        debug_assert!(len <= 16, "this bitset is tuned for MAX_SCAN = 16");
        debug_assert!(start + len <= self.len_bits);

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

    /// Find the first 0-bit (vacant slot) in [start, start+len).
    ///
    /// Returns the *global* bit index if found, or None.
    ///
    /// Again, tuned for len <= MAX_SCAN (16); at most 2 word loads.
    #[inline(always)]
    pub fn first_zero_in_range(&self, start: usize, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        debug_assert!(len <= 16, "this bitset is tuned for MAX_SCAN = 16");
        debug_assert!(start + len <= self.len_bits);

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
    /// starting at offset `start_slot` within that bucket, and wrapping around.
    ///
    /// - `bucket_start` is the global bit index of the bucket's first slot.
    /// - `bucket_len` will be MAX_SCAN (=16) for PoMap.
    /// - `start_slot` is 0..bucket_len (usually the probe offset).
    #[inline(always)]
    pub fn first_zero_in_bucket(
        &self,
        bucket_start: usize,
        bucket_len: usize,
        start_slot: usize,
    ) -> Option<usize> {
        debug_assert!(bucket_len <= 16, "bucket_len should be <= MAX_SCAN");
        debug_assert!(start_slot < bucket_len);
        debug_assert!(bucket_start + bucket_len <= self.len_bits);

        // First try [bucket_start + start_slot .. bucket_start + bucket_len)
        let first_len = bucket_len - start_slot;
        if let Some(idx) = self.first_zero_in_range(bucket_start + start_slot, first_len) {
            return Some(idx);
        }

        // Then wrap to [bucket_start .. bucket_start + start_slot)
        if start_slot > 0 {
            self.first_zero_in_range(bucket_start, start_slot)
        } else {
            None
        }
    }
}
