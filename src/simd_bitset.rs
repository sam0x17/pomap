use core::convert::TryInto;

#[cfg(target_pointer_width = "64")]
type Word = u64;
#[cfg(target_pointer_width = "64")]
const WORD_BITS: usize = 64;
#[cfg(target_pointer_width = "64")]
type SimdVec = wide::u64x4;
#[cfg(target_pointer_width = "64")]
const LANES: usize = 4;

#[cfg(target_pointer_width = "32")]
type Word = u32;
#[cfg(target_pointer_width = "32")]
const WORD_BITS: usize = 32;
#[cfg(target_pointer_width = "32")]
type SimdVec = wide::u32x8;
#[cfg(target_pointer_width = "32")]
const LANES: usize = 8;

#[derive(Clone)]
pub struct SimdBitSet {
    data: Vec<Word>,
    len: usize,
}

impl SimdBitSet {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn with_len(bits: usize) -> Self {
        let words = words_for(bits);
        Self {
            data: vec![0; words],
            len: bits,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    #[inline]
    pub fn set(&mut self, idx: usize, value: bool) {
        debug_assert!(
            idx < self.len,
            "bit index {idx} out of range len {}",
            self.len
        );
        let (word_idx, bit) = word_bit(idx);
        let mask = (1 as Word) << bit;
        if value {
            self.data[word_idx] |= mask;
        } else {
            self.data[word_idx] &= !mask;
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(
            idx < self.len,
            "bit index {idx} out of range len {}",
            self.len
        );
        let (word_idx, bit) = word_bit(idx);
        (self.data[word_idx] >> bit) & 1 != 0
    }

    #[inline]
    pub fn count_ones(&self) -> usize {
        if self.len == 0 {
            return 0;
        }
        let full_words = self.len / WORD_BITS;
        let mut total = 0usize;
        for idx in 0..full_words {
            total += self.data[idx].count_ones() as usize;
        }
        let remainder = self.len % WORD_BITS;
        if remainder != 0 {
            let mask = low_mask(remainder);
            total += (self.data[full_words] & mask).count_ones() as usize;
        }
        total
    }

    #[inline]
    pub fn find_first_zero_in_bucket(
        &self,
        bucket_start: usize,
        bucket_len: usize,
        start_slot: usize,
    ) -> Option<usize> {
        debug_assert!(bucket_start + bucket_len <= self.len);
        if bucket_len == 0 {
            return None;
        }
        let initial_len = bucket_len.saturating_sub(start_slot);
        if let Some(idx) = self.find_first_zero_range(bucket_start + start_slot, initial_len) {
            return Some(idx);
        }
        if start_slot != 0 {
            self.find_first_zero_range(bucket_start, start_slot)
        } else {
            None
        }
    }

    fn find_first_zero_range(&self, start: usize, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        debug_assert!(start + len <= self.len, "range exceeds length");

        let mut idx = start;
        let mut remaining_bits = len;
        let mut word_idx = idx / WORD_BITS;
        let bit_offset = idx % WORD_BITS;

        if bit_offset != 0 {
            let bits_in_word = (WORD_BITS - bit_offset).min(remaining_bits);
            let word = self.data[word_idx] >> bit_offset;
            let zero_mask = (!word) & low_mask(bits_in_word);
            if zero_mask != 0 {
                let tz = zero_mask.trailing_zeros() as usize;
                return Some(idx + tz);
            }
            idx += bits_in_word;
            remaining_bits -= bits_in_word;
            word_idx += 1;
        }

        let mut words_remaining = remaining_bits / WORD_BITS;
        while words_remaining >= LANES {
            let slice = &self.data[word_idx..word_idx + LANES];
            let arr: [Word; LANES] = slice
                .try_into()
                .expect("slice length should equal lane count");
            let chunk = SimdVec::from(arr);
            let inverted = chunk ^ SimdVec::splat(Word::MAX);
            let zeros = inverted.to_array();
            if zeros.iter().any(|&lane| lane != 0) {
                for (lane_idx, &lane) in zeros.iter().enumerate() {
                    if lane != 0 {
                        let tz = lane.trailing_zeros() as usize;
                        return Some(idx + lane_idx * WORD_BITS + tz);
                    }
                }
            }
            idx += WORD_BITS * LANES;
            remaining_bits -= WORD_BITS * LANES;
            word_idx += LANES;
            words_remaining -= LANES;
        }

        while words_remaining > 0 {
            let zeros = !self.data[word_idx];
            if zeros != 0 {
                let tz = zeros.trailing_zeros() as usize;
                return Some(idx + tz);
            }
            idx += WORD_BITS;
            remaining_bits -= WORD_BITS;
            word_idx += 1;
            words_remaining -= 1;
        }

        let tail_bits = remaining_bits;
        if tail_bits != 0 {
            let mask = low_mask(tail_bits);
            let zeros = (!self.data[word_idx]) & mask;
            if zeros != 0 {
                let tz = zeros.trailing_zeros() as usize;
                return Some(idx + tz);
            }
        }

        None
    }
}

#[inline]
fn words_for(bits: usize) -> usize {
    (bits + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn word_bit(idx: usize) -> (usize, usize) {
    (idx / WORD_BITS, idx % WORD_BITS)
}

#[inline]
fn low_mask(bits: usize) -> Word {
    if bits == 0 {
        0
    } else if bits >= WORD_BITS {
        Word::MAX
    } else {
        ((1 as Word) << bits) - 1
    }
}
