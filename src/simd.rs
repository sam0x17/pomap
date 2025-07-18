//! simd_find.rs
//!
//! Branch‑free lane‑index search for `wide::u64x4`.
//! Returns `Some(idx)` if any lane equals `target`, else `None`.
//
//  Fast paths
//  ──────────
//  * x86_64+AVX2 → vpcmpeqq  → vpmovmskb → tzcnt   (safe_arch helpers)
//  * AArch64+NEON → vceqq_u64 → vshrq_n_u64 → vgetq_lane → tzcnt
//  * Portable     → cmp_eq → shift‑pack → tzcnt
//
//  The public API is always `find_match_index(vec, target)` and is safe.

#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx2"),
    allow(unsafe_code)
)]
#![cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(unsafe_code)
)]

use wide::u64x4;

/// Public entry point: picks the best implementation at compile time.
#[inline(always)]
pub fn find_match_index(vec: u64x4, target: u64) -> Option<usize> {
    // ---- x86 + AVX2 -------------------------------------------------------
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return avx2::find_match_index(vec, target);
    }

    // ---- AArch64 + NEON ---------------------------------------------------
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        neon::find_match_index(vec, target)
    }

    // ---- Portable fallback -----------------------------------------------
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        return portable::find_match_index(vec, target);
    }
}

/*─────────────────────────  AVX2 PATH  ─────────────────────────*/
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use super::*;
    use safe_arch::{m256i, movemask_i8_m256i};

    /// Fast AVX2 implementation – **entirely safe** (safe_arch wraps intrinsics).
    #[inline(always)]
    pub fn find_match_index(vec: u64x4, target: u64) -> Option<usize> {
        // 1) SIMD compare
        let cmp = vec.cmp_eq(u64x4::splat(target));

        // 2) movemask : top bit of every byte → 32‑bit mask
        let mask32 = movemask_i8_m256i(m256i::from(cmp));

        // 3) keep bits 7,15,23,31  (top bit of each 64‑bit lane)
        let mask4: u8 = (((mask32 >> 7) & 1)
            | ((mask32 >> 15) & 2)
            | ((mask32 >> 23) & 4)
            | ((mask32 >> 31) & 8)) as u8;

        if mask4 == 0 {
            None
        } else {
            Some(mask4.trailing_zeros() as usize)
        }
    }
}

/*─────────────────────────  NEON PATH  ─────────────────────────*/
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod neon {
    use super::*;
    use core::arch::aarch64::{uint64x2_t, vceqq_u64, vdupq_n_u64, vgetq_lane_u64, vshrq_n_u64};

    /// Fast NEON implementation.
    ///
    /// # Safety
    /// Requires the `neon` target feature (guaranteed by the `cfg` gate).
    #[inline(always)]
    pub unsafe fn find_match_index(vec: u64x4, target: u64) -> Option<usize> {
        // Split 256‑bit vector into two 128‑bit halves
        let (lo, hi): (uint64x2_t, uint64x2_t) =
            unsafe { core::mem::transmute::<u64x4, (uint64x2_t, uint64x2_t)>(vec) };

        let tgt = unsafe { vdupq_n_u64(target) };

        // 1) lane‑wise compare
        let cmp_lo = unsafe { vceqq_u64(lo, tgt) };
        let cmp_hi = unsafe { vceqq_u64(hi, tgt) };

        // 2) keep the MSB of each lane
        let top_lo = unsafe { vshrq_n_u64::<63>(cmp_lo) };
        let top_hi = unsafe { vshrq_n_u64::<63>(cmp_hi) };

        // 3) pack four bits into a byte
        let mask: u8 = (unsafe { vgetq_lane_u64::<0>(top_lo) } as u8)
            | ((unsafe { vgetq_lane_u64::<1>(top_lo) } as u8) << 1)
            | ((unsafe { vgetq_lane_u64::<0>(top_hi) } as u8) << 2)
            | ((unsafe { vgetq_lane_u64::<1>(top_hi) } as u8) << 3);

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

/*─────────────────────  PORTABLE FALLBACK  ───────────────────*/
pub mod portable {
    use super::*;

    #[inline(always)]
    pub fn find_match_index(vec: u64x4, target: u64) -> Option<usize> {
        let cmp = vec.cmp_eq(u64x4::splat(target)); // lane = !0 if equal
        let hi: u64x4 = cmp >> 63u64; // keep top bit

        let lanes = hi.to_array(); // [u64; 4] (0/1)
        let mask: u8 = (lanes[0] as u8)
            | ((lanes[1] as u8) << 1)
            | ((lanes[2] as u8) << 2)
            | ((lanes[3] as u8) << 3);

        if mask == 0 {
            None
        } else {
            Some(mask.trailing_zeros() as usize)
        }
    }
}

/*──────────────────────────  TESTS  ──────────────────────────*/
#[cfg(test)]
mod tests {
    use super::*;
    use wide::u64x4;

    #[test]
    fn all_hits_and_misses() {
        let v = u64x4::from([1, 2, 3, 4]);
        assert_eq!(find_match_index(v, 1), Some(0));
        assert_eq!(find_match_index(v, 2), Some(1));
        assert_eq!(find_match_index(v, 3), Some(2));
        assert_eq!(find_match_index(v, 4), Some(3));
        assert_eq!(find_match_index(v, 9), None);
    }
}
