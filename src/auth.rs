use core::{array, mem, ptr};
use sha2::digest::Output;
use sha2::{Digest, Sha256};

type HashOf<H> = Output<H>;

const MIN_CAPACITY: usize = 16;
/// Number of **authenticated interior** levels above the virtual bottom.
/// 0 => only virtual leaf (no interior nodes). Tweak this to benchmark depth tradeoffs.
pub const LEVELS: usize = 2;

// ======================= Canonical MSW (LE: no-op) =======================

/// Load the *most-significant word* under **little-endian numeric order**:
/// i.e., the **last** native word of the digest. LE hosts: no swap; BE: one bswap.
#[inline(always)]
fn msw_usize_le_unaligned<H: Digest>(h: &HashOf<H>) -> usize
where
    HashOf<H>: AsRef<[u8]>,
{
    let bytes = h.as_ref();
    let n = mem::size_of::<usize>();
    debug_assert!(bytes.len() >= n);
    // SAFETY: we only read; alignment not guaranteed so use read_unaligned.
    let p = unsafe { bytes.as_ptr().add(bytes.len() - n) as *const usize };
    let raw = unsafe { ptr::read_unaligned(p) };
    usize::from_le(raw) // LE: no-op; BE: one bswap
}

// ======================= Full path & small helpers =======================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FullPath<const L: usize> {
    /// Node id per authenticated level, top (0) → bottom (L-1).
    pub level_ids: [usize; L],
    /// Virtual leaf slot inside the bottom bucket (low `r` bits).
    pub in_bucket: usize,
    /// Flattened id of the bottom interior node (concatenation of all interior bits).
    pub global_id: usize,
}

#[inline(always)]
const fn lsb_mask(bits: u32) -> usize {
    if bits == 0 {
        0
    } else if bits == usize::BITS {
        usize::MAX
    } else {
        (1usize << bits) - 1
    }
}

// ======================= Table meta =======================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableMeta<const L: usize> {
    // Bit budget
    pub index_bits: u32,      // m = ilog2(capacity)
    pub in_bucket_bits: u32,  // r = ilog2(m)
    pub level_bits: [u32; L], // interior bits, top..bottom; sum(level_bits)+r == m

    // Sizes / mapping
    pub nodes_per_level: [usize; L], // nodes at each level (global), cumulative
    pub span_slots: [usize; L],      // leaf slots covered by one node at each level
    pub bucket_capacity: usize,      // 1 << r
    pub num_bottom_nodes: usize,     // == 1 << sum(level_bits)

    // Hot-path shifts/masks
    pub index_shift: u32,        // idx = msw >> index_shift
    pub in_bucket_mask: usize,   // (1<<r)-1
    pub level_masks: [usize; L], // (1<<level_bits[i]) - 1 (or 0)
    pub level_shifts: [u32; L],  // sum of bits below level i (bottomwards), precomputed
}

impl<const L: usize> TableMeta<L> {
    /// Derive layout from `capacity` only: m=ilog2(cap), r=ilog2(m), remaining bits split across levels by iterated ilog2.
    #[inline(always)]
    pub(crate) fn from_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two() && capacity >= MIN_CAPACITY);

        // m = ilog2(capacity)
        let m: u32 = capacity.trailing_zeros();
        // r = ilog2(m) (virtual leaf bits)
        let r: u32 = if m == 0 {
            0
        } else {
            (m as usize).ilog2() as u32
        };

        // Distribute remaining bits across L interior levels by iterated ilog2.
        let mut rem = (m - r) as usize;
        let mut level_bits = [0u32; L];
        if L > 0 {
            for i in 0..L {
                let li = if i + 1 < L {
                    if rem == 0 { 0 } else { rem.ilog2() as u32 }
                } else {
                    rem as u32
                };
                level_bits[i] = li;
                rem = rem.saturating_sub(li as usize);
            }
        }
        debug_assert_eq!(rem, 0);

        // Derived sizes/masks
        let mut nodes_per_level = [0usize; L];
        let mut level_masks = [0usize; L];
        let mut span_slots = [0usize; L];

        // cumulative nodes at each level (top..bottom)
        let mut cum = 0usize;
        for i in 0..L {
            cum += level_bits[i] as usize;
            nodes_per_level[i] = 1usize << cum;
            level_masks[i] = lsb_mask(level_bits[i]);
        }
        let num_bottom_nodes = if L == 0 { 1 } else { nodes_per_level[L - 1] };

        let bucket_capacity = 1usize << r;

        // span of one node at each level, measured in leaf slots
        // bottom interior level (if any) has span == bucket_capacity.
        let mut below_sum = 0usize;
        for i in (0..L).rev() {
            span_slots[i] = bucket_capacity << below_sum;
            below_sum += level_bits[i] as usize;
        }

        let w = usize::BITS;
        let index_shift = w - m;
        let in_bucket_mask = lsb_mask(r);

        // Precompute shifts to slice out each level’s id from the authenticated prefix.
        // For level i (top..bottom), shift == sum(level_bits[j]) for j in (i+1..L).
        let mut level_shifts = [0u32; L];
        let mut sum_below = 0usize;
        for i in (0..L).rev() {
            level_shifts[i] = sum_below as u32;
            sum_below += level_bits[i] as usize;
        }

        Self {
            index_bits: m,
            in_bucket_bits: r,
            level_bits,
            nodes_per_level,
            span_slots,
            bucket_capacity,
            num_bottom_nodes,
            index_shift,
            in_bucket_mask,
            level_masks,
            level_shifts,
        }
    }

    /// Absolute start (in `hashes`) of a node by (level, id). O(1).
    #[inline(always)]
    pub(crate) fn node_start(&self, level: usize, id: usize) -> usize {
        debug_assert!(level < L);
        id * self.span_slots[level]
    }

    /// **Branchless** hot-path full path:
    /// returns all level IDs (top..bottom), the virtual leaf slot, and the flattened bottom id.
    #[inline(always)]
    pub(crate) fn full_path_from_hash<H: Digest>(&self, h: &HashOf<H>) -> FullPath<L>
    where
        HashOf<H>: AsRef<[u8]>,
    {
        // 1) Canonical MSW (LE no-op), then top-m index
        let msw = msw_usize_le_unaligned::<H>(h);
        let idx = (msw >> self.index_shift) as usize;

        // 2) Split: low-r = bin; high-(m-r) = authenticated prefix
        let in_bucket = idx & self.in_bucket_mask;
        let auth = idx >> self.in_bucket_bits; // concatenation of all interior bits

        // 3) Slice each level's id from `auth` without branches
        let mut ids = [0usize; L];
        for i in 0..L {
            ids[i] = (auth >> self.level_shifts[i]) & self.level_masks[i];
        }

        // 4) Flattened bottom id is exactly the authenticated prefix
        FullPath {
            level_ids: ids,
            in_bucket,
            global_id: auth,
        }
    }

    // ---------- Bin jump helpers (no per-bin counts) ----------

    /// Top-m index from digest.
    #[inline(always)]
    pub(crate) fn index_from_hash<H: Digest>(&self, h: &HashOf<H>) -> usize
    where
        HashOf<H>: AsRef<[u8]>,
    {
        let msw = msw_usize_le_unaligned::<H>(h);
        (msw >> self.index_shift) as usize
    }

    /// (global bottom id, in-bucket bin) from the top-m index.
    #[inline(always)]
    pub(crate) fn gid_and_bin_from_index(&self, idx: usize) -> (usize, usize) {
        let bin = idx & self.in_bucket_mask; // low r
        let gid = idx >> self.in_bucket_bits; // high (m - r)
        (gid, bin)
    }

    /// **One-shot**: absolute bin index and enclosing bucket bounds.
    /// Returns `(bin_index, bucket_lo, bucket_hi)`.
    #[inline(always)]
    pub(crate) fn bin_index_from_hash<H: Digest>(&self, h: &HashOf<H>) -> (usize, usize, usize)
    where
        HashOf<H>: AsRef<[u8]>,
    {
        let msw = msw_usize_le_unaligned::<H>(h);
        let idx = (msw >> self.index_shift) as usize;

        let bin = idx & self.in_bucket_mask;
        let gid = idx >> self.in_bucket_bits;

        let bucket_lo = gid * self.bucket_capacity;
        (bucket_lo + bin, bucket_lo, bucket_lo + self.bucket_capacity)
    }

    /// Bucket bounds (bottom interior node span) for `h` (if you ever need just the bounds).
    #[inline(always)]
    pub(crate) fn bucket_bounds_for_hash<H: Digest>(&self, h: &HashOf<H>) -> (usize, usize)
    where
        HashOf<H>: AsRef<[u8]>,
    {
        let idx = self.index_from_hash::<H>(h);
        let gid = idx >> self.in_bucket_bits;
        let lo = gid * self.bucket_capacity;
        (lo, lo + self.bucket_capacity)
    }
}

// ======================= Unified interior meta =======================

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct LevelMeta<H: Clone + Default + Digest> {
    /// Authenticated commitment for this node.
    pub hash: HashOf<H>,
    /// Live items under this node (maintained incrementally as you insert/delete).
    pub len: usize,
}

// ======================= Main table =======================

#[derive(Clone)]
pub struct PoAuthTable<H: Clone + Default + Digest = Sha256, const L: usize = LEVELS>
where
    HashOf<H>: Clone + PartialEq + Eq + PartialOrd + Ord + Default,
{
    root_hash: HashOf<H>,
    meta: TableMeta<L>,
    /// Per-level arrays of interior nodes (top..bottom). Lengths = nodes_per_level[i].
    level: [Vec<LevelMeta<H>>; L],
    /// Flat leaf array; in your full impl store (hash,key,val) sorted by LE-canonical (hash,key).
    hashes: Vec<HashOf<H>>,
}

impl<H: Clone + Default + Digest, const L: usize> PoAuthTable<H, L>
where
    HashOf<H>: Clone + PartialEq + Eq + PartialOrd + Ord + Default,
{
    #[inline(always)]
    pub const fn root_hash(&self) -> &HashOf<H> {
        &self.root_hash
    }

    #[inline(always)]
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(MIN_CAPACITY);
        let meta = TableMeta::<L>::from_capacity(cap);
        let default_hash = HashOf::<H>::default();

        // Build interior levels; node i at level ℓ covers span_slots[ℓ] leaf slots.
        let level: [Vec<LevelMeta<H>>; L] = array::from_fn(|ell| {
            vec![
                LevelMeta {
                    hash: default_hash.clone(),
                    len: 0
                };
                meta.nodes_per_level[ell]
            ]
        });

        Self {
            root_hash: default_hash,
            meta,
            level,
            hashes: Vec::with_capacity(cap),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::array;
    use sha2::Sha256;

    fn hash_with_msw(msw: usize) -> HashOf<Sha256> {
        let mut hash = HashOf::<Sha256>::default();
        let bytes = hash.as_mut_slice();
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
        }
        let word_bytes = msw.to_le_bytes();
        let start = bytes.len() - word_bytes.len();
        bytes[start..].copy_from_slice(&word_bytes);
        hash
    }

    #[test]
    fn msw_function_extracts_inserted_value() {
        let inserted = usize::MAX / 3;
        let hash = hash_with_msw(inserted);
        assert_eq!(msw_usize_le_unaligned::<Sha256>(&hash), inserted);
    }

    #[test]
    fn table_meta_layout_consistent_for_common_capacities() {
        let capacities = [
            MIN_CAPACITY,
            MIN_CAPACITY * 4,
            MIN_CAPACITY * 16,
            MIN_CAPACITY * 64,
        ];
        for &cap in capacities.iter() {
            let meta = TableMeta::<LEVELS>::from_capacity(cap);
            assert_eq!(meta.index_bits, cap.trailing_zeros());
            assert_eq!(meta.bucket_capacity, 1usize << meta.in_bucket_bits);
            let sum_level_bits: u32 = meta.level_bits.iter().sum();
            assert_eq!(sum_level_bits + meta.in_bucket_bits, meta.index_bits);
            let expected_bottom = meta.nodes_per_level.last().copied().unwrap_or(1);
            assert_eq!(meta.num_bottom_nodes, expected_bottom);

            let mut cumulative = 0usize;
            for i in 0..LEVELS {
                cumulative += meta.level_bits[i] as usize;
                assert_eq!(meta.nodes_per_level[i], 1usize << cumulative);
                assert_eq!(meta.level_masks[i], lsb_mask(meta.level_bits[i]));
            }

            let mut below = 0usize;
            for i in (0..LEVELS).rev() {
                assert_eq!(meta.span_slots[i], meta.bucket_capacity << below);
                below += meta.level_bits[i] as usize;
            }

            let mut shift_sum = 0usize;
            for i in (0..LEVELS).rev() {
                assert_eq!(meta.level_shifts[i], shift_sum as u32);
                shift_sum += meta.level_bits[i] as usize;
            }
        }
    }

    #[test]
    fn node_start_matches_span_slots() {
        if LEVELS == 0 {
            return;
        }
        let meta = TableMeta::<LEVELS>::from_capacity(MIN_CAPACITY * 4);
        for level in 0..LEVELS {
            let span = meta.span_slots[level];
            assert_eq!(meta.node_start(level, 0), 0);
            if meta.nodes_per_level[level] > 1 {
                assert_eq!(meta.node_start(level, 1), span);
            }
            let last = meta.nodes_per_level[level].saturating_sub(1);
            assert_eq!(meta.node_start(level, last), last * span);
        }
    }

    #[test]
    fn full_path_and_bucket_helpers_are_coherent() {
        const CAPACITY: usize = MIN_CAPACITY * 4;
        let meta = TableMeta::<LEVELS>::from_capacity(CAPACITY);

        let level_ids: [usize; LEVELS] = array::from_fn(|i| {
            let mask = meta.level_masks[i];
            if mask == 0 { 0 } else { mask.min(i + 1) }
        });

        let mut auth = 0usize;
        for i in 0..LEVELS {
            auth |= level_ids[i] << meta.level_shifts[i];
        }

        let desired_bin = if meta.in_bucket_mask == 0 {
            0
        } else {
            meta.in_bucket_mask.min(3)
        };
        let idx = (auth << meta.in_bucket_bits) | desired_bin;
        let msw = idx << meta.index_shift;
        let hash = hash_with_msw(msw);

        assert_eq!(meta.index_from_hash::<Sha256>(&hash), idx);

        let path = meta.full_path_from_hash::<Sha256>(&hash);
        assert_eq!(path.global_id, auth);
        assert_eq!(path.in_bucket, desired_bin);
        assert_eq!(path.level_ids, level_ids);

        let (gid, bin) = meta.gid_and_bin_from_index(idx);
        assert_eq!((gid, bin), (auth, desired_bin));

        let (bin_index, bucket_lo, bucket_hi) = meta.bin_index_from_hash::<Sha256>(&hash);
        assert_eq!(bin_index, gid * meta.bucket_capacity + bin);
        assert_eq!(bucket_lo, gid * meta.bucket_capacity);
        assert_eq!(bucket_hi, bucket_lo + meta.bucket_capacity);
        assert_eq!(
            meta.bucket_bounds_for_hash::<Sha256>(&hash),
            (bucket_lo, bucket_hi)
        );
    }

    #[test]
    fn poauth_table_new_uses_min_capacity() {
        let table = PoAuthTable::<Sha256, LEVELS>::new();
        assert_eq!(table.hashes.len(), 0);
        assert_eq!(table.hashes.capacity(), MIN_CAPACITY);
        assert_eq!(table.meta, TableMeta::<LEVELS>::from_capacity(MIN_CAPACITY));
        assert_eq!(table.root_hash(), &HashOf::<Sha256>::default());

        for (level_idx, nodes) in table.level.iter().enumerate() {
            assert_eq!(nodes.len(), table.meta.nodes_per_level[level_idx]);
            assert!(
                nodes
                    .iter()
                    .all(|node| node.len == 0 && node.hash == HashOf::<Sha256>::default())
            );
        }
    }

    #[test]
    fn with_capacity_rounds_up_and_initializes_levels() {
        let requested = MIN_CAPACITY * 3 + 5;
        let expected_cap = requested.next_power_of_two().max(MIN_CAPACITY);
        let table = PoAuthTable::<Sha256, LEVELS>::with_capacity(requested);
        assert_eq!(table.hashes.len(), 0);
        assert_eq!(table.hashes.capacity(), expected_cap);
        assert_eq!(table.meta, TableMeta::<LEVELS>::from_capacity(expected_cap));

        for (level_idx, nodes) in table.level.iter().enumerate() {
            assert_eq!(nodes.len(), table.meta.nodes_per_level[level_idx]);
            assert!(
                nodes
                    .iter()
                    .all(|node| node.len == 0 && node.hash == HashOf::<Sha256>::default())
            );
        }
    }
}
