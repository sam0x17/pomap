use sha2::digest::Output;
use sha2::{Digest, Sha256};

/// Digest output for the default [`Sha256`] hasher
pub type Sha256Hash = Output<Sha256>;

type HashOf<H> = Output<H>;

const MIN_CAPACITY: usize = 16;

#[derive(Default, Clone)]
pub struct PoAHT<H: Clone + Default + Digest = Sha256>
where
    HashOf<H>: Clone + PartialEq + Eq + PartialOrd + Ord + Default,
{
    root_hash: HashOf<H>,
    table_meta: TableMeta,
    bucket_metas: Vec<BucketMeta<H>>,
    hashes: Vec<HashOf<H>>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
struct TableMeta {
    // How many top bits of the hash we use to choose the bucket.
    bucket_bits: u8, // b

    // How many following bits we use to choose where inside the bucket.
    in_bucket_bits: u8, // r

    // Derived: total index bits (b + r) == log2(capacity).
    index_bits: u8, // m = b + r

    // Number of buckets and bucket capacity in slots.
    num_buckets: usize,     // 1 << bucket_bits
    bucket_capacity: usize, // 1 << in_bucket_bits

    // shift to get index: idx = msw >> index_shift
    index_shift: u32,

    // masks after you have idx:
    //   bucket_id  = (idx >> in_bucket_bits) & bucket_mask
    //   in_bucket  = idx & in_bucket_mask
    bucket_mask: usize,
    in_bucket_mask: usize,
}

impl TableMeta {
    #[inline(always)]
    const fn from_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity >= MIN_CAPACITY);

        // m = log2(capacity)
        let m: u32 = capacity.trailing_zeros(); // m >= 4

        // r = in-bucket bits, derived the same "ilog2" way from m
        let r: u32 = (m as usize).ilog2() as u32; // r >= 2 for m>=4

        // b = bucket bits
        let b: u32 = m - r;

        let num_buckets = 1usize << b;
        let bucket_capacity = 1usize << r;

        let w = usize::BITS;
        let index_shift = w - m; // top m bits → index

        // With capacity >= 16 we know 0 < b,r < w; no branches needed here.
        let bucket_mask = (1usize << b) - 1;
        let in_bucket_mask = (1usize << r) - 1;

        Self {
            bucket_bits: b as u8,
            in_bucket_bits: r as u8,
            index_bits: m as u8,
            num_buckets,
            bucket_capacity,
            index_shift,
            bucket_mask,
            in_bucket_mask,
        }
    }

    /// Map a hash to (bucket_id, in_bucket_offset).
    #[inline(always)]
    fn bucket_and_offset_from_hash<H: Digest>(&self, h: &HashOf<H>) -> (usize, usize)
    where
        HashOf<H>: AsRef<[u8]>,
    {
        let msw = msw_usize_le_unaligned::<H>(h);
        let idx = msw >> self.index_shift; // top m bits as index

        let r = self.in_bucket_bits as u32;
        let bucket_id = (idx >> r) & self.bucket_mask;
        let in_bucket = idx & self.in_bucket_mask;

        (bucket_id, in_bucket)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
struct BucketMeta<H: Clone + Default + Digest> {
    hash: HashOf<H>,
    start: usize,
    len: usize,
}

impl<H: Clone + Default + Digest> PoAHT<H> {
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
        let table_meta = TableMeta::from_capacity(cap);
        let default_hash = HashOf::<H>::default();
        let bucket_metas = (0..table_meta.num_buckets)
            .map(|i| {
                let start = i * table_meta.bucket_capacity;
                BucketMeta {
                    hash: default_hash.clone(),
                    start,
                    len: 0,
                }
            })
            .collect();

        Self {
            root_hash: default_hash,
            table_meta,
            bucket_metas,
            hashes: Vec::with_capacity(cap),
        }
    }
}

/// Load the *most-significant word* under **little-endian numeric order**:
/// i.e., the **last** native word of the digest. LE hosts: no swap; BE: one bswap.
#[inline(always)]
fn msw_usize_le_unaligned<H: Digest>(h: &HashOf<H>) -> usize
where
    HashOf<H>: AsRef<[u8]>,
{
    let bytes = h.as_ref();
    let n = core::mem::size_of::<usize>();
    debug_assert!(bytes.len() >= n);
    let p = unsafe { bytes.as_ptr().add(bytes.len() - n) as *const usize };
    let raw = unsafe { core::ptr::read_unaligned(p) };
    usize::from_le(raw) // LE = no-op; BE = byteswap
}
