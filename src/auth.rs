use sha2::digest::Output;
use sha2::{Digest, Sha256};

type HashOf<H> = Output<H>;

const MIN_CAPACITY: usize = 16;

#[derive(Default, Clone)]
pub struct PoAuthTable<H: Clone + Default + Digest = Sha256>
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
    bucket_bits: usize, // b

    // How many following bits we use to choose where inside the bucket.
    in_bucket_bits: usize, // r

    // Derived: total index bits (b + r) == log2(capacity).
    index_bits: usize, // m = b + r

    // Number of buckets and bucket capacity in slots.
    num_buckets: usize,     // 1 << bucket_bits
    bucket_capacity: usize, // 1 << in_bucket_bits

    // shift to get index: idx = msw >> index_shift
    index_shift: u32,

    // mask after you have idx:
    //   bucket_id  = idx >> in_bucket_bits          // range [0, 2^b)
    //   in_bucket  = idx & in_bucket_mask           // low r bits
    in_bucket_mask: usize,
}

impl TableMeta {
    #[inline(always)]
    const fn from_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity >= MIN_CAPACITY);

        // m = log2(capacity)
        let m_u32: u32 = capacity.trailing_zeros(); // m >= 4 for capacity >= 16
        let m: usize = m_u32 as usize;

        // r = in-bucket bits, derived the same "ilog2" way from m
        let r: usize = m.ilog2() as usize; // r >= 2 for m>=4

        // b = bucket bits
        let b: usize = m - r;

        let num_buckets = 1usize << b;
        let bucket_capacity = 1usize << r;

        let w = usize::BITS;
        // For this design, we assume m <= usize::BITS (we only use one word of prefix)
        debug_assert!(m as u32 <= w);
        let index_shift = w - (m as u32); // top m bits → index

        // With capacity >= 16 we know 0 < r < w; no branches needed here.
        let in_bucket_mask = (1usize << r) - 1;

        Self {
            bucket_bits: b,
            in_bucket_bits: r,
            index_bits: m,
            num_buckets,
            bucket_capacity,
            index_shift,
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

        let bucket_id = idx >> self.in_bucket_bits; // no mask needed; range is [0, 2^b)
        let in_bucket = idx & self.in_bucket_mask;

        (bucket_id, in_bucket)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
struct BucketMeta<H: Clone + Default + Digest> {
    hash: HashOf<H>, // per-bucket commitment (eventual)
    start: usize,    // run start in `hashes`
    len: usize,      // live elements in this bucket
}

impl<H: Clone + Default + Digest> PoAuthTable<H> {
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

    /// Convenience: compute (bucket_id, in_bucket_offset) for a given hash.
    #[inline(always)]
    pub fn bucket_and_offset_for_hash(&self, h: &HashOf<H>) -> (usize, usize) {
        self.table_meta.bucket_and_offset_from_hash::<H>(h)
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

#[test]
fn test_new_with_capacity() {
    let table: PoAuthTable<Sha256> = PoAuthTable::with_capacity(100);
    assert_eq!(table.hashes.capacity(), 128);
    assert_eq!(
        table.table_meta,
        TableMeta {
            bucket_bits: 5,
            in_bucket_bits: 2,
            index_bits: 7,
            num_buckets: 32,
            bucket_capacity: 4,
            index_shift: 57,
            in_bucket_mask: 3
        }
    );
    let table: PoAuthTable<sha2::Sha512> = PoAuthTable::with_capacity(1000);
    assert_eq!(table.hashes.capacity(), 1024);
    assert_eq!(
        table.table_meta,
        TableMeta {
            bucket_bits: 7,
            in_bucket_bits: 3,
            index_bits: 10,
            num_buckets: 128,
            bucket_capacity: 8,
            index_shift: 54,
            in_bucket_mask: 7
        }
    );
}
