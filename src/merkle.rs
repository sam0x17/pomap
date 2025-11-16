use sha2::digest::Output;
use sha2::{Digest, Sha256};

/// Digest output for the default [`Sha256`] hasher
pub type Sha256Hash = Output<Sha256>;

type HashOf<H> = Output<H>;

#[derive(Default, Clone)]
pub struct PoaHT<H: Digest = Sha256>
where
    HashOf<H>: Clone + PartialEq + Eq + PartialOrd + Ord + Default,
{
    root_hash: HashOf<H>,
    bucket_metas: Vec<HashOf<H>>,
    hashes: Vec<HashOf<H>>,
    prefix_bits: u8,
    bucket_prefix_bits: u8,
    prefix_mask: usize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
struct BucketMeta<H: Digest> {
    hash: HashOf<H>,
    len: u32,
}

impl<H: Digest> PoaHT<H> {
    #[inline(always)]
    pub const fn root_hash(&self) -> &HashOf<H> {
        &self.root_hash
    }
}

#[inline(always)]
pub const fn num_buckets(n: usize) -> usize {
    n / (n).ilog2() as usize
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
