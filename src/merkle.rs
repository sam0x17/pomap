use sha2::digest::Output;
use sha2::{Digest, Sha256};

/// Digest output for the default [`Sha256`] hasher
pub type Sha256Hash = Output<Sha256>;

type HashOf<H> = Output<H>;

#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PoaHT<H: Digest = Sha256>
where
    HashOf<H>: Clone + PartialEq + Eq + PartialOrd + Ord + Default,
{
    root_hash: HashOf<H>,
    bucket_hashes: Vec<HashOf<H>>,
    hashes: Vec<HashOf<H>>,
}

impl<H: Digest> PoaHT<H> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            root_hash: HashOf::<H>::default(),
            bucket_hashes: Vec::new(),
            hashes: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            root_hash: HashOf::<H>::default(),
            bucket_hashes: Vec::with_capacity(num_buckets(capacity)),
            hashes: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub const fn root_hash(&self) -> &HashOf<H> {
        &self.root_hash
    }
}

#[inline(always)]
pub const fn num_buckets(n: usize) -> usize {
    n / (n).ilog2() as usize
}
