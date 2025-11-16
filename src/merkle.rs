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

impl<H: Digest + Clone + PartialOrd + Ord + PartialEq + Eq> PoaHT<H> {
    #[inline(always)]
    pub const fn root_hash(&self) -> &HashOf<H> {
        &self.root_hash
    }
}
