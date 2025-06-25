use core::num::NonZeroUsize;

use std::io::Write;

use integer_encoding::VarIntWriter;
use sha2::{Digest, Sha256};

use crate::types::{NonEmptyBz, U7, U63};

use super::{
    NodeKey, SHA256_HASH_LEN, SerializationError,
    info::{Drafted, Drafter, Hashed, Hasher, Saved, Saver},
};

type SavedLeafNode<K, V, VERSION, HASH, HAUX, NONCE> =
    LeafNode<Drafter<K, Hasher<VERSION, HASH, HAUX, Saver<NONCE>>>, V>;

#[derive(Debug, Clone)]
pub struct LeafNode<INFO, V = NonEmptyBz> {
    info: INFO,
    value: V,
}

impl<INFO, V> LeafNode<INFO, V> {
    pub const HEIGHT: U7 = U7::MIN;

    pub const SIZE: U63 = U63::ONE;

    const SUBTREE_HEIGHT_VARINT_ENCODED: [u8; 1] = [0];

    const SIZE_VARINT_ENCODED: [u8; 1] = [2];

    const SUBTREE_HEIGHT_AND_SIZE_VARINT_ENCODED: [u8; 2] = [
        Self::SUBTREE_HEIGHT_VARINT_ENCODED[0],
        Self::SIZE_VARINT_ENCODED[0],
    ];

    pub fn value(&self) -> &V {
        &self.value
    }
}

#[bon::bon]
impl LeafNode<Drafted> {
    #[builder]
    pub fn new(key: NonEmptyBz, value: NonEmptyBz) -> Self {
        Self {
            info: Drafted::new(key),
            value,
        }
    }
}

impl LeafNode<Drafted> {
    pub fn into_hashed(self, version: U63) -> LeafNode<Hashed> {
        let mut hasher = Sha256::new();

        hasher.update(Self::SUBTREE_HEIGHT_AND_SIZE_VARINT_ENCODED);

        // unwrap calls are safe because write on Sha256's hasher is infalliable

        hasher.write_varint(version.to_signed()).unwrap();

        hasher.write_varint(self.key().len()).unwrap();
        hasher.update(self.key().get());

        hasher.write_varint(SHA256_HASH_LEN).unwrap();
        hasher.update(Sha256::digest(self.value.get()));

        LeafNode {
            info: self.info.into_hashed(version, hasher.finalize().into(), ()),
            value: self.value,
        }
    }
}

impl<STAGE> LeafNode<Drafter<NonEmptyBz, STAGE>> {
    pub fn serialize<W>(&self, mut writer: W) -> Result<NonZeroUsize, SerializationError>
    where
        W: Write,
    {
        writer.write_all(&Self::SUBTREE_HEIGHT_AND_SIZE_VARINT_ENCODED)?;

        let key_bytes_len = super::serialize_bytes(self.key(), &mut writer)?;
        let value_bytes_len = super::serialize_bytes(self.value(), writer)?;

        Self::SUBTREE_HEIGHT_AND_SIZE_VARINT_ENCODED
            .len()
            .checked_add(key_bytes_len)
            .and_then(|len| len.checked_add(value_bytes_len))
            .and_then(NonZeroUsize::new)
            .ok_or(SerializationError::Overflow)
    }
}

impl<K, V, VERSION, HASH, HAUX> LeafNode<Drafter<K, Hasher<VERSION, HASH, HAUX>>, V> {
    pub fn into_saved<NONCE>(
        self,
        nonce: NONCE,
    ) -> SavedLeafNode<K, V, VERSION, HASH, HAUX, NONCE> {
        LeafNode {
            info: self.info.into_saved(nonce, ()),
            value: self.value,
        }
    }
}

impl<HAUX, SAUX> LeafNode<Saved<HAUX, SAUX>> {
    pub fn node_key(&self) -> NodeKey {
        NodeKey::builder()
            .version(*self.version())
            .nonce(*self.nonce())
            .build()
    }
}

impl<K, V, STAGE> LeafNode<Drafter<K, STAGE>, V> {
    pub fn key(&self) -> &K {
        self.info.key()
    }
}

impl<K, V, VERSION, HASH, HAUX, STATUS>
    LeafNode<Drafter<K, Hasher<VERSION, HASH, HAUX, STATUS>>, V>
{
    pub fn version(&self) -> &VERSION {
        self.info.version()
    }

    pub fn hash(&self) -> &HASH {
        self.info.hash()
    }
}

impl<K, V, VERSION, HASH, HAUX, NONCE, SAUX>
    LeafNode<Drafter<K, Hasher<VERSION, HASH, HAUX, Saver<NONCE, SAUX>>>, V>
{
    pub fn nonce(&self) -> &NONCE {
        self.info.nonce()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::node::info::Drafted;

    use super::LeafNode;

    mod utils {
        use bytes::Bytes;

        use crate::{
            node::{info::Drafted, leaf::LeafNode},
            types::NonEmptyBz,
        };

        pub fn draft_leaf_node<K, V>(key: K, value: V) -> LeafNode<Drafted>
        where
            K: AsRef<[u8]>,
            V: AsRef<[u8]>,
        {
            LeafNode::builder()
                .key(NonEmptyBz::new(Bytes::copy_from_slice(key.as_ref())).unwrap())
                .value(NonEmptyBz::new(Bytes::copy_from_slice(value.as_ref())).unwrap())
                .build()
        }
    }

    #[rstest]
    #[case(utils::draft_leaf_node("key", "value"), "0002036b65790576616c7565")]
    fn serialize_draft_leaf_node<E>(#[case] node: LeafNode<Drafted>, #[case] hex_serialized: E)
    where
        E: AsRef<[u8]>,
    {
        // Arrange
        let expected_serialized = const_hex::decode(hex_serialized).unwrap();

        // Act
        let mut serialized = vec![];
        let used = node.serialize(&mut serialized).unwrap();

        // Assert
        assert_eq!(expected_serialized, serialized);
        assert_eq!(expected_serialized.len(), used.get());
    }
}
