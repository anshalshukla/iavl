mod error;

pub use self::error::NodeDbError;

use bytes::{BufMut, BytesMut};

use crate::{
    kvstore::{KVStore, MutKVStore},
    types::{NonEmptyBz, U63},
};

use super::{DeserializedNode, NodeKey, kind::SavedNode};

use self::error::Result;

const NODE_DB_KEY_LEN: usize = size_of::<u8>() + size_of::<u64>() + size_of::<u32>();

const NODE_DB_KEY_PREFIX: u8 = b's';

#[derive(Debug, Clone)]
pub struct NodeDb<DB> {
    db: DB,
    first_version: U63,
    latest_version: U63,
}

impl<DB> NodeDb<DB>
where
    DB: KVStore,
{
    pub fn fetch_one_node(&self, nk: &NodeKey) -> Result<Option<DeserializedNode>> {
        let key = NonEmptyBz::from_owned_array(make_node_db_key(NODE_DB_KEY_PREFIX, nk));
        self.db
            .get(&key)
            .map_err(From::from)
            .map_err(NodeDbError::Store)?
            .as_ref()
            .map(NonEmptyBz::get)
            .map(AsRef::<[u8]>::as_ref)
            .map(DeserializedNode::deserialize)
            .transpose()
            .map_err(From::from)
    }
}

impl<DB> NodeDb<DB>
where
    DB: MutKVStore,
{
    /// Overwrites and returns true if another node existed for the same [`NodeKey`].
    pub fn save_overwriting_one_node(&self, node: &SavedNode) -> Result<bool> {
        let serialized = {
            let mut serialized = BytesMut::new().writer();

            node.serialize(&mut serialized)?;

            NonEmptyBz::new(serialized.into_inner().freeze())
                .ok_or(NodeDbError::Other("serialized must be non-empty".into()))?
        };

        let node_db_key =
            NonEmptyBz::from_owned_array(make_node_db_key(NODE_DB_KEY_PREFIX, &node.node_key()));

        self.db
            .insert(&node_db_key, &serialized)
            .map_err(From::from)
            .map_err(NodeDbError::Store)
    }
}

impl<DB> NodeDb<DB>
where
    DB: MutKVStore + KVStore,
{
    pub fn save_non_overwririting_one_node(
        &self,
        node: &SavedNode,
    ) -> Result<Option<DeserializedNode>> {
        let nk = node.node_key();
        if let existing @ Some(_) = self.fetch_one_node(&nk)? {
            return Ok(existing);
        }

        assert!(
            !self.save_overwriting_one_node(node)?,
            "key conflict must not occur"
        );

        Ok(None)
    }
}

const fn make_node_db_key(prefix: u8, nk: &NodeKey) -> [u8; NODE_DB_KEY_LEN] {
    let mut key = [0; NODE_DB_KEY_LEN];
    key[0] = prefix;

    let version_be_bytes = nk.version().get().to_be_bytes();
    let mut i = 0;
    while i < size_of::<u64>() {
        key[i + 1] = version_be_bytes[i];
        i += 1;
    }

    let nonce_be_bytes = nk.nonce().get().to_be_bytes();
    let mut i = 0;
    while i < size_of::<u32>() {
        key[i + 1 + size_of::<u64>()] = nonce_be_bytes[i];
        i += 1;
    }

    key
}
