pub mod db;
pub mod info;

mod error;
mod inner;
mod kind;
mod leaf;

pub use self::{
    error::{DeserializationError, NodeError, SerializationError},
    inner::{Child, InnerNode, InnerNodeError},
    kind::{DeserializedNode, DraftedNode, HashedNode, SavedNode},
    leaf::LeafNode,
};

use std::{
    borrow::Cow,
    io::Write,
    num::NonZeroUsize,
    sync::{Arc, RwLock},
};

use bon::Builder;
use integer_encoding::VarIntWriter;

use crate::{
    kvstore::KVStore,
    types::{NonEmptyBz, U7, U31, U63},
};

use self::{db::NodeDb, error::Result};

pub const SHA256_HASH_LEN: usize = 32;

pub type NodeHash<const N: usize = SHA256_HASH_LEN> = [u8; N];

pub type NodeHashPair = (NodeHash, NodeHash);

pub type NodeKeyPair = (NodeKey, NodeKey);

pub type ArlockNode = Arc<RwLock<Node>>;

/// NodeKey represents a key of node in the DB
#[derive(Debug, Clone, PartialEq, Eq, Builder)]
pub struct NodeKey<V = U63, N = U31> {
    /// version of the IAVL that this node was first added in
    version: V,

    /// local nonce for the same version   
    nonce: N,
}

#[derive(Debug)]
pub enum Node {
    Drafted(DraftedNode),
    Hashed(HashedNode),
    Saved(SavedNode),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BalanceFactor {
    CriticalLeft,
    HeavyLeft,
    Par,
    HeavyRight,
    CriticalRight,
}

impl<V, N> NodeKey<V, N> {
    pub const fn version(&self) -> &V {
        &self.version
    }

    pub const fn nonce(&self) -> &N {
        &self.nonce
    }
}

impl NodeKey {
    pub fn serialize<W>(&self, mut writer: W) -> Result<NonZeroUsize, SerializationError>
    where
        W: Write,
    {
        writer
            .write_varint(self.version().to_signed())
            .and_then(|vlen| {
                writer
                    .write_varint(self.nonce().to_signed())
                    .map(|nlen| vlen + nlen) // direct addition won't overflow
            })
            .map(NonZeroUsize::new)
            .transpose()
            .unwrap() // unwrap is safe here as vlen + nlen > 0
            .map_err(From::from)
    }
}

impl Node {
    pub fn key(&self) -> &NonEmptyBz {
        match self {
            Self::Drafted(drafted_node) => drafted_node.key(),
            Self::Hashed(hashed_node) => hashed_node.key(),
            Self::Saved(saved_node) => saved_node.key(),
        }
    }

    pub fn height(&self) -> U7 {
        match self {
            Self::Drafted(drafted_node) => drafted_node.height(),
            Self::Hashed(hashed_node) => hashed_node.height(),
            Self::Saved(saved_node) => saved_node.height(),
        }
    }

    pub fn size(&self) -> U63 {
        match self {
            Self::Drafted(drafted_node) => drafted_node.size(),
            Self::Hashed(hashed_node) => hashed_node.size(),
            Self::Saved(saved_node) => saved_node.size(),
        }
    }

    pub fn hash(&self) -> Option<&NodeHash> {
        match self {
            Self::Hashed(hashed_node) => Some(hashed_node.hash()),
            Self::Saved(saved_node) => Some(saved_node.hash()),
            _ => None,
        }
    }

    pub fn node_key(&self) -> Option<NodeKey> {
        match self {
            Self::Saved(node) => Some(node.node_key()),
            _ => None,
        }
    }

    pub fn as_drafted(&self) -> Option<&DraftedNode> {
        match self {
            Self::Drafted(drafted_node) => Some(drafted_node),
            _ => None,
        }
    }

    pub fn as_hashed(&self) -> Option<&HashedNode> {
        match self {
            Self::Hashed(hashed_node) => Some(hashed_node),
            _ => None,
        }
    }

    pub fn as_saved(&self) -> Option<&SavedNode> {
        match self {
            Self::Saved(saved_node) => Some(saved_node),
            _ => None,
        }
    }

    pub fn left(&self) -> Option<&Child> {
        match self {
            Self::Drafted(drafted_node) => drafted_node.left(),
            Self::Hashed(hashed_node) => hashed_node.left(),
            Self::Saved(saved_node) => saved_node.left(),
        }
    }

    pub fn right(&self) -> Option<&Child> {
        match self {
            Self::Drafted(drafted_node) => drafted_node.right(),
            Self::Hashed(hashed_node) => hashed_node.right(),
            Self::Saved(saved_node) => saved_node.right(),
        }
    }

    pub fn left_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Drafted(drafted_node) => drafted_node.left_mut(),
            Self::Hashed(hashed_node) => hashed_node.left_mut(),
            Self::Saved(saved_node) => saved_node.left_mut(),
        }
    }

    pub fn right_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Drafted(drafted_node) => drafted_node.right_mut(),
            Self::Hashed(hashed_node) => hashed_node.right_mut(),
            Self::Saved(saved_node) => saved_node.right_mut(),
        }
    }

    pub fn serialize<W>(&self, writer: W) -> Result<Option<NonZeroUsize>, SerializationError>
    where
        W: Write,
    {
        match self {
            Self::Drafted(DraftedNode::Leaf(node)) => node.serialize(writer).map(Some),
            Self::Hashed(HashedNode::Leaf(node)) => node.serialize(writer).map(Some),
            Self::Saved(node) => node.serialize(writer).map(Some),
            _ => Ok(None),
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            Self::Drafted(DraftedNode::Leaf(_))
                | Self::Hashed(HashedNode::Leaf(_))
                | Self::Saved(SavedNode::Leaf(_)),
        )
    }

    pub fn value(&self) -> Option<&NonEmptyBz> {
        match self {
            Self::Drafted(DraftedNode::Leaf(leaf)) => Some(leaf.value()),
            Self::Hashed(HashedNode::Leaf(leaf)) => Some(leaf.value()),
            Self::Saved(SavedNode::Leaf(leaf)) => Some(leaf.value()),
            _ => None,
        }
    }

    pub fn compute_balance_factor<DB>(
        &self,
        ndb: &NodeDb<DB>,
    ) -> Result<BalanceFactor, InnerNodeError>
    where
        DB: KVStore,
    {
        match self {
            Node::Drafted(drafted_node) => drafted_node.compute_balance_factor(ndb),
            Node::Hashed(hashed_node) => hashed_node.compute_balance_factor(ndb),
            Node::Saved(saved_node) => saved_node.compute_balance_factor(ndb),
        }
    }

    pub fn get<DB>(
        &self,
        ndb: &NodeDb<DB>,
        key: &NonEmptyBz,
    ) -> Result<(U63, Option<Cow<'_, NonEmptyBz>>), NodeError>
    where
        DB: KVStore,
    {
        // leaf node check
        if let Some(value) = self.value() {
            if key == self.key() {
                return Ok((U63::MIN, Some(Cow::Borrowed(value))));
            }

            return Ok((U63::MIN, None));
        }

        // unwrap is safe because self is inner node
        if key < self.key() {
            return self
                .left()
                .map(|left| left.fetch_full(ndb))
                .transpose()?
                .unwrap()
                .read()?
                .get(ndb, key)
                .map(|(i, v)| (i, v.map(Cow::into_owned).map(Cow::Owned)));
        }

        // unwrap is safe because self is inner node
        let right = self
            .right()
            .map(|right| right.fetch_full(ndb))
            .transpose()?
            .unwrap();
        let right = right.read()?;
        let right_size = right.size().get();

        right.get(ndb, key).map(|(i, v)| {
            (
                // direct subtraction is safe because parent's size always exceeds that of the child
                i.get()
                    .checked_add(self.size().get() - right_size)
                    .and_then(U63::new)
                    .unwrap(),
                v.map(Cow::into_owned).map(Cow::Owned),
            )
        })
    }
}

impl From<Node> for ArlockNode {
    fn from(node: Node) -> Self {
        Arc::new(RwLock::new(node))
    }
}

fn serialize_bytes<W>(bz: &NonEmptyBz, mut writer: W) -> Result<usize, SerializationError>
where
    W: Write,
{
    let head = writer.write_varint(bz.len())?;
    writer.write_all(bz.get())?;

    head.checked_add(bz.len())
        .ok_or(SerializationError::Overflow)
}
