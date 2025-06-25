use core::num::NonZeroUsize;

use std::io::{self, Read, Write};

use bytes::{BufMut, BytesMut};
use integer_encoding::VarIntReader;

use crate::{
    kvstore::KVStore,
    types::{NonEmptyBz, U7, U31, U63},
};

use super::{
    ArlockNode, BalanceFactor, DeserializationError, InnerNode, InnerNodeError, LeafNode, Node,
    NodeHash, NodeHashPair, NodeKey, NodeKeyPair, SHA256_HASH_LEN, SerializationError,
    db::NodeDb,
    info::{Drafted, Hashed, Saved},
    inner::Child,
};

#[derive(Debug)]
pub enum DraftedNode {
    Inner(InnerNode<Drafted>),
    Leaf(LeafNode<Drafted>),
}

#[derive(Debug)]
pub enum HashedNode {
    Inner(InnerNode<Hashed<NodeHashPair>>),
    Leaf(LeafNode<Hashed>),
}

#[derive(Debug)]
pub enum SavedNode {
    Inner(InnerNode<Saved<NodeHashPair, NodeKeyPair>>),
    Leaf(LeafNode<Saved>),
}

#[derive(Debug)]
pub enum DeserializedNode {
    Inner(InnerNode<Drafted>, NodeHash),
    Leaf(LeafNode<Drafted>),
}

impl DeserializedNode {
    pub fn deserialize<R>(mut reader: R) -> Result<Self, DeserializationError>
    where
        R: Read,
    {
        let height = reader
            .read_varint::<i8>()
            .map(U7::from_signed)?
            .ok_or(DeserializationError::InvalidInteger)?;

        let size = reader
            .read_varint::<i64>()
            .map(U63::from_signed)?
            .ok_or(DeserializationError::InvalidInteger)?;

        let key = deserialize_bytes(&mut reader)?;

        if height.get() == 0 {
            let value = deserialize_bytes(&mut reader)?;

            let node = LeafNode::builder().key(key).value(value).build();

            return Ok(Self::Leaf(node));
        }

        let hash = deserialize_hash(&mut reader)?;

        if reader.read_varint::<u8>()? != 0 {
            return Err(DeserializationError::InvalidMode);
        }

        let read_node_key = |reader: &mut R| -> Result<_, DeserializationError> {
            let version = reader
                .read_varint::<i64>()
                .map(U63::from_signed)?
                .ok_or(DeserializationError::InvalidInteger)?;

            let nonce = reader
                .read_varint::<i32>()
                .map(U31::from_signed)?
                .ok_or(DeserializationError::InvalidInteger)?;

            Ok(NodeKey::builder().version(version).nonce(nonce).build())
        };

        let left = read_node_key(&mut reader).map(Child::Part)?;
        let right = read_node_key(&mut reader).map(Child::Part)?;

        let inner_node = InnerNode::builder()
            .key(key)
            .height(height)
            .size(size)
            .left(left)
            .right(right)
            .build();

        Ok(Self::Inner(inner_node, hash))
    }
}

impl DraftedNode {
    pub fn key(&self) -> &NonEmptyBz {
        match self {
            Self::Inner(inner_node) => inner_node.key(),
            Self::Leaf(leaf_node) => leaf_node.key(),
        }
    }

    pub fn height(&self) -> U7 {
        match self {
            Self::Inner(inner_node) => inner_node.height(),
            Self::Leaf(_) => LeafNode::<()>::HEIGHT,
        }
    }

    pub fn size(&self) -> U63 {
        match self {
            Self::Inner(inner_node) => inner_node.size(),
            Self::Leaf(_) => LeafNode::<()>::SIZE,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    pub fn left(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right()),
            Self::Leaf(_) => None,
        }
    }

    pub fn left_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left_mut()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right_mut()),
            Self::Leaf(_) => None,
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
            Self::Inner(inner_node) => inner_node.compute_balance_factor(ndb),
            Self::Leaf(_) => Ok(BalanceFactor::Par),
        }
    }
}

impl HashedNode {
    pub fn key(&self) -> &NonEmptyBz {
        match self {
            Self::Inner(inner_node) => inner_node.key(),
            Self::Leaf(leaf_node) => leaf_node.key(),
        }
    }

    pub fn height(&self) -> U7 {
        match self {
            Self::Inner(inner_node) => inner_node.height(),
            Self::Leaf(_) => LeafNode::<()>::HEIGHT,
        }
    }

    pub fn size(&self) -> U63 {
        match self {
            Self::Inner(inner_node) => inner_node.size(),
            Self::Leaf(_) => LeafNode::<()>::SIZE,
        }
    }

    pub fn hash(&self) -> &NodeHash {
        match self {
            Self::Inner(inner_node) => inner_node.hash(),
            Self::Leaf(leaf_node) => leaf_node.hash(),
        }
    }

    pub fn version(&self) -> U63 {
        match self {
            Self::Inner(inner_node) => *inner_node.version(),
            Self::Leaf(leaf_node) => *leaf_node.version(),
        }
    }

    pub fn left(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right()),
            Self::Leaf(_) => None,
        }
    }

    pub fn left_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left_mut()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right_mut()),
            Self::Leaf(_) => None,
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
            Self::Inner(inner_node) => inner_node.compute_balance_factor(ndb),
            Self::Leaf(_) => Ok(BalanceFactor::Par),
        }
    }
}

impl SavedNode {
    pub fn key(&self) -> &NonEmptyBz {
        match self {
            Self::Inner(inner_node) => inner_node.key(),
            Self::Leaf(leaf_node) => leaf_node.key(),
        }
    }

    pub fn height(&self) -> U7 {
        match self {
            Self::Inner(inner_node) => inner_node.height(),
            Self::Leaf(_) => LeafNode::<()>::HEIGHT,
        }
    }

    pub fn hash(&self) -> &NodeHash {
        match self {
            Self::Inner(inner_node) => inner_node.hash(),
            Self::Leaf(leaf_node) => leaf_node.hash(),
        }
    }

    pub fn node_key(&self) -> NodeKey {
        NodeKey::builder()
            .version(self.version())
            .nonce(self.nonce())
            .build()
    }

    pub fn version(&self) -> U63 {
        match self {
            Self::Inner(inner_node) => *inner_node.version(),
            Self::Leaf(leaf_node) => *leaf_node.version(),
        }
    }

    pub fn nonce(&self) -> U31 {
        match self {
            Self::Inner(inner_node) => *inner_node.nonce(),
            Self::Leaf(leaf_node) => *leaf_node.nonce(),
        }
    }

    pub fn size(&self) -> U63 {
        match self {
            Self::Inner(inner_node) => inner_node.size(),
            Self::Leaf(_) => LeafNode::<()>::SIZE,
        }
    }

    pub fn left(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right(&self) -> Option<&Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right()),
            Self::Leaf(_) => None,
        }
    }

    pub fn left_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.left_mut()),
            Self::Leaf(_) => None,
        }
    }

    pub fn right_mut(&mut self) -> Option<&mut Child> {
        match self {
            Self::Inner(inner_node) => Some(inner_node.right_mut()),
            Self::Leaf(_) => None,
        }
    }

    pub fn serialize<W>(&self, writer: W) -> Result<NonZeroUsize, SerializationError>
    where
        W: Write,
    {
        match self {
            Self::Inner(inner_node) => inner_node.serialize(writer),
            Self::Leaf(leaf_node) => leaf_node.serialize(writer),
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
            Self::Inner(inner_node) => inner_node.compute_balance_factor(ndb),
            Self::Leaf(_) => Ok(BalanceFactor::Par),
        }
    }
}

impl From<DeserializedNode> for DraftedNode {
    fn from(node: DeserializedNode) -> Self {
        match node {
            DeserializedNode::Inner(inner_node, _) => Self::Inner(inner_node),
            DeserializedNode::Leaf(leaf_node) => Self::Leaf(leaf_node),
        }
    }
}

impl From<LeafNode<Drafted>> for DraftedNode {
    fn from(node: LeafNode<Drafted>) -> Self {
        Self::Leaf(node)
    }
}

impl From<InnerNode<Drafted>> for DraftedNode {
    fn from(node: InnerNode<Drafted>) -> Self {
        Self::Inner(node)
    }
}

impl From<DraftedNode> for Node {
    fn from(node: DraftedNode) -> Self {
        Self::Drafted(node)
    }
}

impl From<HashedNode> for Node {
    fn from(node: HashedNode) -> Self {
        Self::Hashed(node)
    }
}

impl From<SavedNode> for Node {
    fn from(node: SavedNode) -> Self {
        Self::Saved(node)
    }
}

impl From<DeserializedNode> for Node {
    fn from(node: DeserializedNode) -> Self {
        Self::Drafted(node.into())
    }
}

impl From<LeafNode<Drafted>> for Node {
    fn from(node: LeafNode<Drafted>) -> Self {
        DraftedNode::from(node).into()
    }
}

impl From<InnerNode<Drafted>> for Node {
    fn from(node: InnerNode<Drafted>) -> Self {
        DraftedNode::from(node).into()
    }
}

impl From<DraftedNode> for ArlockNode {
    fn from(node: DraftedNode) -> Self {
        Node::from(node).into()
    }
}

impl From<HashedNode> for ArlockNode {
    fn from(node: HashedNode) -> Self {
        Node::from(node).into()
    }
}

impl From<SavedNode> for ArlockNode {
    fn from(node: SavedNode) -> Self {
        Node::from(node).into()
    }
}

impl From<DeserializedNode> for ArlockNode {
    fn from(node: DeserializedNode) -> Self {
        Node::from(node).into()
    }
}

impl From<LeafNode<Drafted>> for ArlockNode {
    fn from(node: LeafNode<Drafted>) -> Self {
        Node::from(node).into()
    }
}

impl From<InnerNode<Drafted>> for ArlockNode {
    fn from(node: InnerNode<Drafted>) -> Self {
        Node::from(node).into()
    }
}

fn deserialize_hash<R>(mut reader: R) -> Result<NodeHash, DeserializationError>
where
    R: Read,
{
    let len: usize = reader
        .read_varint::<u64>()
        .map_err(DeserializationError::from)?
        .try_into()?;

    if len != SHA256_HASH_LEN {
        return Err(DeserializationError::PrefixLengthMismatch);
    }

    let mut hash: NodeHash = Default::default();

    reader
        .read_exact(&mut hash)
        .map(|_| hash)
        .map_err(From::from)
}

fn deserialize_bytes<R>(mut reader: R) -> Result<NonEmptyBz, DeserializationError>
where
    R: Read,
{
    reader
        .read_varint::<u64>()
        .map_err(From::from)
        .and_then(|len| {
            if len == 0 {
                return Err(DeserializationError::ZeroPrefixLength);
            }

            let mut buf = BytesMut::with_capacity(len.try_into()?).writer();

            // unwrap is safe because len > 0
            io::copy(&mut reader.by_ref().take(len), &mut buf)?
                .eq(&len)
                .then(|| NonEmptyBz::new(buf.into_inner().freeze()).unwrap())
                .ok_or(DeserializationError::PrefixLengthMismatch)
        })
}
