mod error;

pub use self::error::MutableTreeError;

use core::mem;

use crate::{
    kvstore::{KVStore, MutKVStore},
    node::{Child, DeserializedNode, InnerNode, LeafNode, SavedNode},
    types::{NonEmptyBz, U7, U63},
};

use super::{
    immutable::ImmutableTree,
    node::{ArlockNode, db::NodeDb},
};

use self::error::Result;

pub struct MutableTree<DB> {
    root: ArlockNode,
    ndb: NodeDb<DB>,
    last_saved: ImmutableTree<DB>,
}

#[bon::bon]
impl<DB> MutableTree<DB>
where
    DB: MutKVStore + KVStore + Clone,
{
    #[builder]
    pub fn new(root: ArlockNode, ndb: NodeDb<DB>) -> Result<Self> {
        let version = root
            .read()?
            .as_saved()
            .map(|sn| save_new_root_node_checked(sn, &ndb).map(|_| sn.version()))
            .transpose()?
            .ok_or(MutableTreeError::MissingNodeKey)?;

        let last_saved = ImmutableTree::builder()
            .root(root.clone())
            .ndb(ndb.clone())
            .version(version)
            .build();

        Ok(Self {
            root,
            ndb,
            last_saved,
        })
    }
}

impl<DB> MutableTree<DB> {
    pub fn root(&self) -> &ArlockNode {
        &self.root
    }

    pub fn last_saved(&self) -> &ImmutableTree<DB> {
        &self.last_saved
    }
}

impl<DB> MutableTree<DB>
where
    DB: MutKVStore + KVStore + Clone,
{
    /// inserts/updates the node with given key-value pair, and returns the old root node along with
    /// the boolean value [`true`]
    pub fn insert(&mut self, key: NonEmptyBz, value: NonEmptyBz) -> Result<(ArlockNode, bool)> {
        recursive_insert(&self.root, &self.ndb, key, value)
            .map(|(new_root, updated)| (mem::replace(&mut self.root, new_root), updated))
    }
}

fn save_new_root_node_checked<DB>(saved_root_node: &SavedNode, ndb: &NodeDb<DB>) -> Result<()>
where
    DB: MutKVStore + KVStore,
{
    let maybe_existing = ndb.save_non_overwririting_one_node(saved_root_node)?;

    match (saved_root_node, maybe_existing) {
        (_, None) => Ok(()),
        (SavedNode::Inner(root), Some(DeserializedNode::Inner(deserialized_drafted, hash))) => {
            if root.hash() != &hash {
                return Err(MutableTreeError::ConflictingRoot);
            }

            let deserialized_hashed = deserialized_drafted.into_hashed(*root.version())?;

            root.hash()
                .eq(deserialized_hashed.hash())
                .then_some(())
                .ok_or(MutableTreeError::ConflictingRoot)
        }
        (SavedNode::Leaf(root), Some(DeserializedNode::Leaf(deserialized_drafted))) => {
            deserialized_drafted
                .into_hashed(*root.version())
                .hash()
                .eq(root.hash())
                .then_some(())
                .ok_or(MutableTreeError::ConflictingRoot)
        }
        _ => Err(MutableTreeError::ConflictingRoot),
    }
}

fn handle_leaf_insert_case(
    node: &ArlockNode,
    existing_key: &NonEmptyBz,
    new_key: NonEmptyBz,
    new_value: NonEmptyBz,
) -> Result<(ArlockNode, bool)> {
    let new_leaf = LeafNode::builder().key(new_key).value(new_value).build();

    if new_leaf.key() == existing_key {
        return Ok((new_leaf.into(), true));
    }

    let (inner_key, left, right) = if new_leaf.key() < existing_key {
        (
            existing_key.clone(),
            ArlockNode::from(new_leaf),
            node.clone(),
        )
    } else {
        (
            new_leaf.key().clone(),
            node.clone(),
            ArlockNode::from(new_leaf),
        )
    };

    let inner = InnerNode::builder()
        .key(inner_key)
        .height(U7::ONE)
        .size(U63::TWO)
        .left(left.into())
        .right(right.into())
        .build();

    Ok((inner.into(), false))
}

fn recursive_insert<DB>(
    node: &ArlockNode,
    ndb: &NodeDb<DB>,
    key: NonEmptyBz,
    value: NonEmptyBz,
) -> Result<(ArlockNode, bool)>
where
    DB: KVStore,
{
    // TODO: replace with if-let chain after Rust 1.88 release
    {
        let gnode = node.read()?;
        if gnode.is_leaf() {
            return handle_leaf_insert_case(node, gnode.key(), key, value);
        }
    }

    // unwraps are safe because inner node must contain children
    let (left, right) = {
        let mut gnode = node.write()?;
        let left = gnode
            .left_mut()
            .map(Child::extract)
            .transpose()?
            .map(|c| c.fetch_full(ndb))
            .transpose()?
            .unwrap();
        let right = gnode
            .right_mut()
            .map(Child::extract)
            .transpose()?
            .map(|c| c.fetch_full(ndb))
            .transpose()?
            .unwrap();

        (left, right)
    };

    let gnode = node.read()?;

    let (left, right, updated) = if &key < gnode.key() {
        let (new_left, updated) = recursive_insert(&left, ndb, key, value)?;
        (new_left, right, updated)
    } else {
        let (new_right, updated) = recursive_insert(&right, ndb, key, value)?;
        (left, new_right, updated)
    };

    let mut inner = InnerNode::builder()
        .key(gnode.key().clone())
        .height(gnode.height())
        .size(gnode.size())
        .left(Child::Full(left))
        .right(Child::Full(right))
        .build();

    if updated {
        return Ok((inner.into(), true));
    }

    inner.make_balanced(ndb)?;

    Ok((inner.into(), updated))
}
