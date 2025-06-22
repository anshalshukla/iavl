use super::{Node, NodeError};
use crate::iterator::Iterator;
use crate::node_db::{KVStoreWithBatch, NodeDB};
use crate::types::{BoundedUintTrait, U7, U63};
use std::sync::{Arc, RwLock};

/// ImmutableTree contains the immutable tree at a given version. It is typically created by calling
/// MutableTree.GetImmutable(), in which case the returned tree is safe for concurrent access as
/// long as the version is not deleted via DeleteVersion() or the tree's pruning settings.
///
/// Returned key/value byte slices must not be modified, since they may point to data located inside
/// IAVL which would also be modified.
///
/// TODO: implement fast storage optimization
///
#[derive(Clone)]
pub struct ImmutableTree<DB>
where
    DB: KVStoreWithBatch,
{
    /// Root node of the tree
    pub root: Arc<RwLock<Node>>,

    /// Root node of the tree
    //    pub root: Option<Node>,

    /// Node database for persistent storage
    pub ndb: Arc<NodeDB<DB>>,

    /// Version of the tree
    pub version: U63,

    /// Flag to control fast storage upgrade behavior
    /// TODO: implement this later
    pub skip_fast_storage_upgrade: bool,
}

impl<DB> ImmutableTree<DB>
where
    DB: KVStoreWithBatch,
{
    pub fn new(
        root: Arc<RwLock<Node>>,
        ndb: Arc<NodeDB<DB>>,
        version: U63,
        skip_fast_storage_upgrade: bool,
    ) -> Self {
        Self {
            root,
            ndb,
            version,
            skip_fast_storage_upgrade,
        }
    }

    // String returns a string representation of Tree.
    pub fn string(&self) -> String {
        todo!()
    }

    // Has returns whether or not a key exists.
    pub fn has(&self, key: &Vec<u8>) -> Result<bool, NodeError> {
        return self.root.read()?.has(self, key);
    }

    // Size returns the number of leaf nodes in the tree.
    pub fn size(&self) -> U63 {
        // TODO: check if this is correct way to handle the error
        let root = self.root.read();
        if root.is_err() {
            return U63::zero();
        }

        return root.unwrap().size;
    }

    // Version returns the version of the tree.
    pub fn version(&self) -> U63 {
        self.version
    }

    // Height returns the height of the tree.
    pub fn height(&self) -> U7 {
        let root = self.root.read();
        if root.is_err() {
            return U7::zero();
        }

        return root.unwrap().subtree_height;
    }

    // Hash returns the root hash.
    pub fn hash(&self) -> Result<Vec<u8>, NodeError> {
        return self
            .root
            .write()?
            .hash_with_count(U63::new(self.version.get() + 1)?);
    }

    // GetWithIndex returns the index and value of the specified key if it exists, or None and the next index
    // otherwise. The returned value must not be modified, since it may point to data stored within
    // IAVL.
    //
    // The index is the index in the list of leaf nodes sorted lexicographically by key. The leftmost leaf has index 0.
    // Its neighbor has index 1 and so on.
    pub fn get_with_index(&self, key: &[u8]) -> Result<(i64, Option<Vec<u8>>), NodeError> {
        return self.root.read()?.get(self, key);
    }

    // GetByIndex gets the key and value at the specified index.
    pub fn get_by_index(&self, index: i64) -> Result<(Vec<u8>, Vec<u8>), NodeError> {
        return self.root.read()?.get_by_index(self, index);
    }

    // Get returns the value of the specified key if it exists, or None.
    // The returned value must not be modified, since it may point to data stored within IAVL.
    // Get potentially employs a more performant strategy than GetWithIndex for retrieving the value.
    // If tree.skip_fast_storage_upgrade is true, this will work almost the same as GetWithIndex.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, NodeError> {
        if !self.skip_fast_storage_upgrade {
            todo!()
            // TODO: Implement fast storage optimization
            // For now, fall back to regular node traversal
        }

        // Use regular strategy for reading from the current tree
        let (_, result) = self.root.read()?.get(self, key)?;
        Ok(result)
    }

    // Iterate iterates over all keys of the tree. The keys and values must not be modified,
    // since they may point to data stored within IAVL. Returns true if stopped by callback, false otherwise
    pub fn iterate(
        self: Arc<Self>,
        callback: fn(&Vec<u8>, Option<Vec<u8>>) -> bool,
    ) -> Result<bool, NodeError> {
        let mut iter = self.iterator(&[], &[], true)?;

        while iter.valid() {
            if callback(iter.key(), iter.value()) {
                return Ok(true);
            }
            iter.next();
        }

        return Ok(false);
    }

    // Iterator returns an iterator over the immutable tree.
    pub fn iterator(
        self: Arc<Self>,
        start: &[u8],
        end: &[u8],
        ascending: bool,
    ) -> Result<Iterator<DB>, NodeError> {
        if !self.skip_fast_storage_upgrade {
            todo!()
        }
        //TODO: confirm the default values for inclusive and post
        return Ok(Iterator::new(start, end, ascending, false, false, self));
    }

    // IterateRange makes a callback for all nodes with key between start and end non-inclusive.
    // If either are empty, then it is open on that side (empty, empty is the same as Iterate). The keys and
    // values must not be modified, since they may point to data stored within IAVL.
    pub fn iterate_range(
        self: Arc<Self>,
        start: &[u8],
        end: &[u8],
        ascending: bool,
        callback: fn(&[u8], &[u8]) -> bool,
    ) -> Result<bool, NodeError> {
        let fb = |node: &Node| -> bool {
            if node.is_leaf() {
                if let Some(ref value) = node.value {
                    return callback(&node.key, value);
                }
            }
            false
        };

        self.root
            .read()?
            .traverse_in_range(self.clone(), start, end, ascending, false, false, fb)
    }

    // IterateRangeInclusive makes a callback for all nodes with key between start and end inclusive.
    // If either are empty, then it is open on that side (empty, empty is the same as Iterate). The keys and
    // values must not be modified, since they may point to data stored within IAVL.
    pub fn iterate_range_inclusive(
        self: Arc<Self>,
        start: &[u8],
        end: &[u8],
        ascending: bool,
        callback: fn(&[u8], &[u8], i64) -> bool,
    ) -> Result<bool, NodeError> {
        let fb = |node: &Node| -> bool {
            if node.is_leaf() {
                if let (Some(value), Some(node_key)) = (&node.value, &node.node_key) {
                    return callback(&node.key, value, node_key.version.as_signed());
                }
            }
            false
        };

        self.root
            .read()?
            .traverse_in_range(self.clone(), start, end, ascending, true, false, fb)
    }

    // isLatestTreeVersion returns true if the tree is the latest version.
    pub fn is_latest_tree_version(&self) -> Result<bool, NodeError> {
        let latest_version = self
            .ndb
            .get_latest_version()
            .map_err(|_| NodeError::DBError)?;
        Ok(self.version == latest_version)
    }

    // nodeSize is like Size, but includes inner nodes too.
    // used only for testing.
    pub fn node_size(&self) -> i64 {
        let root = self.root.read();
        if root.is_err() {
            return 0;
        }

        return root.unwrap().size.as_signed() * 2 - 1;
    }

    pub fn get_root(&self) -> Arc<RwLock<Node>> {
        self.root.clone()
    }
}
