use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::{
    Node, NodeKey,
    immutable_tree::ImmutableTree,
    node_db::{Batch, DBError, KVStoreWithBatch, NodeDB},
    types::{BoundedUintTrait, U7, U31, U63},
};

/// Mock database implementation for testing
#[derive(Clone)]
pub struct MockDB {
    data: Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl MockDB {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        self.data.write().unwrap().insert(key, value);
    }

    pub fn clear(&self) {
        self.data.write().unwrap().clear();
    }
}

impl crate::node_db::KVStore for MockDB {
    type Error = DBError;

    fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.data.read().unwrap().get(&key).cloned())
    }

    fn has(&self, key: Vec<u8>) -> Result<bool, Self::Error> {
        Ok(self.data.read().unwrap().contains_key(&key))
    }

    fn set(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), Self::Error> {
        self.data.write().unwrap().insert(key, value);
        Ok(())
    }

    fn delete(&self, key: Vec<u8>) -> Result<(), Self::Error> {
        self.data.write().unwrap().remove(&key);
        Ok(())
    }

    fn iterator(
        &self,
        _start: Vec<u8>,
        _end: Vec<u8>,
    ) -> Result<Box<dyn crate::node_db::Iterator<Error = DBError>>, Self::Error> {
        // Return a simple mock iterator
        Ok(Box::new(MockIterator::new()))
    }

    fn reverse_iterator(
        &self,
        _start: Vec<u8>,
        _end: Vec<u8>,
    ) -> Result<Box<dyn crate::node_db::Iterator<Error = DBError>>, Self::Error> {
        // Return a simple mock iterator
        Ok(Box::new(MockIterator::new()))
    }
}

impl crate::node_db::BatchCreator for MockDB {
    fn new_batch(&self) -> Box<dyn Batch<Error = DBError>> {
        Box::new(MockBatch::new())
    }

    fn new_batch_with_size(&self) -> Box<dyn Batch<Error = DBError>> {
        Box::new(MockBatch::new())
    }
}

impl KVStoreWithBatch for MockDB {}

/// Mock batch implementation
pub struct MockBatch {
    operations: Vec<MockOperation>,
}

#[derive(Clone)]
enum MockOperation {
    Set { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

impl MockBatch {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

impl Batch for MockBatch {
    type Error = DBError;

    fn set(&self, _key: Vec<u8>, _value: Vec<u8>) -> Result<(), Self::Error> {
        // In a real implementation, we'd store this operation
        Ok(())
    }

    fn delete(&self, _key: Vec<u8>) -> Result<(), Self::Error> {
        // In a real implementation, we'd store this operation
        Ok(())
    }

    fn write(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn write_sync(&self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn get_byte_size(&self) -> Result<u128, Self::Error> {
        Ok(self.operations.len() as u128 * 64)
    }
}

/// Mock iterator for database operations
struct MockIterator {
    valid: bool,
}

impl MockIterator {
    fn new() -> Self {
        Self { valid: false }
    }
}

impl crate::node_db::Iterator for MockIterator {
    type Error = DBError;

    fn domain(&self) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), Self::Error> {
        Ok((None, None))
    }

    fn valid(&mut self) -> bool {
        self.valid
    }

    fn next(&mut self) -> Option<Result<(Box<[u8]>, Box<[u8]>), Self::Error>> {
        self.valid = false;
        None
    }

    fn key(&mut self) -> Option<Vec<u8>> {
        None
    }

    fn value(&mut self) -> Option<Vec<u8>> {
        None
    }
}

/// Test utilities for creating trees and nodes
pub struct TestUtils;

impl TestUtils {
    const CACHE_SIZE: u64 = 100;

    pub fn create_mock_db() -> MockDB {
        MockDB::new()
    }

    /// Create a mock immutable tree for testing
    pub fn create_mock_tree_with_root() -> Result<Arc<ImmutableTree<MockDB>>, DBError> {
        let db = MockDB::new();
        let ndb = Arc::new(NodeDB::new(db, Self::CACHE_SIZE).unwrap());

        println!("reached here");

        let root = Self::create_leaf_node(6, "root_value", 1, 1);

        // Save using NodeDB's save_node method
        let node_copy = {
            let node_guard = root.read().unwrap();
            Box::new(node_guard.clone())
        };
        ndb.save_node(node_copy)?;

        let tree = ImmutableTree::new(root, ndb, U63::new(1).unwrap(), false);
        Ok(Arc::new(tree))
    }

    /// Create a mock immutable tree with a root node
    pub fn create_mock_tree_with_root_and_children() -> Result<Arc<ImmutableTree<MockDB>>, DBError>
    {
        let db = MockDB::new();
        let ndb = Arc::new(NodeDB::new(db, Self::CACHE_SIZE).unwrap());

        let left = Self::create_leaf_node(5, "left_value", 1, 2);
        // Save left child using NodeDB's save_node method
        let left_copy = {
            let node_guard = left.read().unwrap();
            Box::new(node_guard.clone())
        };
        ndb.save_node(left_copy)?;

        let right = Self::create_leaf_node(7, "right_value", 1, 3);
        // Save right child using NodeDB's save_node method
        let right_copy = {
            let node_guard = right.read().unwrap();
            Box::new(node_guard.clone())
        };
        ndb.save_node(right_copy)?;

        let root = Self::create_branch_node(6, Some(left), Some(right), 1, 1);
        // Save root using NodeDB's save_node method
        let root_copy = {
            let node_guard = root.read().unwrap();
            Box::new(node_guard.clone())
        };
        ndb.save_node(root_copy)?;

        let tree = ImmutableTree::new(root, ndb, U63::new(1).unwrap(), false);
        Ok(Arc::new(tree))
    }

    /// Create a leaf node for testing
    pub fn create_leaf_node(key: u32, value: &str, version: u64, nonce: u32) -> Arc<RwLock<Node>> {
        let node = Node {
            key: key.to_be_bytes().to_vec(),
            value: Some(value.as_bytes().to_vec()),
            hash: Vec::with_capacity(32), // placeholder hash
            node_key: Some(Box::new(NodeKey {
                version: U63::new(version).unwrap(),
                nonce: U31::new(nonce).unwrap(),
            })),
            left_node_key: None,
            right_node_key: None,
            size: U63::new(1).unwrap(),
            left_node: None,
            right_node: None,
            subtree_height: U7::new(0).unwrap(),
        };
        Arc::new(RwLock::new(node))
    }

    /// Create a branch node for testing
    pub fn create_branch_node(
        key: u32,
        left_child: Option<Arc<RwLock<Node>>>,
        right_child: Option<Arc<RwLock<Node>>>,
        version: u64,
        nonce: u32,
    ) -> Arc<RwLock<Node>> {
        let node = Node {
            key: key.to_be_bytes().to_vec(),
            value: None,       // Branch nodes typically don't have values
            hash: vec![0; 32], // placeholder hash
            node_key: Some(Box::new(NodeKey {
                version: U63::new(version).unwrap(),
                nonce: U31::new(nonce).unwrap(),
            })),
            left_node_key: left_child.clone().unwrap().read().unwrap().node_key.clone(),
            right_node_key: right_child
                .clone()
                .unwrap()
                .read()
                .unwrap()
                .node_key
                .clone(),
            size: U63::new(2).unwrap(), // Placeholder size
            left_node: left_child,
            right_node: right_child,
            subtree_height: U7::new(1).unwrap(),
        };
        Arc::new(RwLock::new(node))
    }

    /// Create a random IAVL tree with specified height
    /// All branch nodes will have both left and right children
    /// Keys follow BST property: left < parent <= right
    /// Versions follow constraint: child_version <= parent_version
    pub fn create_random_iavl_tree(height: u8) -> Result<Arc<ImmutableTree<MockDB>>, DBError> {
        if height == 0 {
            return Err(DBError::Other("Height must be greater than 0".to_string()));
        }

        let db = MockDB::new();
        let ndb = Arc::new(NodeDB::new(db, Self::CACHE_SIZE)?);
        let mut rng = rand::thread_rng();
        let mut nonce_counter = 1u32;

        // Use a much larger key range to accommodate tree splits
        let key_range = 1u32 << (height + 10); // Exponential range based on height

        // Generate the tree recursively
        let root = Self::generate_random_subtree(
            height,
            0,         // min_key
            key_range, // max_key (large enough range)
            1000,      // root version (large version space)
            &mut nonce_counter,
            &mut rng,
            &ndb,
        )?;

        let tree = ImmutableTree::new(root, ndb, U63::new(1000).unwrap(), true);
        Ok(Arc::new(tree))
    }

    /// Recursively generate a random subtree with given constraints
    fn generate_random_subtree(
        height: u8,
        min_key: u32,
        max_key: u32,
        max_version: u64,
        nonce_counter: &mut u32,
        rng: &mut impl Rng,
        ndb: &Arc<NodeDB<MockDB>>,
    ) -> Result<Arc<RwLock<Node>>, DBError> {
        if height == 0 {
            // Create leaf node
            let key_val = if min_key == max_key {
                min_key
            } else {
                rng.gen_range(min_key..=max_key)
            };
            let version = if max_version == 1 {
                1
            } else {
                rng.gen_range(1..=max_version)
            };
            let nonce = *nonce_counter;
            *nonce_counter += 1;

            let node =
                Self::create_leaf_node(key_val, &format!("value_{}", key_val), version, nonce);

            // Save using NodeDB's save_node method
            let node_copy = {
                let node_guard = node.read().unwrap();
                Box::new(node_guard.clone())
            };
            ndb.save_node(node_copy)?;

            return Ok(node);
        }

        // Create branch node
        // Ensure we have enough range for both children
        if max_key <= min_key + 1 {
            // Not enough range for proper split, create a simple structure
            let branch_key = min_key;
            let branch_version = if max_version == 1 {
                1
            } else {
                rng.gen_range(1..=max_version)
            };
            let branch_nonce = *nonce_counter;
            *nonce_counter += 1;

            // Create simple left and right children
            let left_child = Self::generate_random_subtree(
                0, // leaf
                min_key,
                min_key,
                branch_version,
                nonce_counter,
                rng,
                ndb,
            )?;

            let right_child = Self::generate_random_subtree(
                0, // leaf
                max_key,
                max_key,
                branch_version,
                nonce_counter,
                rng,
                ndb,
            )?;

            let branch_node = Self::create_branch_node(
                branch_key,
                Some(left_child.clone()),
                Some(right_child.clone()),
                branch_version,
                branch_nonce,
            );

            // Set proper node structure
            Self::finalize_branch_node(
                &branch_node,
                &Some(left_child),
                &Some(right_child),
                height,
                ndb,
            )?;

            return Ok(branch_node);
        }

        // Choose a split point that gives reasonable ranges to both children
        let range_size = max_key - min_key;
        let split_offset =
            rng.gen_range(range_size / 4..=(3 * range_size / 4).max(range_size / 4 + 1));
        let branch_key = min_key + split_offset;

        let branch_version = if max_version == 1 {
            1
        } else {
            rng.gen_range(1..=max_version)
        };
        let branch_nonce = *nonce_counter;
        *nonce_counter += 1;

        // Generate left subtree (keys < branch_key)
        let left_child = Some(Self::generate_random_subtree(
            height - 1,
            min_key,
            branch_key.saturating_sub(1).max(min_key),
            branch_version,
            nonce_counter,
            rng,
            ndb,
        )?);

        // Generate right subtree (keys >= branch_key)
        let right_child = Some(Self::generate_random_subtree(
            height - 1,
            branch_key,
            max_key,
            branch_version,
            nonce_counter,
            rng,
            ndb,
        )?);

        // Create the branch node
        let branch_node = Self::create_branch_node(
            branch_key,
            left_child.clone(),
            right_child.clone(),
            branch_version,
            branch_nonce,
        );

        // Finalize the branch node with proper structure
        Self::finalize_branch_node(&branch_node, &left_child, &right_child, height, ndb)?;

        Ok(branch_node)
    }

    /// Helper function to finalize branch node with proper node keys, size, and storage
    fn finalize_branch_node(
        branch_node: &Arc<RwLock<Node>>,
        left_child: &Option<Arc<RwLock<Node>>>,
        right_child: &Option<Arc<RwLock<Node>>>,
        height: u8,
        ndb: &Arc<NodeDB<MockDB>>,
    ) -> Result<(), DBError> {
        // Set the node keys for children and calculate size
        {
            let mut branch_guard = branch_node.write().unwrap();
            if let Some(left) = left_child {
                branch_guard.left_node_key = left.read().unwrap().node_key.clone();
            }
            if let Some(right) = right_child {
                branch_guard.right_node_key = right.read().unwrap().node_key.clone();
            }

            // Calculate size: left_size + right_size + 1
            let left_size = if let Some(left) = left_child {
                left.read().unwrap().size.get()
            } else {
                0
            };
            let right_size = if let Some(right) = right_child {
                right.read().unwrap().size.get()
            } else {
                0
            };
            branch_guard.size = U63::new(left_size + right_size).unwrap();

            // Set correct height
            branch_guard.subtree_height = U7::new(height).unwrap();
        }

        // Save using NodeDB's save_node method
        let node_copy = {
            let branch_guard = branch_node.read().unwrap();
            Box::new(branch_guard.clone())
        };
        ndb.save_node(node_copy)?;

        Ok(())
    }
}
