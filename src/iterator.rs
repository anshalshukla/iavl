use std::sync::{Arc, RwLock};

use crate::{
    Node, NodeKey,
    immutable_tree::ImmutableTree,
    node_db::{DBError, KVStoreWithBatch, NodeDB},
    types::{BoundedUintTrait, U7, U31, U63},
};

#[derive(Clone)]
pub struct DelayedNode {
    node: Arc<RwLock<Node>>,
    delayed: bool,
}

#[derive(Default, Clone)]
pub struct DelayedNodes {
    nodes: Vec<DelayedNode>,
}

impl DelayedNodes {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn pop(&mut self) -> Option<(Arc<RwLock<Node>>, bool)> {
        self.nodes.pop().map(|node| (node.node, node.delayed))
    }

    pub fn push(&mut self, node: Arc<RwLock<Node>>, delayed: bool) {
        self.nodes.push(DelayedNode { node, delayed });
    }

    pub fn length(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone)]
pub struct Traversal<DB>
where
    DB: KVStoreWithBatch,
{
    pub tree: Arc<ImmutableTree<DB>>, // Using Arc for shared ownership
    pub start: Vec<u8>,               // iteration domain start
    pub end: Vec<u8>,                 // iteration domain end
    pub ascending: bool,              // ascending traversal
    pub inclusive: bool,              // end key inclusiveness
    pub post: bool,                   // postorder traversal
    pub delayed_nodes: DelayedNodes,  // delayed nodes to be traversed
}

impl<DB> Traversal<DB>
where
    DB: KVStoreWithBatch,
{
    pub fn new_traversal(
        tree: Arc<ImmutableTree<DB>>,
        start: Vec<u8>,
        end: Vec<u8>,
        ascending: bool,
        inclusive: bool,
        post: bool,
    ) -> Traversal<DB> {
        let mut delayed_nodes = DelayedNodes::new();

        Traversal {
            tree,
            start,
            end,
            ascending,
            inclusive,
            post,
            delayed_nodes,
        }
    }

    pub fn next(&mut self) -> Result<Option<Arc<RwLock<Node>>>, DBError> {
        // End of traversal
        if self.delayed_nodes.length() == 0 {
            return Ok(None);
        }

        // Get next node to process
        let delayed_node = self.delayed_nodes.pop();
        if delayed_node.is_none() {
            return Ok(None);
        }

        let (node, delayed) = delayed_node.unwrap();

        // Already expanded, immediately return
        if !delayed {
            return Ok(Some(node));
        }

        // Check if node is within bounds
        let after_start =
            self.start.is_empty() || self.start.as_slice() < node.read()?.key.as_slice();
        let start_or_after = after_start || self.start.as_slice() == node.read()?.key.as_slice();
        let before_end = self.end.is_empty() || node.read()?.key.as_slice() < self.end.as_slice();
        let before_end = if self.inclusive {
            before_end || node.read()?.key.as_slice() == self.end.as_slice()
        } else {
            before_end
        };

        // Case of postorder (A-1 and B-1)
        // Recursively process left sub-tree, then right-subtree, then node itself
        if self.post && (!node.read()?.is_leaf() || (start_or_after && before_end)) {
            self.delayed_nodes.push(node.clone(), false);
        }

        // Case of branch node, traversing children (A-2)
        if !node.read()?.is_leaf() {
            if self.ascending {
                // Ascending: traverse left subtree first, then right
                if before_end {
                    // Push right node for later traversal
                    if let Some(right_node) = node.read()?.get_right_node(&self.tree) {
                        self.delayed_nodes.push(right_node, true);
                    }
                }
                if after_start {
                    // Push left node for immediate traversal
                    if let Some(left_node) = node.read()?.get_left_node(&self.tree) {
                        self.delayed_nodes.push(left_node, true);
                    }
                }
            } else {
                // Descending: traverse right subtree first, then left
                if after_start {
                    if let Some(left_node) = node.read()?.get_left_node(&self.tree) {
                        self.delayed_nodes.push(left_node, true);
                    }
                }
                if before_end {
                    if let Some(right_node) = node.read()?.get_right_node(&self.tree) {
                        self.delayed_nodes.push(right_node, true);
                    }
                }
            }
        }

        // Case of preorder traversal (A-3 and B-2)
        // Process root then recursively process left child, then right child
        if !self.post && (!node.read()?.is_leaf() || (start_or_after && before_end)) {
            return Ok(Some(node));
        }

        // Keep traversing and expanding remaining delayed nodes (A-4)
        self.next()
    }
}

pub struct NodeIterator<DB>
where
    DB: KVStoreWithBatch,
{
    nodes_to_visit: Vec<Arc<RwLock<Node>>>,
    node_db: Arc<NodeDB<DB>>,
    is_valid: bool,
}

impl<DB> NodeIterator<DB>
where
    DB: KVStoreWithBatch,
{
    pub fn new(root_key: Vec<u8>, ndb: Arc<NodeDB<DB>>) -> Result<Self, DBError> {
        let mut nodes_to_visit = Vec::new();

        if root_key.is_empty() {
            // If root key is empty, return iterator with empty array
            return Ok(NodeIterator {
                nodes_to_visit,
                node_db: ndb,
                is_valid: true,
            });
        }

        // Get the node for the root key
        let node = ndb.get_node(root_key)?;

        // Put it in the array
        nodes_to_visit.push(node);

        Ok(NodeIterator {
            nodes_to_visit,
            node_db: ndb,
            is_valid: true,
        })
    }

    // GetNode returns the current visiting node.
    pub fn get_node(&self) -> Arc<RwLock<Node>> {
        self.nodes_to_visit[self.nodes_to_visit.len() - 1].clone()
    }

    // Valid checks if the validator is valid.
    pub fn valid(&self) -> bool {
        self.is_valid && self.nodes_to_visit.len() > 0
    }

    // Next moves forward the traversal.
    pub fn next(&mut self, is_skipped: bool) {
        if !self.valid() {
            return;
        }
        let node = self.get_node();
        self.nodes_to_visit = self.nodes_to_visit[..self.nodes_to_visit.len() - 1].to_vec();

        if is_skipped {
            return;
        }

        let node = node.as_ref().read();
        if node.is_err() {
            self.is_valid = false;
            return;
        }

        if node.as_ref().unwrap().is_leaf() {
            return;
        }

        if node.as_ref().unwrap().right_node_key.is_some() {
            let right_node = self.node_db.get_node(
                node.as_ref()
                    .unwrap()
                    .right_node_key
                    .as_ref()
                    .unwrap()
                    .serialize(),
            );
            if right_node.is_err() {
                self.is_valid = false;
                return;
            }
            self.nodes_to_visit.push(right_node.unwrap());
        }

        if node.as_ref().unwrap().left_node_key.is_some() {
            let left_node = self.node_db.get_node(
                node.as_ref()
                    .unwrap()
                    .left_node_key
                    .as_ref()
                    .unwrap()
                    .serialize(),
            );
            if left_node.is_err() {
                self.is_valid = false;
                return;
            }
            self.nodes_to_visit.push(left_node.unwrap());
        }
    }
}

#[derive(Debug)]
pub enum IteratorError {
    NilTreeGiven,
    TraversalError(DBError),
}

impl From<DBError> for IteratorError {
    fn from(err: DBError) -> Self {
        IteratorError::TraversalError(err)
    }
}

/// Iterator is a store.Iterator for ImmutableTree
pub struct Iterator<DB>
where
    DB: KVStoreWithBatch,
{
    start: Vec<u8>,
    end: Vec<u8>,
    key: Vec<u8>,
    value: Option<Vec<u8>>,
    valid: bool,
    traversal: Traversal<DB>,
}

impl<DB> Iterator<DB>
where
    DB: KVStoreWithBatch,
{
    /// Returns a new iterator over the immutable tree. If the tree is None, the iterator will be invalid.
    pub fn new(
        start: &[u8],
        end: &[u8],
        ascending: bool,
        inclusive: bool,
        post: bool,
        tree: Arc<ImmutableTree<DB>>,
    ) -> Self {
        let root = tree.get_root();

        let mut iter = Iterator {
            start: start.into(),
            end: end.into(),
            key: Vec::new(),
            value: None,
            valid: true,
            traversal: Traversal::new_traversal(
                tree,
                start.into(),
                end.into(),
                ascending,
                inclusive,
                post,
            ),
        };

        iter.traversal.delayed_nodes.push(root, true);

        iter.next();

        iter
    }

    /// Domain returns the start and end range of the iterator
    pub fn domain(&self) -> (&Vec<u8>, &Vec<u8>) {
        (&self.start, &self.end)
    }

    /// Valid returns whether the iterator is valid
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// Key returns the current key
    pub fn key(&self) -> &Vec<u8> {
        &self.key
    }

    /// Value returns the current value
    pub fn value(&self) -> Option<Vec<u8>> {
        self.value.clone()
    }

    /// Next moves the iterator to the next item
    pub fn next(&mut self) {
        let next_node = self.traversal.next();

        if next_node.is_err() {
            println!("Testing log: next_node error: {:?}", next_node.err());
            self.valid = false;
            return;
        }

        match next_node {
            Ok(Some(node)) => {
                let node = node.as_ref().read();
                if node.is_err() {
                    println!("Testing log: next_node error:7");

                    self.valid = false;
                    return;
                }

                // If an error occurred or no more nodes, update iterator state accordingly
                if node.as_ref().unwrap().is_leaf() {
                    self.key = node.as_ref().unwrap().key.clone();
                    self.value = node.as_ref().unwrap().value.clone();
                    return;
                }

                // Continue to next if this is not a leaf node
                self.next();
            }
            Ok(None) => {
                println!("Testing log: next_node error:8");

                self.valid = false;
                return;
            }
            Err(err) => {
                println!("Testing log: next_node error:1 {:?}", err);

                // Error occurred
                self.valid = false;
                return;
            }
        }
    }

    /// IsFast returns true if iterator uses fast strategy
    pub fn is_fast(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{MockDB, TestUtils};

    // /// Test configuration for iterator tests
    // #[derive(Clone)]
    // pub struct IteratorTestConfig {
    //     pub start_byte_to_set: u8,
    //     pub end_byte_to_set: u8,
    //     pub start_iterate: Option<Vec<u8>>,
    //     pub end_iterate: Option<Vec<u8>>,
    //     pub ascending: bool,
    // }

    // /// Setup test data for iterator testing
    // pub fn setup_test_data(config: &IteratorTestConfig) -> Vec<(Vec<u8>, Vec<u8>)> {
    //     let mut data = Vec::new();

    //     for i in config.start_byte_to_set..=config.end_byte_to_set {
    //         let key = vec![i];
    //         let value = format!("value_{}", i as char).into_bytes();
    //         data.push((key, value));
    //     }

    //     data
    // }

    // /// Create test configuration for various test scenarios
    // pub fn create_test_configs() -> Vec<IteratorTestConfig> {
    //     vec![
    //         // Empty range
    //         IteratorTestConfig {
    //             start_byte_to_set: b'a',
    //             end_byte_to_set: b'z',
    //             start_iterate: Some(b"e".to_vec()),
    //             end_iterate: Some(b"w".to_vec()),
    //             ascending: true,
    //         },
    //         // Normal range ascending
    //         IteratorTestConfig {
    //             start_byte_to_set: b'a',
    //             end_byte_to_set: b'z',
    //             start_iterate: Some(b"e".to_vec()),
    //             end_iterate: Some(b"w".to_vec()),
    //             ascending: false,
    //         },
    //         // Normal range descending
    //         IteratorTestConfig {
    //             start_byte_to_set: b'a',
    //             end_byte_to_set: b'z',
    //             start_iterate: None,
    //             end_iterate: None,
    //             ascending: true,
    //         },
    //         // Full range ascending
    //         IteratorTestConfig {
    //             start_byte_to_set: b'a',
    //             end_byte_to_set: b'z',
    //             start_iterate: None,
    //             end_iterate: None,
    //             ascending: false,
    //         },
    //         // Full range descending
    //         IteratorTestConfig {
    //             start_byte_to_set: b'a',
    //             end_byte_to_set: b'z',
    //             start_iterate: None,
    //             end_iterate: None,
    //             ascending: false,
    //         },
    //     ]
    // }

    // // Helper function to create test configurations
    // fn create_test_config(
    //     start_byte: u8,
    //     end_byte: u8,
    //     start_iter: Option<&[u8]>,
    //     end_iter: Option<&[u8]>,
    //     ascending: bool,
    // ) -> IteratorTestConfig {
    //     IteratorTestConfig {
    //         start_byte_to_set: start_byte,
    //         end_byte_to_set: end_byte,
    //         start_iterate: start_iter.map(|s| s.to_vec()),
    //         end_iterate: end_iter.map(|e| e.to_vec()),
    //         ascending,
    //     }
    // }

    // #[test]
    // fn test_iterator_nil_tree_failure() {
    //     // This test simulates the Go test: TestIterator_NewIterator_NilTree_Failure
    //     // Since our iterator requires an Arc<ImmutableTree>, we'll test error conditions
    //     // by testing the domain and validity functions instead

    //     let start = b"a";
    //     let end = b"c";

    //     // Since we can't create an iterator with nil tree in Rust due to type safety,
    //     // we test that our mock can simulate this scenario
    //     let mock_error = DBError::NilTreeGiven;
    //     assert_eq!(format!("{}", mock_error), "nil tree given");
    // }

    // #[test]
    // fn test_iterator_empty_invalid() {
    //     // This test simulates: TestIterator_Empty_Invalid
    //     let config = create_test_config(b'a', b'z', Some(b"a"), Some(b"a"), true);

    //     // Test that our test config was created correctly
    //     assert_eq!(config.start_byte_to_set, b'a');
    //     assert_eq!(config.end_byte_to_set, b'z');
    //     assert_eq!(config.start_iterate, Some(b"a".to_vec()));
    //     assert_eq!(config.end_iterate, Some(b"a".to_vec()));
    //     assert!(config.ascending);

    //     // In a real implementation, this would test that an iterator with
    //     // start == end would be invalid (empty range)
    // }

    // #[test]
    // fn test_iterator_basic_ranged_ascending_success() {
    //     // This test simulates: TestIterator_Basic_Ranged_Ascending_Success
    //     let config = create_test_config(b'a', b'z', Some(b"e"), Some(b"w"), true);

    //     assert_eq!(config.start_iterate, Some(b"e".to_vec()));
    //     assert_eq!(config.end_iterate, Some(b"w".to_vec()));
    //     assert!(config.ascending);

    //     // Test data setup
    //     let test_data = TestUtils::setup_test_data(&config);
    //     assert!(!test_data.is_empty());

    //     // Verify data is in expected range
    //     for (key, _) in &test_data {
    //         assert!(key[0] >= config.start_byte_to_set);
    //         assert!(key[0] <= config.end_byte_to_set);
    //     }
    // }

    // #[test]
    // fn test_iterator_basic_ranged_descending_success() {
    //     // This test simulates: TestIterator_Basic_Ranged_Descending_Success
    //     let config = create_test_config(b'a', b'z', Some(b"e"), Some(b"w"), false);

    //     assert_eq!(config.start_iterate, Some(b"e".to_vec()));
    //     assert_eq!(config.end_iterate, Some(b"w".to_vec()));
    //     assert!(!config.ascending); // descending

    //     let test_data = TestUtils::setup_test_data(&config);
    //     assert!(!test_data.is_empty());
    // }

    // #[test]
    // fn test_iterator_basic_full_ascending_success() {
    //     // This test simulates: TestIterator_Basic_Full_Ascending_Success
    //     let config = create_test_config(b'a', b'z', None, None, true);

    //     assert_eq!(config.start_iterate, None);
    //     assert_eq!(config.end_iterate, None);
    //     assert!(config.ascending);

    //     let test_data = TestUtils::setup_test_data(&config);
    //     assert!(!test_data.is_empty());
    //     // Should include full range from 'a' to 'z'
    //     assert_eq!(test_data.len(), (b'z' - b'a' + 1) as usize);
    // }

    // #[test]
    // fn test_iterator_basic_full_descending_success() {
    //     // This test simulates: TestIterator_Basic_Full_Descending_Success
    //     let config = create_test_config(b'a', b'z', None, None, false);

    //     assert_eq!(config.start_iterate, None);
    //     assert_eq!(config.end_iterate, None);
    //     assert!(!config.ascending); // descending

    //     let test_data = TestUtils::setup_test_data(&config);
    //     assert!(!test_data.is_empty());
    //     assert_eq!(test_data.len(), (b'z' - b'a' + 1) as usize);
    // }

    // #[test]
    // fn test_node_iterator_with_empty_root() {
    //     // This test simulates: TestNodeIterator_WithEmptyRoot

    //     // Test with nil root key (empty vec in Rust)
    //     let empty_root: Vec<u8> = vec![];
    //     assert!(empty_root.is_empty());

    //     // Test with None equivalent (empty vec)
    //     let nil_root: Vec<u8> = vec![];
    //     assert!(nil_root.is_empty());

    //     // In the actual implementation, both should result in invalid iterators
    // }

    // #[test]
    // fn test_iterator_next_error_handling() {
    //     // This test simulates: TestIterator_Next_ErrorHandling

    //     // Test that we can create error scenarios
    //     let db_error = crate::node_db::DBError::NodeKeyNotFound;
    //     let iter_error = IteratorError::from(db_error);

    //     match iter_error {
    //         IteratorError::TraversalError(_) => {
    //             // Expected - error was properly converted
    //         }
    //         _ => panic!("Expected TraversalError variant"),
    //     }
    // }

    // #[test]
    // fn test_delayed_nodes() {
    //     let mut delayed = DelayedNodes::new();
    //     assert_eq!(delayed.length(), 0);

    //     let node = TestUtils::create_leaf_node("test", "value");
    //     delayed.push(node.clone(), true);
    //     assert_eq!(delayed.length(), 1);

    //     let popped = delayed.pop().unwrap();
    //     assert_eq!(delayed.length(), 0);
    //     assert!(popped.1); // delayed flag should be true
    // }

    // #[test]
    // fn test_delayed_nodes_operations() {
    //     let mut delayed = DelayedNodes::new();

    //     // Test empty state
    //     assert_eq!(delayed.length(), 0);
    //     assert!(delayed.pop().is_none());

    //     // Test single push/pop
    //     let node1 = TestUtils::create_leaf_node("key1", "value1");
    //     delayed.push(node1, true);
    //     assert_eq!(delayed.length(), 1);

    //     let (_, delayed_flag) = delayed.pop().unwrap();
    //     assert_eq!(delayed.length(), 0);
    //     assert!(delayed_flag);

    //     // Test multiple pushes - LIFO (stack behavior)
    //     let node2 = TestUtils::create_leaf_node("key2", "value2");
    //     let node3 = TestUtils::create_leaf_node("key3", "value3");

    //     delayed.push(node2, false);
    //     delayed.push(node3, true);
    //     assert_eq!(delayed.length(), 2);

    //     // Pop order should be LIFO
    //     let (_, flag1) = delayed.pop().unwrap();
    //     assert!(flag1); // Last pushed was delayed=true

    //     let (_, flag2) = delayed.pop().unwrap();
    //     assert!(!flag2); // First pushed was delayed=false

    //     assert_eq!(delayed.length(), 0);
    // }

    // #[test]
    // fn test_iterator_error_enum() {
    //     // Test error enum variants
    //     let db_err = crate::node_db::DBError::NodeKeyNotFound;
    //     let iter_err = IteratorError::from(db_err);

    //     match iter_err {
    //         IteratorError::TraversalError(_) => {
    //             // Expected conversion
    //         }
    //         _ => panic!("Expected TraversalError variant"),
    //     }

    //     let nil_err = IteratorError::NilTreeGiven;
    //     match nil_err {
    //         IteratorError::NilTreeGiven => {
    //             // Expected variant
    //         }
    //         _ => panic!("Expected NilTreeGiven variant"),
    //     }

    //     // Test debug formatting
    //     let debug_str = format!("{:?}", nil_err);
    //     assert!(debug_str.contains("NilTreeGiven"));
    // }

    // #[test]
    // fn test_mock_error_types() {
    //     // Test our mock error types that simulate Go error conditions
    //     let errors = vec![
    //         MockError::NilTreeGiven,
    //         MockError::NilNdbGiven,
    //         MockError::NilAdditionsGiven,
    //         MockError::NilRemovalsGiven,
    //     ];

    //     for error in errors {
    //         let error_str = format!("{}", error);
    //         let debug_str = format!("{:?}", error);

    //         // Each error should have a non-empty string representation
    //         assert!(!error_str.is_empty());
    //         assert!(!debug_str.is_empty());
    //     }
    // }

    // #[test]
    // fn test_node_creation() {
    //     // Test that we can create nodes for testing
    //     let node = TestUtils::create_leaf_node("test_key", "test_value");
    //     let node_read = node.read().unwrap();

    //     assert_eq!(node_read.key, b"test_key");
    //     assert_eq!(node_read.value, Some(b"test_value".to_vec()));
    //     assert!(node_read.is_leaf());
    //     assert_eq!(node_read.size.get(), 1);

    //     // Test branch node creation
    //     let left_child = TestUtils::create_leaf_node("left", "left_val");
    //     let right_child = TestUtils::create_leaf_node("right", "right_val");

    //     let branch = TestUtils::create_branch_node("branch", Some(left_child), Some(right_child));
    //     let branch_read = branch.read().unwrap();

    //     assert_eq!(branch_read.key, b"branch");
    //     assert_eq!(branch_read.value, None); // Branch nodes don't have values
    //     assert!(!branch_read.is_leaf()); // Should be a branch node
    // }

    // #[test]
    // fn test_test_configurations() {
    //     // Test that we can create various test configurations
    //     let configs = TestUtils::create_test_configs();
    //     assert!(!configs.is_empty());

    //     // Verify we have different types of configurations
    //     let has_ascending = configs.iter().any(|c| c.ascending);
    //     let has_descending = configs.iter().any(|c| !c.ascending);
    //     let has_range = configs.iter().any(|c| c.start_iterate.is_some());
    //     let has_full = configs.iter().any(|c| c.start_iterate.is_none());

    //     assert!(has_ascending);
    //     assert!(has_descending);
    //     assert!(has_range);
    //     assert!(has_full);
    // }

    #[test]
    fn test_iterator_with_random_tree() {
        let tree = TestUtils::create_random_iavl_tree(10).unwrap();
        let mut iter = Iterator::new(b"", b"", true, false, false, tree);
        assert!(iter.valid());
    }

    #[test]
    fn test_node_iterator_with_empty_root() {
        let db = TestUtils::create_mock_db();
        let ndb = Arc::new(NodeDB::new(db, 100).unwrap());

        // empty root key
        let node_iterator = NodeIterator::new(vec![], ndb.clone());
        assert!(node_iterator.is_ok());
        assert!(!node_iterator.unwrap().valid());

        // empty root key
        let node_iterator = NodeIterator::new(vec![1, 2, 3], ndb);
        assert!(node_iterator.is_err());
    }

    #[test]
    fn test_iterator_next_error_handling() {
        let tree = TestUtils::create_mock_tree_with_root().unwrap();

        let left = TestUtils::create_leaf_node("5", "left_value", 1, 2);

        tree.root.write().unwrap().left_node_key =
            Some(left.read().unwrap().node_key.clone().unwrap());
        tree.root.write().unwrap().right_node = Some(left);

        tree.root.write().unwrap().size = U63::new(1).unwrap();
        tree.root.write().unwrap().subtree_height = U7::new(2).unwrap();
        tree.root.write().unwrap().value = None;

        let mut iter = Iterator::new(b"", b"", true, false, false, tree);

        iter.next();

        assert!(!iter.valid());
    }
}
