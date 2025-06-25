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
    // root_key is the node db key of the root node
    pub fn new(root_key: Vec<u8>, ndb: Arc<NodeDB<DB>>) -> Result<Self, DBError> {
        let mut nodes_to_visit = Vec::new();

        if root_key.is_empty() {
            // If root key is empty, return iterator with empty array
            return Err(DBError::EmptyRootKey);
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
                    .get_key(),
            );
            if right_node.is_err() {
                println!("Testing log: right_node error: {:?}", right_node.err());
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
                    .get_key(),
            );
            if left_node.is_err() {
                println!("Testing log: left_node error: {:?}", left_node.err());
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
    use crate::{
        key_format::FastKeyFormat,
        test_util::{MockDB, TestUtils},
    };
    use rand::Rng;

    #[test]
    fn test_node_iterator_success() {
        // Create a random IAVL tree for testing
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Get the root key for the NodeIterator
        let root_key = {
            let root_guard = tree.root.read().unwrap();
            root_guard.get_key().expect("Failed to get root key")
        };

        // Test 1: Check if the iterating count is same with the entire node count of the tree
        let mut node_iterator = NodeIterator::new(root_key.clone(), tree.ndb.clone());

        assert!(node_iterator.is_ok());

        let mut node_iterator = node_iterator.unwrap();

        let mut node_count = 0;
        while node_iterator.valid() {
            node_count += 1;
            node_iterator.next(false);
        }

        // The formula from Go: tree.Size() * 2 - 1
        // This represents all nodes in the tree (internal + leaf nodes)
        let expected_count = tree.size().get() * 2 - 1;
        assert_eq!(
            node_count as u64, expected_count,
            "Node count {} should match expected count {}",
            node_count, expected_count
        );

        // Test 2: Check if the skipped node count is right
        let mut node_iterator2 = NodeIterator::new(root_key, tree.ndb.clone())
            .expect("Failed to create second NodeIterator");

        let mut update_count = 0;
        let mut skip_count = 0;

        while node_iterator2.valid() {
            let node = node_iterator2.get_node();
            update_count += 1;

            let should_skip = {
                let node_guard = node.read().unwrap();
                let node_version = node_guard
                    .node_key
                    .as_ref()
                    .map(|nk| nk.version.get())
                    .unwrap_or(0);
                node_version < tree.version().get()
            };

            if should_skip {
                let node_guard = node.read().unwrap();
                // The size of the subtree without the root: node.size * 2 - 2
                skip_count += (node_guard.size.get() * 2 - 2) as i32;
            }

            node_iterator2.next(should_skip);
        }

        assert_eq!(
            node_count,
            update_count + skip_count,
            "Total node count {} should equal update count {} + skip count {}",
            node_count,
            update_count,
            skip_count
        );
    }

    #[test]
    fn test_node_iterator_empty_root() {
        let db = MockDB::new();
        let ndb = Arc::new(NodeDB::new(db, 100).unwrap());

        // Test with empty root key
        let node_iterator = NodeIterator::new(vec![], ndb.clone());

        assert!(node_iterator.is_err());
    }

    #[test]
    fn test_node_iterator_single_node() {
        let tree =
            TestUtils::create_mock_tree_with_root().expect("Failed to create mock tree with root");

        let root_key = {
            let root_guard = tree.root.read().unwrap();
            root_guard.get_key().expect("Failed to get root key")
        };

        let mut node_iterator =
            NodeIterator::new(root_key, tree.ndb.clone()).expect("Failed to create NodeIterator");

        let mut count = 0;
        while node_iterator.valid() {
            count += 1;
            node_iterator.next(false);
        }

        // Single leaf node should result in count of 1
        assert_eq!(count, 1);
    }

    #[test]
    fn test_node_iterator_with_children() {
        let tree = TestUtils::create_mock_tree_with_root_and_children()
            .expect("Failed to create mock tree with children");

        let root_key = tree.root.read().unwrap().node_key.clone().unwrap();
        let root_key = root_key.get_key();

        let mut node_iterator =
            NodeIterator::new(root_key, tree.ndb.clone()).expect("Failed to create NodeIterator");

        let mut count = 0;
        while node_iterator.valid() {
            count += 1;
            node_iterator.next(false);
        }

        // Tree with root + 2 children = 3 nodes total
        assert_eq!(count, 3);

        // Verify this matches the formula: size * 2 - 1
        let expected_count = tree.size().get() * 2 - 1;
        assert_eq!(count as u64, expected_count);
    }

    #[test]
    fn test_iterator_with_random_tree() {
        let tree = TestUtils::create_random_iavl_tree(2).unwrap();
        let mut iter = Iterator::new(b"", b"", true, false, false, tree.clone());

        let hash = tree.hash().unwrap();

        assert!(iter.valid());
    }

    #[test]
    fn test_node_iterator_with_empty_root() {
        let db = TestUtils::create_mock_db();
        let ndb = Arc::new(NodeDB::new(db, 100).unwrap());

        // empty root key
        let node_iterator = NodeIterator::new(vec![], ndb.clone());
        assert!(node_iterator.is_err());

        // empty root key
        let node_iterator = NodeIterator::new(vec![1, 2, 3], ndb);
        assert!(node_iterator.is_err());
    }

    #[test]
    fn test_iterator_next_error_handling() {
        let tree = TestUtils::create_mock_tree_with_root().unwrap();

        let left = TestUtils::create_leaf_node(5, "left_value", 1, 2);

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

    // #[test]
    // fn test_random_iavl_tree_ascending_iteration() {
    //     // Create a random IAVL tree with height 4
    //     let tree = TestUtils::create_random_iavl_tree(4).expect("Failed to create random tree");

    //     // Create iterator with nil start and end (full range) in ascending order
    //     let mut iterator = tree
    //         .iterator(&[], &[], true)
    //         .expect("Failed to create iterator");

    //     let mut previous_key: Option<Vec<u8>> = None;
    //     let mut count = 0;

    //     // Iterate through all keys and verify ascending order
    //     while iterator.valid() {
    //         let current_key = iterator.key().clone();

    //         // Verify that current key is greater than or equal to previous key
    //         if let Some(ref prev_key) = previous_key {
    //             assert!(
    //                 current_key >= *prev_key,
    //                 "Keys not in ascending order: {:?} should be >= {:?}",
    //                 current_key,
    //                 prev_key
    //             );
    //         }

    //         previous_key = Some(current_key);
    //         count += 1;
    //         iterator.next();
    //     }

    //     // Verify we actually iterated over some keys
    //     assert!(count > 0, "Iterator should have returned at least one key");

    //     // Verify count matches tree size
    //     assert_eq!(
    //         count as u64,
    //         tree.size().get(),
    //         "Iterator count {} should match tree size {}",
    //         count,
    //         tree.size().get()
    //     );

    //     println!("Successfully iterated {} keys in ascending order", count);
    // }

    // #[test]
    // fn test_random_iavl_tree_descending_iteration() {
    //     // Create a random IAVL tree with height 4
    //     let tree = TestUtils::create_random_iavl_tree(4).expect("Failed to create random tree");

    //     // Create iterator with nil start and end (full range) in descending order
    //     let mut iterator = tree
    //         .iterator(&[], &[], false)
    //         .expect("Failed to create iterator");

    //     let mut previous_key: Option<Vec<u8>> = None;
    //     let mut count = 0;

    //     // Iterate through all keys and verify descending order
    //     while iterator.valid() {
    //         let current_key = iterator.key().clone();

    //         // Verify that current key is less than or equal to previous key
    //         if let Some(ref prev_key) = previous_key {
    //             assert!(
    //                 current_key <= *prev_key,
    //                 "Keys not in descending order: {:?} should be <= {:?}",
    //                 current_key,
    //                 prev_key
    //             );
    //         }

    //         previous_key = Some(current_key);
    //         count += 1;
    //         iterator.next();
    //     }

    //     // Verify we actually iterated over some keys
    //     assert!(count > 0, "Iterator should have returned at least one key");

    //     // Verify count matches tree size
    //     assert_eq!(
    //         count as u64,
    //         tree.size().get(),
    //         "Iterator count {} should match tree size {}",
    //         count,
    //         tree.size().get()
    //     );

    //     println!("Successfully iterated {} keys in descending order", count);
    // }

    #[test]
    fn test_random_iavl_tree_both_directions() {
        // Create a random IAVL tree with height 3
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Collect all keys in ascending order
        let mut ascending_keys = Vec::new();
        let mut ascending_iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create ascending iterator");

        while ascending_iterator.valid() {
            ascending_keys.push(ascending_iterator.key().clone());
            ascending_iterator.next();
        }

        // Collect all keys in descending order
        let mut descending_keys = Vec::new();
        let mut descending_iterator = tree
            .iterator(&[], &[], false)
            .expect("Failed to create descending iterator");

        while descending_iterator.valid() {
            descending_keys.push(descending_iterator.key().clone());
            descending_iterator.next();
        }

        // Verify both iterators returned the same number of keys
        assert_eq!(
            ascending_keys.len(),
            descending_keys.len(),
            "Ascending and descending iterators should return same number of keys"
        );

        // Verify ascending order
        for i in 1..ascending_keys.len() {
            assert!(
                ascending_keys[i] >= ascending_keys[i - 1],
                "Ascending keys not in order at position {}: {:?} should be >= {:?}",
                i,
                ascending_keys[i],
                ascending_keys[i - 1]
            );
        }

        // Verify descending order
        for i in 1..descending_keys.len() {
            assert!(
                descending_keys[i] <= descending_keys[i - 1],
                "Descending keys not in order at position {}: {:?} should be <= {:?}",
                i,
                descending_keys[i],
                descending_keys[i - 1]
            );
        }

        // Verify that descending is reverse of ascending
        let mut reversed_ascending = ascending_keys.clone();
        reversed_ascending.reverse();
        assert_eq!(
            reversed_ascending, descending_keys,
            "Descending keys should be reverse of ascending keys"
        );

        println!(
            "Successfully verified both ascending and descending iteration with {} keys",
            ascending_keys.len()
        );
    }

    #[test]
    fn test_random_iavl_tree_range_iteration() {
        // Create a random IAVL tree with height 4
        let tree = TestUtils::create_random_iavl_tree(4).expect("Failed to create random tree");

        // First, collect all keys to determine the actual range
        let mut all_keys = Vec::new();
        let mut full_iterator = tree
            .clone()
            .iterator(&[], &[], true)
            .expect("Failed to create full iterator");

        while full_iterator.valid() {
            all_keys.push(full_iterator.key().clone());
            full_iterator.next();
        }

        if all_keys.is_empty() {
            println!("No keys in tree, skipping range test");
            return;
        }

        // Sort keys to ensure proper ordering
        all_keys.sort();

        let mut rng = rand::thread_rng();

        // Perform multiple random range tests
        for test_iteration in 0..5 {
            println!("Range test iteration {}", test_iteration + 1);

            // Pick random start and end indices
            let start_idx = rng.gen_range(0..all_keys.len());
            let end_idx = rng.gen_range(start_idx..all_keys.len());

            let start_key = &all_keys[start_idx];
            let end_key = if end_idx < all_keys.len() - 1 {
                &all_keys[end_idx + 1] // Make end exclusive
            } else {
                &[].to_vec() // Use empty slice for open end
            };

            println!("Testing range: start={:?}, end={:?}", start_key, end_key);

            // Test ascending iteration with range
            let mut ascending_iterator = tree
                .clone()
                .iterator(start_key, end_key, true)
                .expect("Failed to create ascending range iterator");

            let mut ascending_keys = Vec::new();
            let mut ascending_count = 0;

            while ascending_iterator.valid() {
                let current_key = ascending_iterator.key().clone();

                // Verify key is >= start
                assert!(
                    current_key >= *start_key,
                    "Ascending: Key {:?} should be >= start {:?}",
                    current_key,
                    start_key
                );

                // Verify key is < end (if end is specified)
                if !end_key.is_empty() {
                    assert!(
                        current_key < *end_key,
                        "Ascending: Key {:?} should be < end {:?}",
                        current_key,
                        end_key
                    );
                }

                ascending_keys.push(current_key);
                ascending_count += 1;
                ascending_iterator.next();
            }

            // Test descending iteration with range
            let mut descending_iterator = tree
                .clone()
                .iterator(start_key, end_key, false)
                .expect("Failed to create descending range iterator");

            let mut descending_keys = Vec::new();
            let mut descending_count = 0;

            while descending_iterator.valid() {
                let current_key = descending_iterator.key().clone();

                // Verify key is >= start
                assert!(
                    current_key >= *start_key,
                    "Descending: Key {:?} should be >= start {:?}",
                    current_key,
                    start_key
                );

                // Verify key is < end (if end is specified)
                if !end_key.is_empty() {
                    assert!(
                        current_key < *end_key,
                        "Descending: Key {:?} should be < end {:?}",
                        current_key,
                        end_key
                    );
                }

                descending_keys.push(current_key);
                descending_count += 1;
                descending_iterator.next();
            }

            // Verify both iterators returned the same number of keys
            assert_eq!(
                ascending_count, descending_count,
                "Ascending and descending should return same number of keys for range"
            );

            // Verify ascending order
            for i in 1..ascending_keys.len() {
                assert!(
                    ascending_keys[i] >= ascending_keys[i - 1],
                    "Ascending keys not in order at position {}: {:?} should be >= {:?}",
                    i,
                    ascending_keys[i],
                    ascending_keys[i - 1]
                );
            }

            // Verify descending order
            for i in 1..descending_keys.len() {
                assert!(
                    descending_keys[i] <= descending_keys[i - 1],
                    "Descending keys not in order at position {}: {:?} should be <= {:?}",
                    i,
                    descending_keys[i],
                    descending_keys[i - 1]
                );
            }

            // Verify that descending is reverse of ascending
            if !ascending_keys.is_empty() {
                let mut reversed_ascending = ascending_keys.clone();
                reversed_ascending.reverse();
                assert_eq!(
                    reversed_ascending, descending_keys,
                    "Descending keys should be reverse of ascending keys for range"
                );
            }

            println!(
                "  Range test {}: {} keys in range [{:?}, {:?})",
                test_iteration + 1,
                ascending_count,
                start_key,
                end_key
            );
        }
    }

    #[test]
    fn test_random_iavl_tree_specific_ranges() {
        // Create a random IAVL tree with height 3
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Test with various specific key ranges using u32 big-endian encoding
        let test_ranges = vec![
            (1u32.to_be_bytes().to_vec(), 100u32.to_be_bytes().to_vec()),
            (50u32.to_be_bytes().to_vec(), 200u32.to_be_bytes().to_vec()),
            (100u32.to_be_bytes().to_vec(), 300u32.to_be_bytes().to_vec()),
            (200u32.to_be_bytes().to_vec(), 500u32.to_be_bytes().to_vec()),
        ];

        for (i, (start, end)) in test_ranges.iter().enumerate() {
            println!(
                "Testing specific range {}: start={:?}, end={:?}",
                i + 1,
                u32::from_be_bytes(start.clone().try_into().unwrap()),
                u32::from_be_bytes(end.clone().try_into().unwrap())
            );

            // Test ascending
            let mut ascending_iterator = tree
                .clone()
                .iterator(start, end, true)
                .expect("Failed to create ascending iterator");

            let mut ascending_count = 0;
            while ascending_iterator.valid() {
                let key = ascending_iterator.key().clone();

                assert!(key >= *start, "Key should be >= start");
                assert!(key < *end, "Key should be < end");

                ascending_count += 1;
                ascending_iterator.next();
            }

            // Test descending
            let mut descending_iterator = tree
                .clone()
                .iterator(start, end, false)
                .expect("Failed to create descending iterator");

            let mut descending_count = 0;
            while descending_iterator.valid() {
                let key = descending_iterator.key().clone();

                assert!(key >= *start, "Key should be >= start");
                assert!(key < *end, "Key should be < end");

                descending_count += 1;
                descending_iterator.next();
            }

            assert_eq!(
                ascending_count, descending_count,
                "Ascending and descending counts should match"
            );

            println!("  Found {} keys in range", ascending_count);
        }
    }

    #[test]
    fn test_random_iavl_tree_edge_case_ranges() {
        // Create a random IAVL tree with height 3
        let tree = TestUtils::create_random_iavl_tree(3).expect("Failed to create random tree");

        // Test edge cases

        // 1. Empty range (start == end)
        let same_key = 100u32.to_be_bytes().to_vec();
        let mut empty_iterator = tree
            .clone()
            .iterator(&same_key, &same_key, true)
            .expect("Failed to create empty range iterator");

        let mut empty_count = 0;
        while empty_iterator.valid() {
            empty_count += 1;
            empty_iterator.next();
        }

        // Should return 0 keys since range is empty (start == end)
        assert_eq!(empty_count, 0, "Empty range should return no keys");

        // 2. Very narrow range
        let narrow_start = 100u32.to_be_bytes().to_vec();
        let narrow_end = 101u32.to_be_bytes().to_vec();

        let mut narrow_iterator = tree
            .clone()
            .iterator(&narrow_start, &narrow_end, true)
            .expect("Failed to create narrow range iterator");

        let mut narrow_count = 0;
        while narrow_iterator.valid() {
            let key = narrow_iterator.key().clone();
            assert!(key >= narrow_start, "Key should be >= narrow start");
            assert!(key < narrow_end, "Key should be < narrow end");

            narrow_count += 1;
            narrow_iterator.next();
        }

        println!("Narrow range [100, 101) returned {} keys", narrow_count);

        // 3. Range beyond all keys
        let high_start = 1000000u32.to_be_bytes().to_vec();
        let high_end = 2000000u32.to_be_bytes().to_vec();

        let mut high_iterator = tree
            .iterator(&high_start, &high_end, true)
            .expect("Failed to create high range iterator");

        let mut high_count = 0;
        while high_iterator.valid() {
            high_count += 1;
            high_iterator.next();
        }

        // Should return 0 keys since range is beyond all existing keys
        assert_eq!(high_count, 0, "Range beyond all keys should return no keys");

        println!("Edge case ranges tested successfully");
    }
}
