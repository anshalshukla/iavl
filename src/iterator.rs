use std::sync::{Arc, RwLock};

use crate::{
    Node,
    immutable_tree::ImmutableTree,
    node_db::{DBError, KVStoreWithBatch, NodeDB},
};

#[derive(Clone)]
struct DelayedNode {
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
    pub fn new(start: &[u8], end: &[u8], ascending: bool, tree: Arc<ImmutableTree<DB>>) -> Self {
        let mut iter = Iterator {
            start: start.into(),
            end: end.into(),
            key: Vec::new(),
            value: None,
            valid: false,
            traversal: Traversal::new_traversal(
                tree,
                start.into(),
                end.into(),
                ascending,
                false,
                false,
            ),
        };

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
            self.valid = false;
            return;
        }

        match next_node {
            Ok(Some(node)) => {
                let node = node.as_ref().read();
                if node.is_err() {
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
                self.valid = false;
            }
            Err(err) => {
                // Error occurred
                self.valid = false;
            }
        }
    }

    /// IsFast returns true if iterator uses fast strategy
    pub fn is_fast(&self) -> bool {
        false
    }
}
