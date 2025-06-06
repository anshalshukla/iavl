use crate::types::BoundedUintTrait;
use encoding::{encode_32bytes_hash, encode_bytes};
use prost::encoding as prost_encoding;
use sha2::{Digest, Sha256};
use thiserror::Error;
use zigzag::ZigZag;

mod encoding;
mod key_format;
mod types;

// NodeKey represents a key of node in the DB
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct NodeKey {
    version: types::U63, // version of the IAVL that this node was first added in
    nonce: types::U31,   // local nonce for the same version
}

// Node represents a node in a Tree
#[derive(PartialEq, Eq, Debug)]
pub struct Node {
    key: Vec<u8>,                         // key for the node.
    value: Option<Vec<u8>>,               // value of leaf node. If inner node, value = None
    hash: Vec<u8>, // hash of above field and left node's hash, right node's hash
    node_key: Option<Box<NodeKey>>, // node key of the nodeDB
    left_node_key: Option<Box<NodeKey>>, // node key of the left child
    right_node_key: Option<Box<NodeKey>>, // node key of the right child
    size: types::U63, // number of leaves that are under the current node. Leaf nodes have size = 1
    left_node: Option<Box<Node>>, // pointer to left child
    right_node: Option<Box<Node>>, // pointer to right child
    subtree_height: types::U7, // height of the node. Leaf nodes have height 0
}

#[derive(Error, Debug)]
pub enum NodeError {
    #[error("serialization failed: {0}")]
    SerializationError(String),
    #[error("hashing failed: {0}")]
    HashingError(String),
    #[error("deserialization failed: {0}")]
    DeserializationError(String),
    #[error("get key failed")]
    GetKeyError,
    #[error("types")]
    TypesError(#[from] types::BoundedUintError),
}

const NODE_KEY_LENGTH: usize = 12;
// non legacy
const MODE: u64 = 0;

impl TryFrom<&[u8]> for NodeKey {
    type Error = NodeError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        if bytes.len() != NODE_KEY_LENGTH {
            return Err(NodeError::DeserializationError(
                "node key length not equal to 12".into(),
            ));
        }

        let version = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let nonce = u32::from_be_bytes(bytes[8..12].try_into().unwrap());

        Ok(NodeKey {
            version: types::U63::new(version)?,
            nonce: types::U31::new(nonce)?,
        })
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            key: Vec::new(),
            value: None,
            hash: Vec::new(),
            node_key: None,
            left_node_key: None,
            right_node_key: None,
            size: types::U63::new(0).unwrap(),
            left_node: None,
            right_node: None,
            subtree_height: types::U7::new(0).unwrap(),
        }
    }
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.subtree_height.as_signed() == 0
    }

    // get_key return a byte slice of the NodeKey
    fn get_key(&self) -> Result<Vec<u8>, NodeError> {
        let nk = self.node_key.as_ref().ok_or(NodeError::GetKeyError)?;
        Ok([nk.version.to_be_bytes(), nk.nonce.to_be_bytes()].concat())
    }

    // Computes the hash of the node without computing its descendants
    // Must be called on nodes which have descendant node hashes already computed
    fn _hash(&mut self, version: types::U63) -> Result<Vec<u8>, NodeError> {
        if !self.hash.is_empty() {
            return Ok(self.hash.clone());
        }

        let data = &self.calculate_hash_bytes(version)?;
        let hash = Sha256::digest(data);
        self.hash = hash.to_vec();

        Ok(hash.to_vec())
    }

    // Serializes the node as a byte slice
    fn serialize(&self) -> Result<Vec<u8>, NodeError> {
        let mut result = Vec::new();

        let subtree_height = ZigZag::encode(self.subtree_height.as_signed());
        prost_encoding::encode_varint(subtree_height.into(), &mut result);

        let size = ZigZag::encode(self.size.as_signed());
        prost_encoding::encode_varint(size, &mut result);

        // Unlike calculate_hash_bytes, key is written for inner nodes
        result.extend(encode_bytes(&self.key));

        if self.is_leaf() {
            let value_bytes = self
                .value
                .as_ref()
                .ok_or_else(|| NodeError::SerializationError("value is none".into()))?;

            result.extend(encode_bytes(value_bytes));
        } else {
            result.extend(encode_32bytes_hash(&self.hash));

            // Encodeds the isLegacy bit, we don't support legacy structure
            // Only for the purpose of serialization consistency with go implementation
            prost_encoding::encode_varint(MODE, &mut result);

            let left = self
                .left_node_key
                .as_ref()
                .ok_or_else(|| NodeError::SerializationError("left node key is none".into()))?;

            let right = self
                .right_node_key
                .as_ref()
                .ok_or_else(|| NodeError::SerializationError("right node key is none".into()))?;

            for node_key in [left, right] {
                let version = ZigZag::encode(node_key.version.as_signed());
                let nonce = ZigZag::encode(node_key.nonce.as_signed());

                prost_encoding::encode_varint(version, &mut result);
                prost_encoding::encode_varint(nonce.into(), &mut result);
            }
        }

        Ok(result)
    }

    // Calculates the hash input bytes for this node
    // Assumes that child hashes have already been set
    fn calculate_hash_bytes(&self, version: types::U63) -> Result<Vec<u8>, NodeError> {
        let mut result = Vec::new();

        let subtree_height = ZigZag::encode(self.subtree_height.as_signed());
        prost_encoding::encode_varint(subtree_height.into(), &mut result);

        let size = ZigZag::encode(self.size.as_signed());
        prost_encoding::encode_varint(size, &mut result);

        let version = ZigZag::encode(version.as_signed());
        prost_encoding::encode_varint(version, &mut result);

        // Key is not written for inner nodes, unlike serialize

        if self.is_leaf() {
            result.extend(encode_bytes(&self.key));

            let value = self
                .value
                .as_ref()
                .ok_or_else(|| NodeError::HashingError("value is none".into()))?;

            let hash = Sha256::digest(value);
            result.extend(encode_32bytes_hash(&hash));
        } else {
            let (left, right) = match (&self.left_node, &self.right_node) {
                (Some(l), Some(r)) => (l, r),
                _ => return Err(NodeError::HashingError("left or right node is None".into())),
            };
            result.extend_from_slice(&encode_32bytes_hash(&left.hash));
            result.extend_from_slice(&encode_32bytes_hash(&right.hash));
        }

        Ok(result)
    }

    fn make_node(node_key: &[u8], node: &[u8]) -> Result<Box<Node>, NodeError> {
        let mut node = node;

        // Decode subtree height
        let subtree_height = prost_encoding::decode_varint(&mut node)
            .map_err(|_| NodeError::DeserializationError("failed to decode subtree height".into()))
            .and_then(|val| {
                let height: i64 = ZigZag::decode(val);
                types::U7::new(height as u8)
                    .map_err(|_| NodeError::DeserializationError("invalid subtree height".into()))
            })?;

        // Decode size
        let size = prost_encoding::decode_varint(&mut node)
            .map_err(|_| NodeError::DeserializationError("failed to decode size".into()))
            .and_then(|val| {
                let s: i64 = ZigZag::decode(val);
                types::U63::new(s as u64)
                    .map_err(|_| NodeError::DeserializationError("invalid size".into()))
            })?;

        // Decode key
        let key = encoding::decode_bytes(&mut node)?;

        // Initialize node
        let mut result = Node {
            key,
            size,
            subtree_height,
            ..Default::default()
        };

        // Set the node_key from the parameter (must be exactly 12 bytes)
        // Parse the node_key parameter which should be in the format of version(8 bytes) + nonce(4 bytes)
        if node_key.len() != NODE_KEY_LENGTH {
            return Err(NodeError::DeserializationError(format!(
                "node key length not equal to {NODE_KEY_LENGTH}"
            )));
        }

        let nk = Box::new(NodeKey::try_from(node_key)?);
        result.node_key = Some(nk.clone());

        // If it's a leaf node
        if result.is_leaf() {
            let value = encoding::decode_bytes(&mut node)?;
            result.value = Some(value);

            let _ = result._hash(nk.version);
        } else {
            // Read hash
            let hash = encoding::decode_bytes(&mut node)?;
            result.hash = hash;

            // Read legacy bit
            let legacy = prost_encoding::decode_varint(&mut node)
                .map_err(|_| NodeError::DeserializationError("failed to decode mode".into()))?;
            if legacy != 0 {
                return Err(NodeError::DeserializationError(
                    "legact mode not supported".into(),
                ));
            }

            result.left_node_key = Some(Box::new(encoding::decode_node_key(&mut node)?));
            result.right_node_key = Some(Box::new(encoding::decode_node_key(&mut node)?));
        }

        Ok(Box::new(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex::ToHex;
    use rand::Rng;
    use rstest::rstest;

    fn generate_random_bytes(n: usize) -> Vec<u8> {
        let mut rng = rand::rng();
        (0..n).map(|_| rng.random::<u8>()).collect()
    }

    fn create_test_node() -> Box<Node> {
        let node_key = NodeKey {
            version: types::U63::new(1).unwrap(),
            nonce: types::U31::new(1).unwrap(),
        };

        let node_key = Box::new(node_key);

        let node = Node {
            key: generate_random_bytes(10),
            value: Some(generate_random_bytes(10)),
            subtree_height: types::U7::new(0).unwrap(),
            size: types::U63::new(100).unwrap(),
            hash: generate_random_bytes(20),
            node_key: Some(node_key.clone()),
            left_node_key: Some(node_key.clone()),
            left_node: None,
            right_node_key: Some(node_key.clone()),
            right_node: None,
        };

        Box::new(node)
    }

    #[test]
    fn test_leaf_node_encoded_size() {
        let node = create_test_node();

        // leaf node
        assert_eq!(node.serialize().unwrap().len(), 25);
    }

    #[test]
    fn test_non_leaf_node_encoded_size() {
        let mut node = create_test_node();

        // make it non-leaf node
        // -1 to remove the extra mode (isLegacy) encoded byte
        node.subtree_height = types::U7::new(1).unwrap();
        assert_eq!(node.serialize().unwrap().len() - 1, 39);
    }

    fn child_node() -> (Box<NodeKey>, Vec<u8>) {
        let node_key = NodeKey {
            version: types::U63::new(1).unwrap(),
            nonce: types::U31::new(1).unwrap(),
        };

        let node_key = Box::new(node_key);

        let node_hash = vec![
            0x7f, 0x68, 0x90, 0xca, 0x16, 0xde, 0xa6, 0xe8, 0x89, 0x3d, 0x96, 0xf0, 0xa3, 0xd, 0xa,
            0x14, 0xe5, 0x55, 0x59, 0xfc, 0x9b, 0x83, 0x4, 0x91, 0xe3, 0xd2, 0x45, 0x1c, 0x81,
            0xf6, 0xd1, 0xe,
        ];

        (node_key, node_hash)
    }

    fn inner_node() -> Box<Node> {
        let (child_node_key, child_node_hash) = child_node();

        let inner_node = Node {
            subtree_height: types::U7::new(3).unwrap(),
            size: types::U63::new(7).unwrap(),
            key: "key".into(),
            node_key: Some(Box::new(NodeKey {
                version: types::U63::new(2).unwrap(),
                nonce: types::U31::new(1).unwrap(),
            })),
            left_node_key: Some(child_node_key.clone()),
            right_node_key: Some(child_node_key.clone()),
            hash: child_node_hash,
            value: None,
            left_node: None,
            right_node: None,
        };

        let inner_node = Box::new(inner_node);

        inner_node
    }

    fn leaf_node() -> Box<Node> {
        let leaf_node = Node {
            subtree_height: types::U7::new(0).unwrap(),
            size: types::U63::new(1).unwrap(),
            key: "key".into(),
            node_key: Some(Box::new(NodeKey {
                version: types::U63::new(3).unwrap(),
                nonce: types::U31::new(1).unwrap(),
            })),
            left_node_key: None,
            right_node_key: None,
            hash: vec![
                0x7f, 0x68, 0x90, 0xca, 0x16, 0xde, 0xa6, 0xe8, 0x89, 0x3d, 0x96, 0xf0, 0xa3, 0xd,
                0xa, 0x14, 0xe5, 0x55, 0x59, 0xfc, 0x9b, 0x83, 0x4, 0x91, 0xe3, 0xd2, 0x45, 0x1c,
                0x81, 0xf6, 0xd1, 0xe,
            ],
            value: Some("value".into()),
            left_node: None,
            right_node: None,
        };

        let leaf_node = Box::new(leaf_node);

        leaf_node
    }

    #[rstest]
    #[case(
        inner_node(),
        "060e036b6579207f6890ca16dea6e8893d96f0a30d0a14e55559fc9b830491e3d2451c81f6d10e0002020202"
    )]
    #[case(leaf_node(), "0002036b65790576616c7565")]
    fn test_node_encode(#[case] node: Box<Node>, #[case] expected: String) {
        let encoded = node.serialize().unwrap();
        let hash = encoded.encode_hex::<String>();
        assert_eq!(hash, expected);
    }

    #[rstest]
    #[case(
        inner_node(),
        "060e036b6579207f6890ca16dea6e8893d96f0a30d0a14e55559fc9b830491e3d2451c81f6d10e0002020202"
    )]
    #[case(leaf_node(), "0002036b65790576616c7565")]
    fn test_node_decode(#[case] node: Box<Node>, #[case] expected: String) {
        let encoded = node.serialize().unwrap();
        let decoded = Node::make_node(&node.get_key().unwrap(), &encoded).unwrap();
        assert_eq!(decoded, node);
    }
}
