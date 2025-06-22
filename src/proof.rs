// Serializes the node as a byte slice

use prost::encoding as prost_encoding;
use sha2::{Digest, Sha256};
use std::{fmt, sync::PoisonError};
use thiserror::Error;
use zigzag::ZigZag;

use crate::{
    NodeError,
    encoding::{decode_bytes, encode_bytes},
    types::{BoundedUintTrait, U7, U63},
};

#[derive(Error, Debug)]
pub enum ProofError {
    #[error("invalid proof")]
    InvalidProof,
    #[error("invalid inputs")]
    InvalidInputs,
    #[error("invalid root")]
    InvalidRoot,
    #[error("both left and right child hashes are set")]
    BothChildrenSet,
    #[error("serialization error: {0}")]
    SerializationError(String),
    #[error("deserialization error: {0}")]
    DeserializationError(String),
    #[error("hashing failed: {0}")]
    HashingError(String),
    #[error("get key failed")]
    GetKeyError,
    #[error("types error")]
    TypesError(#[from] crate::types::BoundedUintError),
    #[error("no root node")]
    NoRootNode,
    #[error("db error")]
    DBError,
    #[error("other error")]
    Poison,

    #[error("leaf node has no value")]
    LeafNodeHasNoValue,
    #[error("node not found")]
    NodeNotFound,

    #[error("no value")]
    NoValue,

    #[error("cannot create NonExistenceProof when key exists in state")]
    KeyExistsInState,
}

impl From<NodeError> for ProofError {
    fn from(err: NodeError) -> Self {
        match err {
            NodeError::SerializationError(msg) => ProofError::SerializationError(msg),
            NodeError::DeserializationError(msg) => ProofError::DeserializationError(msg),
            NodeError::HashingError(msg) => ProofError::HashingError(msg),
            NodeError::GetKeyError => ProofError::GetKeyError,
            NodeError::TypesError(e) => ProofError::TypesError(e),
            NodeError::NoRootNode => ProofError::NoRootNode,
            NodeError::DBError => ProofError::DBError,
            NodeError::Poison => ProofError::Poison,
            NodeError::ProofError(e) => e,
        }
    }
}

impl<T> From<PoisonError<T>> for ProofError {
    fn from(_: PoisonError<T>) -> Self {
        ProofError::Poison
    }
}

/// ProofInnerNode represents an inner node in a Merkle proof
/// Contract: Left and Right can never both be set. Will result in an empty `[]` roothash
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofInnerNode {
    pub height: U7,
    pub size: U63,
    pub version: U63,
    pub left: Option<Vec<u8>>,
    pub right: Option<Vec<u8>>,
}

impl ProofInnerNode {
    /// Creates a new ProofInnerNode
    pub fn new(
        height: U7,
        size: U63,
        version: U63,
        left: Option<Vec<u8>>,
        right: Option<Vec<u8>>,
    ) -> Result<Self, ProofError> {
        // Validate that both left and right are not set
        if left.is_some() && right.is_some() {
            return Err(ProofError::BothChildrenSet);
        }

        Ok(ProofInnerNode {
            height,
            size,
            version,
            left,
            right,
        })
    }

    /// Computes the hash of this inner node given a child hash
    pub fn hash(&self, child_hash: &[u8]) -> Result<Vec<u8>, ProofError> {
        // Validate that both left and right are not set
        if self.left.is_some() && self.right.is_some() {
            return Err(ProofError::BothChildrenSet);
        }

        let mut buffer = Vec::new();

        // Encode height
        let height = ZigZag::encode(self.height.as_signed());
        prost_encoding::encode_varint(height.into(), &mut buffer);

        // Encode size
        let size = ZigZag::encode(self.size.as_signed());
        prost_encoding::encode_varint(size, &mut buffer);

        // Encode version
        let version = ZigZag::encode(self.version.as_signed());
        prost_encoding::encode_varint(version, &mut buffer);

        // Encode children based on which side has a value
        if self.left.is_none() {
            // Child hash goes on the left, right hash from proof goes on the right
            buffer.extend(encode_bytes(child_hash));
            if let Some(ref right_hash) = self.right {
                buffer.extend(encode_bytes(right_hash));
            } else {
                return Err(ProofError::InvalidProof);
            }
        } else {
            // Left hash from proof goes on the left, child hash goes on the right
            if let Some(ref left_hash) = self.left {
                buffer.extend(encode_bytes(left_hash));
            } else {
                return Err(ProofError::InvalidProof);
            }
            buffer.extend(encode_bytes(child_hash));
        }

        Ok(buffer)
    }

    /// Returns a string representation with indentation
    pub fn string_indented(&self, indent: &str) -> String {
        format!(
            "ProofInnerNode{{\n\
            {}  Height: {}\n\
            {}  Size: {}\n\
            {}  Version: {}\n\
            {}  Left: {:02X?}\n\
            {}  Right: {:02X?}\n\
            {}}}",
            indent,
            self.height.get(),
            indent,
            self.size.get(),
            indent,
            self.version.get(),
            indent,
            self.left.as_ref().map_or_else(|| vec![], |v| v.clone()),
            indent,
            self.right.as_ref().map_or_else(|| vec![], |v| v.clone()),
            indent
        )
    }
}

/// ProofLeafNode represents a leaf node in a Merkle proof
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofLeafNode {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub version: U63,
}

/// PathToLeaf represents the path from root to a leaf in the tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathToLeaf {
    pub path: Vec<ProofInnerNode>,
}

impl PathToLeaf {
    pub fn new() -> Self {
        PathToLeaf { path: Vec::new() }
    }

    /// Returns a string representation of the path
    pub fn to_string_indented(&self, indent: &str) -> String {
        if self.path.is_empty() {
            return "empty-PathToLeaf".to_string();
        }

        let mut strs = Vec::with_capacity(self.path.len());
        for (i, pin) in self.path.iter().enumerate() {
            if i == 20 {
                strs.push(format!("... ({} total)", self.path.len()));
                break;
            }
            strs.push(format!(
                "{}:{}",
                i,
                pin.string_indented(&format!("{}  ", indent))
            ));
        }

        format!(
            "PathToLeaf{{\n{}  {}\n{}}}",
            indent,
            strs.join(&format!("\n{}  ", indent)),
            indent
        )
    }

    /// Returns the index represented by this path, or -1 if invalid
    pub fn index(&self) -> i64 {
        let mut idx = 0i64;

        for (i, node) in self.path.iter().enumerate() {
            match (&node.left, &node.right) {
                (None, Some(_)) => {
                    // Left is None, continue to next node
                    continue;
                }
                (Some(_), None) => {
                    // Right is None, we went right at this node
                    if i < self.path.len() - 1 {
                        idx += node.size.get() as i64 - self.path[i + 1].size.get() as i64;
                    } else {
                        idx += node.size.get() as i64 - 1;
                    }
                }
                _ => {
                    // Both are set or both are None - invalid
                    return -1;
                }
            }
        }

        idx
    }
}

impl std::fmt::Display for PathToLeaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_indented(""))
    }
}

impl ProofLeafNode {
    /// Creates a new ProofLeafNode
    pub fn new(key: Vec<u8>, value: Vec<u8>, version: u64) -> Result<Self, ProofError> {
        Ok(ProofLeafNode {
            key,
            value,
            version: U63::new(version).map_err(|_| ProofError::InvalidInputs)?,
        })
    }

    /// Computes the hash of this leaf node
    pub fn hash(&self) -> Result<Vec<u8>, ProofError> {
        let mut buffer = Vec::new();

        // Height is 0 for leaf nodes
        let height = ZigZag::encode(0i8);
        prost_encoding::encode_varint(height.into(), &mut buffer);

        // Size is 1 for leaf nodes
        let size = ZigZag::encode(1i64);
        prost_encoding::encode_varint(size, &mut buffer);

        // Encode version
        let version = ZigZag::encode(self.version.as_signed());
        prost_encoding::encode_varint(version, &mut buffer);

        // Encode key and value
        buffer.extend(encode_bytes(&self.key));
        buffer.extend(encode_bytes(&self.value));

        Ok(buffer)
    }
}

impl fmt::Display for ProofInnerNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, "")
    }
}

impl ProofInnerNode {
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: &str) -> fmt::Result {
        write!(
            f,
            "ProofInnerNode{{\n\
            {}  Height: {}\n\
            {}  Size: {}\n\
            {}  Version: {}\n\
            {}  Left: {:02X?}\n\
            {}  Right: {:02X?}\n\
            {}}}",
            indent,
            self.height.get(),
            indent,
            self.size.get(),
            indent,
            self.version.get(),
            indent,
            self.left.as_ref().map_or_else(|| vec![], |v| v.clone()),
            indent,
            self.right.as_ref().map_or_else(|| vec![], |v| v.clone()),
            indent
        )
    }
}

impl fmt::Display for ProofLeafNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProofLeafNode{{\n\
            {}  Version: {}\n\
            {}  Key: {:02X?}\n\
            {}  Value: {:02X?}\n\
            {}}}",
            "",
            self.version.get(),
            "",
            self.key,
            "",
            self.value,
            ""
        )
    }
}
