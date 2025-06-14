use prost::encoding as prost_encoding;
use zigzag::ZigZag;

use crate::{
    encoding::{self, encode_bytes},
    types::{BoundedUintTrait, U63},
};

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Node {
    key: Vec<u8>,
    value: Vec<u8>,
    version_last_updated_at: U63,
}

#[derive(Debug)]
pub enum NodeError {
    DeserializationError,
}

impl Node {
    pub fn new_node(key: Vec<u8>, value: Vec<u8>, version: U63) -> Box<Node> {
        let node = Self {
            key,
            value,
            version_last_updated_at: version,
        };
        Box::new(node)
    }

    pub fn get_key(&self) -> Vec<u8> {
        self.key.clone()
    }

    pub fn get_value(&self) -> Vec<u8> {
        self.value.clone()
    }

    pub fn get_version_last_updated_at(&self) -> U63 {
        self.version_last_updated_at
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();
        let version = ZigZag::encode(self.version_last_updated_at.as_signed());
        prost_encoding::encode_varint(version, &mut result);

        result.extend(encode_bytes(&self.value));

        result
    }

    pub fn deserialize(node_key: &[u8], node: &[u8]) -> Result<Box<Node>, NodeError> {
        let mut node = node;

        // Decode version
        let version_last_updated_at = prost_encoding::decode_varint(&mut node)
            .map_err(|_| NodeError::DeserializationError)
            .and_then(|val| {
                let version: i64 = ZigZag::decode(val);
                U63::new(version as u64).map_err(|_| NodeError::DeserializationError)
            })?;

        // Decode value
        let value =
            encoding::decode_bytes(&mut node).map_err(|_| NodeError::DeserializationError)?;

        // Key
        let key = node_key.to_vec();
        let node = Self {
            key,
            value,
            version_last_updated_at,
        };

        Ok(Box::new(node))
    }

    pub fn encoded_size(&self) -> usize {
        self.serialize().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex::ToHex;
    use rand::Rng;

    fn generate_random_bytes(n: usize) -> Vec<u8> {
        let mut rng = rand::rng();
        (0..n).map(|_| rng.random::<u8>()).collect()
    }

    #[test]
    fn test_node_encoded_size() {
        let node = Node {
            key: generate_random_bytes(10),
            version_last_updated_at: U63::new(1).unwrap(),
            value: generate_random_bytes(10),
        };

        let expected_size = 1 + node.value.len() + 1;

        assert_eq!(expected_size, node.encoded_size());
    }

    fn create_node() -> Box<Node> {
        let node = Node {
            key: vec![0x4],
            version_last_updated_at: U63::new(1).unwrap(),
            value: vec![0x2],
        };
        Box::new(node)
    }

    #[test]
    fn test_node_encode() {
        let node = create_node();
        let encoded = node.serialize();
        let hash = encoded.encode_hex::<String>();

        assert_eq!(hash, "020102")
    }

    #[test]
    fn test_node_decode() {
        let node = create_node();
        let encoded = node.serialize();
        let decoded = Node::deserialize(&node.key, &encoded).unwrap();
        assert_eq!(decoded, node);
    }
}
