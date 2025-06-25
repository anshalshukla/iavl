use prost::{bytes::Buf, encoding as prost_encoding};

use crate::BoundedUintTrait;
use crate::types;
use crate::{NodeError, NodeKey};
use zigzag::ZigZag;

// encode_bytes returns a length-prefixed byte slice.
pub fn encode_bytes(bytes: &[u8]) -> Vec<u8> {
    let bytes_length = bytes.len();
    let mut result = Vec::with_capacity(1 + bytes_length);
    prost_encoding::encode_varint(bytes_length as u64, &mut result);
    result.extend_from_slice(bytes);

    println!("Testing log: result: {}", result.len());
    result
}

// Encode 32 byte long hash.
pub fn encode_32bytes_hash(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    prost_encoding::encode_varint(32, &mut result);
    // result.push(32 as u8);
    result.extend_from_slice(bytes);
    result
}

pub fn decode_bytes(bytes: &mut &[u8]) -> Result<Vec<u8>, NodeError> {
    let value_len = prost_encoding::decode_varint(bytes)
        .map_err(|_| NodeError::DeserializationError("failed to decode value length".into()))?;

    let value = bytes
        .get(..value_len as usize)
        .ok_or_else(|| NodeError::DeserializationError("invalid value".into()))?;

    bytes.advance(value_len as usize);

    Ok(value.to_vec())
}

pub fn decode_node_key(bytes: &mut &[u8]) -> Result<NodeKey, NodeError> {
    let version = prost_encoding::decode_varint(bytes)
        .map_err(|_| NodeError::DeserializationError("failed to decode version".into()))
        .and_then(|val| {
            let version: i64 = ZigZag::decode(val);
            types::U63::new(version as u64)
                .map_err(|_| NodeError::DeserializationError("invalid size".into()))
        })?;

    let nonce = prost_encoding::decode_varint(bytes)
        .map_err(|_| NodeError::DeserializationError("failed to decode none".into()))
        .and_then(|val| {
            let nonce: i64 = ZigZag::decode(val);
            types::U31::new(nonce as u32)
                .map_err(|_| NodeError::DeserializationError("invalid nonce".into()))
        })?;

    Ok(NodeKey { version, nonce })
}
