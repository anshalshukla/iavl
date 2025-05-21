use sha2::digest::consts::U63;

use crate::types::{self, BoundedUintTrait};

// P: prefix
pub struct NodeKey<const P: u8>(types::U63, types::U31); // version | nonce
pub struct NodeKeyPrefix<const P: u8>(types::U63); // version
pub struct FastKeyPrefix<const P: u8>(Vec<u8>); // key

trait Key {
    fn key_bytes(&self) -> Vec<u8>;
}

impl<const P: u8> Key for NodeKey<P> {
    fn key_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + size_of::<u64>() + size_of::<u32>());

        result.push(P);
        result.extend_from_slice(&self.0.as_signed().to_be_bytes());
        result.extend_from_slice(&self.1.as_signed().to_be_bytes());

        result
    }
}

impl<const P: u8> Key for NodeKeyPrefix<P> {
    fn key_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + size_of::<u64>());

        result.push(P);
        result.extend_from_slice(&self.0.as_signed().to_be_bytes());

        result
    }
}

impl<const P: u8> Key for FastKeyPrefix<P> {
    fn key_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + self.0.len());

        result.push(P);
        result.extend_from_slice(&self.0);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{self, BoundedUintTrait};
    use rstest::rstest;

    fn version() -> types::U63 {
        types::U63::new(u64::from_be_bytes([0, 0, 0, 0, 0, 1, 2, 3])).unwrap()
    }

    fn nonce() -> types::U31 {
        types::U31::new(u32::from_be_bytes([0, 1, 2, 3])).unwrap()
    }

    #[rstest]
    #[case(Box::new(NodeKey::<b'a'>(version(), nonce())), vec![b'a', 0, 0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3])]
    #[case(Box::new(NodeKeyPrefix::<b'a'>(version())), vec![b'a', 0, 0, 0, 0, 0, 1, 2, 3])]
    #[case(Box::new(FastKeyPrefix::<b'a'>(vec![1, 2, 3, 4])), vec![b'a', 1, 2, 3, 4])]
    fn test_keying(#[case] key: Box<dyn Key>, #[case] expected: Vec<u8>) {
        assert_eq!(key.key_bytes(), expected);
    }
}
