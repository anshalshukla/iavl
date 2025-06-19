use crate::types::{self, BoundedUintTrait, U31, U63};

// P: prefix
#[derive(Debug)]
pub struct FastKeyFormat<const P: u8>(types::U63, types::U31); // prefix | version | nonce

#[derive(Debug)]
pub struct FastKeyPrefixFormat<const P: u8>(types::U63); // prefix | version

#[derive(Debug)]
pub struct KeyFormat<const P: u8>(Vec<u8>); // prefix | key

const SIZE_U64: usize = size_of::<u64>();
const SIZE_U32: usize = size_of::<u32>();

impl<const P: u8> FastKeyFormat<P> {
    pub const LENGTH: usize = SIZE_U64 + SIZE_U32;

    pub fn extract_version_nonce(bytes: &[u8]) -> Option<(U63, U31)> {
        let _prefix = &bytes[0..1];
        let version: [u8; SIZE_U64] = bytes[1..1 + SIZE_U64].try_into().ok()?;
        let nonce: [u8; SIZE_U32] = bytes[1 + SIZE_U64..].try_into().ok()?;

        let version = u64::from_be_bytes(version);
        let nonce = u32::from_be_bytes(nonce);

        Some((U63::new(version).ok()?, U31::new(nonce).ok()?))
    }

    pub fn from_key_bytes(nk: &[u8]) -> Option<Vec<u8>> {
        if nk.len() != SIZE_U64 + SIZE_U32 {
            return None;
        }

        let mut result = Vec::with_capacity(1 + SIZE_U64 + SIZE_U32);
        result.push(P);
        result.extend_from_slice(nk);

        Some(result)
    }

    pub fn new(v: U63, n: U31) -> Self {
        FastKeyFormat::<P>(v, n)
    }

    pub fn length() -> usize {
        SIZE_U64 + SIZE_U32
    }
}

impl<const P: u8> FastKeyPrefixFormat<P> {
    pub const LENGTH: usize = SIZE_U64;

    pub fn extract_version(bytes: &[u8]) -> Option<U63> {
        let _prefix = &bytes[0..1];
        let version: [u8; SIZE_U64] = bytes[1..1 + SIZE_U64].try_into().ok()?;
        let version = u64::from_be_bytes(version);
        Some(U63::new(version).ok()?)
    }

    pub fn from_key_bytes(nk: &[u8]) -> Option<Vec<u8>> {
        if nk.len() != SIZE_U64 {
            return None;
        }

        let mut result = Vec::with_capacity(1 + SIZE_U64);
        result.push(P);
        result.extend_from_slice(nk);

        Some(result)
    }

    pub fn new(v: U63) -> Self {
        FastKeyPrefixFormat::<P>(v)
    }

    pub fn prefix() -> u8 {
        P
    }

    pub fn length() -> usize {
        SIZE_U64
    }
}

impl<const P: u8> KeyFormat<P> {
    pub fn extract_keystring(bytes: &[u8]) -> Option<Vec<u8>> {
        if bytes.len() == 0 {
            return None;
        }

        let keystring = &bytes[1..];
        Some(keystring.to_vec())
    }

    pub fn new(k: &[u8]) -> Self {
        KeyFormat::<P>(k.to_vec())
    }

    pub fn from_key_bytes(nk: &[u8]) -> Option<Vec<u8>> {
        let mut result = Vec::new();

        result.push(P);
        result.extend_from_slice(nk);

        Some(result)
    }
}

pub trait Key {
    fn key_bytes(&self) -> Vec<u8>;
}

impl<const P: u8> Key for FastKeyFormat<P> {
    fn key_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + SIZE_U64 + SIZE_U32);

        result.push(P);
        result.extend_from_slice(&self.0.as_signed().to_be_bytes());
        result.extend_from_slice(&self.1.as_signed().to_be_bytes());

        result
    }
}

impl<const P: u8> Key for FastKeyPrefixFormat<P> {
    fn key_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + SIZE_U64);

        result.push(P);
        result.extend_from_slice(&self.0.as_signed().to_be_bytes());

        result
    }
}

impl<const P: u8> Key for KeyFormat<P> {
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
    #[case(Box::new(FastKeyFormat::<b'a'>(version(), nonce())), vec![b'a', 0, 0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3])]
    #[case(Box::new(FastKeyPrefixFormat::<b'a'>(version())), vec![b'a', 0, 0, 0, 0, 0, 1, 2, 3])]
    #[case(Box::new(KeyFormat::<b'a'>(vec![1, 2, 3, 4])), vec![b'a', 1, 2, 3, 4])]
    fn test_keying(#[case] key: Box<dyn Key>, #[case] expected: Vec<u8>) {
        assert_eq!(key.key_bytes(), expected);
    }
}
