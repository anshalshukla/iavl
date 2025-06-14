use super::{
    FastNodeKeyFormat, Iterator, KVStore, KVStoreWithBatch, MetadataKeyFormat, NodeKeyFormat,
};
use super::{Key, NodeKeyPrefixFormat};

#[derive(Debug, Clone, Copy)]
pub struct NodeKeyFormatDBKey;

#[derive(Debug, Clone, Copy)]
pub struct NodeKeyPrefixFormatDBKey;

#[derive(Debug, Clone, Copy)]
pub struct FastNodeKeyFormatDBKey;

#[derive(Debug, Clone, Copy)]
pub struct MetadataKeyFormatDBKey;

impl redb::Value for NodeKeyFormatDBKey {
    type SelfType<'a> = NodeKeyFormat;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        Some(1 + size_of::<u64>() + size_of::<u32>())
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let (version, nonce) = Self::SelfType::extract_version_nonce(data).unwrap();
        Self::SelfType::new(version, nonce)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        <Vec<u8> as redb::Value>::as_bytes(&value.key_bytes())
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("node_key_format")
    }
}

impl redb::Key for NodeKeyFormatDBKey {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        data1.cmp(data2)
    }
}

impl redb::Value for NodeKeyPrefixFormatDBKey {
    type SelfType<'a> = NodeKeyPrefixFormat;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        Some(1 + size_of::<u64>())
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let version = Self::SelfType::extract_version(data).unwrap();
        Self::SelfType::new(version)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        <Vec<u8> as redb::Value>::as_bytes(&value.key_bytes())
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("node_key_prefix_format")
    }
}

impl redb::Key for NodeKeyPrefixFormatDBKey {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        data1.cmp(data2)
    }
}

impl redb::Value for FastNodeKeyFormatDBKey {
    type SelfType<'a> = FastNodeKeyFormat;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let keystring = Self::SelfType::extract_keystring(data).unwrap();
        Self::SelfType::new(&keystring)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        <Vec<u8> as redb::Value>::as_bytes(&value.key_bytes())
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("fast_node_key_format")
    }
}

impl redb::Key for FastNodeKeyFormatDBKey {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        data1.cmp(data2)
    }
}

impl redb::Value for MetadataKeyFormatDBKey {
    type SelfType<'a> = MetadataKeyFormat;
    type AsBytes<'a> = Vec<u8>;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let keystring = Self::SelfType::extract_keystring(data).unwrap();
        Self::SelfType::new(&keystring)
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        <Vec<u8> as redb::Value>::as_bytes(&value.key_bytes())
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("metadata_key_format")
    }
}

impl redb::Key for MetadataKeyFormatDBKey {
    fn compare(data1: &[u8], data2: &[u8]) -> std::cmp::Ordering {
        data1.cmp(data2)
    }
}
