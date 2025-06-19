use crate::key_format::FastKeyFormat;

use super::Key;
use super::keys::{
    FastNodeKeyFormatDBKey, MetadataKeyFormatDBKey, NodeKeyFormatDBKey, NodeKeyPrefixFormatDBKey,
};
use super::{Batch, BatchCreator, Iterator, KVStore, KVStoreWithBatch, NodeKeyFormat};
use redb;

const NODE_KEY_FORMAT_TABLE: redb::TableDefinition<NodeKeyFormatDBKey, Vec<u8>> =
    redb::TableDefinition::new("node_key_format");
const NODE_KEY_PREFIX_FORMAT_TABLE: redb::TableDefinition<NodeKeyPrefixFormatDBKey, Vec<u8>> =
    redb::TableDefinition::new("node_key_prefix_format");
const FAST_NODE_KEY_FORMAT_TABLE: redb::TableDefinition<FastNodeKeyFormatDBKey, Vec<u8>> =
    redb::TableDefinition::new("fast_node_key_format");
const METADATA_KEY_FORMAT_TABLE: redb::TableDefinition<MetadataKeyFormatDBKey, Vec<u8>> =
    redb::TableDefinition::new("metadata_key_format");

struct Db {
    db: redb::Database,
}
