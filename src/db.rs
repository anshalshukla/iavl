use crate::key_format::{FastKeyFormat, FastKeyPrefixFormat, Key, KeyFormat};
use crate::types::U63;
use crate::types::{BoundedUintTrait, U31};
use crate::{Node, NodeKey};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::i64;

mod keys;
mod redb;

// Constants
const STORAGE_VERSION_KEY: &str = "storage_version";
const DEFAULT_STORAGE_VERSION: &str = "1.0.0";
const FAST_STORAGE_VERSION: &str = "1.1.0";
const FAST_NODE_CACHE_SIZE: usize = 100000;

// types
pub type NodeKeyFormat = FastKeyFormat<b's'>; // s<version><nonce>
pub type NodeKeyPrefixFormat = FastKeyPrefixFormat<b's'>; // s<version>
pub type FastNodeKeyFormat = KeyFormat<b'f'>; // f<keystring>
pub type MetadataKeyFormat = KeyFormat<b'm'>; // m<keystring>

// KVStoreWithBatch is an extension of KVStore that allows for batch writes.
pub trait KVStoreWithBatch: KVStore + BatchCreator {}

pub trait Iterator {
    type Error: Error;

    fn domain(&self) -> Result<(Vec<u8>, Vec<u8>), Self::Error>;
    fn valid(&self) -> bool;
    fn next(&self);
    fn key(&self) -> Option<Vec<u8>>;
    fn value(&self) -> Option<Vec<u8>>;
}

// KVStore describes the basic interface for interacting with key-value stores.
pub trait KVStore {
    type Error: Error;

    fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>, Self::Error>;
    fn has(&self, key: Vec<u8>) -> Result<bool, Self::Error>;
    fn set(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), Self::Error>;
    fn delete(&self, key: Vec<u8>) -> Result<(), Self::Error>;
    fn iterator(
        &self,
        start: Vec<u8>,
        end: Vec<u8>,
    ) -> Result<Box<dyn Iterator<Error = DBError>>, Self::Error>;
    fn reverse_iterator(
        &self,
        start: Vec<u8>,
        end: Vec<u8>,
    ) -> Result<Box<dyn Iterator<Error = DBError>>, Self::Error>;
}

// Batch represents a group of writes.
pub trait Batch {
    type Error: Error;

    fn set(&self, key: Vec<u8>, value: Vec<u8>) -> Result<(), Self::Error>;
    fn delete(&self, key: Vec<u8>) -> Result<(), Self::Error>;
    fn write(&self) -> Result<(), Self::Error>;
    fn write_sync(&self) -> Result<(), Self::Error>;
    fn get_byte_size(&self) -> Result<u128, Self::Error>;
}

// BatchCreator defines an interface for creating a new batch.
pub trait BatchCreator {
    fn new_batch(&self) -> Box<dyn Batch<Error = DBError>>;
    fn new_batch_with_size(&self) -> Box<dyn Batch<Error = DBError>>;
}

#[derive(thiserror::Error, Debug)]
pub enum DBError {
    #[error("iterator error")]
    IteratorError,
    #[error("kv store error")]
    KVStoreError,
    #[error("batch error")]
    BatchError,
    #[error("node db error")]
    NodeDBError,
}

pub struct NodeDB<DB> {
    db: Option<DB>,      // Persistent node storage
    first_version: U63,  // First version of node_db
    latest_version: U63, // Latest version of node_db
    prune_version: U63,  // Version to prune up to
    cache: mini_moka::sync::Cache<Vec<u8>, Vec<u8>>,
    version_readers: HashMap<U63, u32>, // Number of active version readers
}

impl<DB> NodeDB<DB>
where
    DB: KVStoreWithBatch,
{
    fn new(db: DB, cache_size: u64) -> Result<Self, DBError> {
        // let key = MetadataKeyFormat::new(STORAGE_VERSION_KEY.as_bytes()).key_bytes();
        // let store_version = db.get(key).unwrap();
        let ndb = Self {
            db: Some(db),
            first_version: U63::new(0).unwrap(),
            latest_version: U63::new(0).unwrap(),
            prune_version: U63::new(0).unwrap(),
            cache: mini_moka::sync::Cache::new(cache_size),
            version_readers: HashMap::new(),
        };

        Ok(ndb)
    }

    fn node_key(nk: &[u8]) -> Option<Vec<u8>> {
        NodeKeyFormat::from_key_bytes(nk)
    }

    fn fast_node_key(nk: &[u8]) -> Option<Vec<u8>> {
        FastNodeKeyFormat::from_key_bytes(nk)
    }

    // get_node gets a node from the memory or disk. If it is an inner node, it does not
    // load its childern
    // `kf`: <version><none>
    fn get_node(&self, kf: Vec<u8>) -> Result<Box<Node>, DBError> {
        let (version, nonce) = NodeKeyFormat::extract_version_nonce(&kf).unwrap();
        let node_key = NodeKey { version, nonce };
        let node_key = node_key.serialize();

        // Check the cache
        if let Some(node) = self.cache.get(&kf) {
            let node = Node::deserialize(&node_key, &node).unwrap();
            return Ok(node);
        }

        let node = self.db.as_ref().unwrap().get(kf).unwrap().unwrap();
        let node = Node::deserialize(&node_key, &node).unwrap();

        Ok(node)
    }

    fn save_node(&self, node: Box<Node>) -> Result<(), DBError> {
        let node_encoded = node.serialize().unwrap();
        let nk = node.as_ref().node_key.clone().unwrap();
        let key = NodeKeyFormat::new(nk.version, nk.nonce);

        self.db
            .as_ref()
            .unwrap()
            .set(key.key_bytes(), node_encoded)
            .unwrap();

        Ok(())
    }

    fn get_latest_version(&self) -> Result<U63, DBError> {
        let latest_version = self.latest_version;

        if latest_version.cmp(U63::new(0).unwrap()) == Ordering::Greater {
            return Ok(latest_version);
        }

        let itr = self
            .db
            .as_ref()
            .unwrap()
            .reverse_iterator(
                NodeKeyPrefixFormat::new(U63::new(0).unwrap()).key_bytes(),
                NodeKeyPrefixFormat::new(U63::new(i64::MAX as u64).unwrap()).key_bytes(),
            )
            .unwrap();

        if itr.valid() {
            let k = itr.key().unwrap();
            let (version, _) = NodeKeyFormat::extract_version_nonce(&k).unwrap();
            return Ok(version);
        }

        Err(DBError::BatchError)
    }

    fn reset_latest_version(&mut self, version: U63) {
        self.latest_version = version;
    }

    fn delete_versions_from(&self, version: U63) -> Result<(), DBError> {
        let latest = self.get_latest_version()?;
        if latest.cmp(version) == Ordering::Less {
            return Ok(());
        }

        for (v, r) in self.version_readers.iter() {
            if v.cmp(version) != Ordering::Less && *r != 0 {
                return Err(DBError::BatchError);
            }
        }

        Ok(())
    }

    fn traverse_range(
        &self,
        start: Vec<u8>,
        end: Vec<u8>,
        f: fn(k: Vec<u8>, v: Vec<u8>) -> Result<(), DBError>,
    ) -> Result<(), DBError> {
        let itr = self.db.as_ref().unwrap().iterator(start, end);

        // for (k, v) in itr.next() {
        //     f(item.ke)
        // }

        Ok(())
    }

    fn delete_version(version: U63) -> Result<(), DBError> {
        let root_key = NodeKey::get_root_key(version);
        todo!()
    }
}
