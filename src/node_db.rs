use crate::db::errors::BatchError;
use crate::fast_node::Node as FastNode;
use crate::key_format::{FastKeyFormat, FastKeyPrefixFormat, Key, KeyFormat};
use crate::types::U63;
use crate::types::{BoundedUintTrait, U31};
use crate::{Node, NodeKey};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, PoisonError, RwLock};
use std::{i64, u8};

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

    fn domain(&self) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), Self::Error>;
    fn valid(&mut self) -> bool;
    fn next(&mut self) -> Option<Result<(Box<[u8]>, Box<[u8]>), Self::Error>>;
    fn key(&mut self) -> Option<Vec<u8>>;
    fn value(&mut self) -> Option<Vec<u8>>;
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

#[derive(Debug, thiserror::Error)]
pub enum DBError {
    #[error("iterator error")]
    IteratorError,
    #[error("kv store error")]
    KVStoreError,

    #[error("db not found")]
    DBNotFound,

    #[error("batch error")]
    BatchError(String),
    #[error("node db error")]
    NodeDBError,

    #[error("batch not found")]
    BatchNotFound,

    #[error("version does not exist")]
    VersionDoesNotExist,

    #[error("invalid reference root: {0}")]
    InvalidReferenceRoot(String),

    #[error("invalid root key: {0}")]
    InvalidRootKey(String),

    #[error("invalid node key: {0}")]
    InvalidNodeKey(String),

    #[error("deserialization error: {0}")]
    DeserializationError(String),

    #[error("nil value")]
    NilValue,

    #[error("value not found")]
    ValueNotFound,

    #[error("nil key")]
    NilKey,

    #[error("node key not found")]
    NodeKeyNotFound,

    #[error("error: {0}")]
    Other(String),

    #[error("poisoned")]
    PoisonedError,
}

impl<T> From<PoisonError<T>> for DBError {
    fn from(_: PoisonError<T>) -> Self {
        DBError::PoisonedError
    }
}

pub struct NodeDB<DB>
where
    DB: KVStoreWithBatch,
{
    db: Option<DB>,      // Persistent node storage
    first_version: U63,  // First version of node_db
    latest_version: U63, // Latest version of node_db
    prune_version: U63,  // Version to prune up to
    cache: mini_moka::sync::Cache<Vec<u8>, Vec<u8>>,
    version_readers: HashMap<U63, u32>, // Number of active version readers
    storage_version: String,
    batch: Option<Box<dyn Batch<Error = BatchError>>>,
}

impl<DB> NodeDB<DB>
where
    DB: KVStoreWithBatch,
{
    pub fn new(db: DB, cache_size: u64) -> Result<Self, DBError> {
        // let key = MetadataKeyFormat::new(STORAGE_VERSION_KEY.as_bytes()).key_bytes();
        // let store_version = db.get(key).unwrap();

        let storage_version =
            db.get(MetadataKeyFormat::new(STORAGE_VERSION_KEY.as_bytes()).key_bytes());

        let storage_version = match storage_version {
            Ok(Some(v)) => String::from_utf8(v).unwrap_or(DEFAULT_STORAGE_VERSION.to_string()),
            _ => DEFAULT_STORAGE_VERSION.to_string(),
        };

        let ndb = Self {
            db: Some(db),
            first_version: U63::new(0).unwrap(),
            latest_version: U63::new(0).unwrap(),
            prune_version: U63::new(0).unwrap(),
            cache: mini_moka::sync::Cache::new(cache_size),
            version_readers: HashMap::new(),
            storage_version,
            batch: None,
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
    pub fn get_node(&self, kf: Vec<u8>) -> Result<Arc<RwLock<Node>>, DBError> {
        let (version, nonce) = NodeKeyFormat::extract_version_nonce(&kf).ok_or(
            DBError::InvalidNodeKey(format!("invalid node key: {:?}", kf)),
        )?;
        let node_key = NodeKey { version, nonce };
        let node_key = node_key.serialize();

        // Check the cache
        if let Some(node) = self.cache.get(&kf) {
            let node = Node::deserialize(&node_key, &node).map_err(|e| {
                DBError::DeserializationError(format!("deserialization error: {:?}", e))
            })?;
            return Ok(Arc::new(RwLock::new(*node)));
        }

        let node = self.db.as_ref().unwrap().get(kf).unwrap().unwrap();
        let node = Node::deserialize(&node_key, &node).unwrap();

        Ok(Arc::new(RwLock::new(*node)))
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

    //TODO, should we use Box<Node> or &Node?...I think we should use &Node(Vaibhav)

    // save_fast_node saves a fast node to the disk.
    fn save_fast_node(&self, node: Box<FastNode>) -> Result<(), DBError> {
        self.save_fast_node_unlocked(node, true)
    }

    fn save_fast_node_unlocked(
        &self,
        node: Box<FastNode>,
        _should_cache: bool,
    ) -> Result<(), DBError> {
        if node.get_key().is_empty() {
            return Err(DBError::NodeDBError);
        }

        if self.batch.is_none() {
            return Err(DBError::BatchNotFound);
        }

        let buf = node.serialize();

        self.batch
            .as_ref()
            .unwrap()
            .set(node.get_key(), buf)
            .map_err(|e| DBError::BatchError(e.to_string()))?;

        Ok(())
    }

    // Returns true if the upgrade to latest storage version has been performed, false otherwise.
    fn has_upgraded_to_fast_storage(&self) -> bool {
        self.storage_version >= FAST_STORAGE_VERSION.to_string()
    }

    // Has checks if a node key exists in the database.
    fn has(&self, nk: Vec<u8>) -> Result<bool, DBError> {
        if self.db.is_none() {
            return Err(DBError::DBNotFound);
        }
        let has = self
            .db
            .as_ref()
            .unwrap()
            .has(nk)
            .map_err(|e| DBError::Other(e.to_string()))?;
        Ok(has)
    }

    pub fn get_latest_version(&self) -> Result<U63, DBError> {
        let latest_version = self.latest_version;

        if latest_version.cmp(U63::new(0).unwrap()) == Ordering::Greater {
            return Ok(latest_version);
        }

        if self.db.is_none() {
            return Err(DBError::DBNotFound);
        }

        let mut itr = self
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

        Err(DBError::BatchError("latest version not found".to_string()))
    }

    fn reset_latest_version(&mut self, version: U63) {
        self.latest_version = version;
    }

    // hasVersion checks if the given version exists.
    fn has_version(&self, version: U63) -> Result<bool, DBError> {
        let root_key = NodeKey::get_root_key(version);
        let key = NodeKeyFormat::from_key_bytes(&root_key).ok_or(DBError::InvalidRootKey(
            format!("invalid root key: {:?}", root_key),
        ))?;
        let has = self.has(key)?;
        Ok(has)
    }

    fn delete_versions_from(&self, version: U63) -> Result<(), DBError> {
        let latest = self.get_latest_version()?;
        if latest.cmp(version) == Ordering::Less {
            return Ok(());
        }

        for (v, r) in self.version_readers.iter() {
            if v.cmp(version) != Ordering::Less && *r != 0 {
                return Err(DBError::BatchError("version readers not found".to_string()));
            }
        }

        Ok(())
    }

    fn get_root_key(&self, version: U63) -> Result<Vec<u8>, String> {
        todo!()
    }

    // SaveEmptyRoot saves the empty root.
    fn save_empty_root(&self, version: U63) -> Result<(), DBError> {
        let root_key = NodeKey::get_root_key(version);
        let key = NodeKeyFormat::from_key_bytes(&root_key).ok_or(DBError::InvalidRootKey(
            format!("invalid root key: {:?}", root_key),
        ))?;
        self.batch
            .as_ref()
            .unwrap()
            .set(key, vec![])
            .map_err(|e| DBError::Other(e.to_string()))?;
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

    fn get_storage_version(&self) -> String {
        self.storage_version.clone()
    }

    fn get_root(&self, version: U63) -> Result<Vec<u8>, DBError> {
        let root_key = NodeKey::get_root_key(version);
        let key = NodeKeyFormat::from_key_bytes(&root_key);

        if key.is_none() {
            return Err(DBError::NodeDBError);
        }

        let val = self
            .db
            .as_ref()
            .unwrap()
            .get(key.unwrap())
            .map_err(|e| DBError::Other(e.to_string()))?;

        if val.is_none() {
            return Err(DBError::NodeDBError);
        }

        if val.as_ref().unwrap().is_empty() {
            return Ok(val.unwrap());
        }

        let (is_ref, n) = is_reference_root(val.as_ref().unwrap());
        if is_ref {
            // point to the prev version
            match n {
                NodeKeyFormat::LENGTH => {
                    let nk = NodeKey::get_node_key(&val.clone().unwrap()[1..]);
                    let val = self
                        .db
                        .as_ref()
                        .unwrap()
                        .get(val.clone().unwrap())
                        .map_err(|e| DBError::Other(e.to_string()))?;

                    if nk.is_none() {
                        return Err(DBError::NilKey);
                    }

                    if val.is_none() {
                        // check if the prev version root is reformatted due to the pruning
                        let rnk = NodeKey {
                            version: nk.unwrap().version,
                            nonce: U31::new(0).unwrap(),
                        };

                        let rnk_key = NodeKeyFormat::from_key_bytes(&rnk.serialize());

                        if rnk_key.is_none() {
                            return Err(DBError::NilKey);
                        }

                        let val = self
                            .db
                            .as_ref()
                            .unwrap()
                            .get(rnk_key.unwrap())
                            .map_err(|e| DBError::Other(e.to_string()))?;

                        if val.is_none() {
                            return Err(DBError::VersionDoesNotExist);
                        }

                        return Ok(rnk.serialize());
                    }

                    return Ok(nk.unwrap().serialize());
                }

                NodeKeyPrefixFormat::LENGTH => {
                    // (prefix, version) before the lazy pruning

                    let mut val = val.unwrap();
                    val.push(1);
                    return Ok(val[1..].to_vec());
                }
                _ => {
                    return Err(DBError::DBNotFound);
                }
            }
        }

        Ok(root_key)
    }
}

pub struct RootKeyCache {
    versions: [i64; 2],
    root_keys: [Option<Vec<u8>>; 2],
    next: usize,
}

impl RootKeyCache {
    pub fn new() -> Self {
        RootKeyCache {
            versions: [-1, -1], // invalid version
            root_keys: [None, None],
            next: 0,
        }
    }

    pub fn get_root_key<DB: KVStoreWithBatch>(
        &mut self,
        version: U63,
        ndb: &NodeDB<DB>,
    ) -> Result<Vec<u8>, String> {
        for i in 0..2 {
            if self.versions[i] == version.as_signed() {
                if let Some(ref key) = self.root_keys[i] {
                    return Ok(key.clone());
                }
            }
        }

        let root_key = ndb.get_root_key(version)?; // simulate ndb.GetRoot
        self.set_root_key(version, root_key.clone());
        Ok(root_key)
    }

    fn set_root_key(&mut self, version: U63, root_key: Vec<u8>) {
        self.versions[self.next] = version.as_signed();
        self.root_keys[self.next] = Some(root_key);
        self.next = (self.next + 1) % 2;
    }
}

fn is_reference_root(bz: &[u8]) -> (bool, usize) {
    if bz[0] == NodeKeyPrefixFormat::prefix() {
        return (true, bz.len());
    }
    (false, 0)
}
