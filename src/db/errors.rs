use std::{error::Error, fmt::Display};

use thiserror::Error;

use crate::node_db::DBError;

#[derive(Debug, Error)]
pub enum BatchError {
    KeyEmpty,
    ValueNil,
    BatchNil,
    RocksDBError(rocksdb::Error),
}

impl Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<rocksdb::Error> for BatchError {
    fn from(err: rocksdb::Error) -> Self {
        BatchError::RocksDBError(err)
    }
}
