use core::num::TryFromIntError;

use std::{borrow::Cow, io, sync::PoisonError};

use super::inner::InnerNodeError;

pub type Result<T, E = NodeError> = core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("poisoned lock error: lock must not be poisoned")]
    PoisonedLock,

    #[error("inner node error: {0}")]
    Inner(#[from] InnerNodeError),

    #[error("deserialization error: {0}")]
    Deserialization(#[from] DeserializationError),

    #[error("serialization error: {0}")]
    Serialization(#[from] SerializationError),

    #[error("other error: {0}")]
    Other(Cow<'static, str>),
}

#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("overflow error")]
    Overflow,
}

#[derive(Debug, thiserror::Error)]
pub enum DeserializationError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid integer error")]
    InvalidInteger,

    #[error("zero prefix length error")]
    ZeroPrefixLength,

    #[error("prefix length mismatch error")]
    PrefixLengthMismatch,

    #[error("invalid mode")]
    InvalidMode,
}

impl<T> From<PoisonError<T>> for NodeError {
    fn from(_err: PoisonError<T>) -> Self {
        Self::PoisonedLock
    }
}

impl From<TryFromIntError> for DeserializationError {
    fn from(_err: TryFromIntError) -> Self {
        Self::InvalidInteger
    }
}
