use std::borrow::Cow;

use crate::node::error::{DeserializationError, SerializationError};

pub type Result<T, E = NodeDbError> = core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum NodeDbError {
    #[error("store eroor: {0}")]
    Store(Box<dyn core::error::Error + Send + Sync>),

    #[error("deserialization error: {0}")]
    Deserialization(#[from] DeserializationError),

    #[error("serialization error: {0}")]
    Serialization(#[from] SerializationError),

    #[error("save unsuppported error: node kind cannot be saved")]
    SaveUnsupported,

    #[error("other error: {0}")]
    Other(Cow<'static, str>),
}
