use std::{borrow::Cow, sync::PoisonError};

use crate::node::{InnerNodeError, db::NodeDbError};

pub type Result<T, E = MutableTreeError> = core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum MutableTreeError {
    #[error("node db error: {0}")]
    NodeDb(#[from] NodeDbError),

    #[error("missing node key error")]
    MissingNodeKey,

    #[error("conflicting root error")]
    ConflictingRoot,

    #[error("invalid root error: {0}")]
    InvalidRoot(Cow<'static, str>),

    #[error("inner node error: {0}")]
    InnerNode(#[from] InnerNodeError),

    #[error("poisoned lock error")]
    PoisonedLock,
}

impl<T> From<PoisonError<T>> for MutableTreeError {
    fn from(_err: PoisonError<T>) -> Self {
        Self::PoisonedLock
    }
}
