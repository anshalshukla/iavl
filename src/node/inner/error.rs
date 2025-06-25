use std::{borrow::Cow, sync::PoisonError};

use crate::node::db::NodeDbError;

pub type Result<T, E = InnerNodeError> = core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum InnerNodeError {
    #[error("into hashed error: {0}")]
    IntoHashed(Cow<'static, str>),

    #[error("into saved error: {0}")]
    IntoSaved(Cow<'static, str>),

    #[error("poisoned lock error: lock must not be poisoned")]
    PoisonedLock,

    #[error("node db error: {0}")]
    NodeDb(#[from] NodeDbError),

    #[error("child absent error: no child present in inner node")]
    ChildAbsent,

    #[error("child not found error: children of inner node must exist")]
    ChildNotFound,

    #[error("overflow error")]
    Overflow,
}

impl<T> From<PoisonError<T>> for InnerNodeError {
    fn from(_err: PoisonError<T>) -> Self {
        Self::PoisonedLock
    }
}
