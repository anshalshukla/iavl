use thiserror::Error;

#[derive(Error, Debug)]
pub enum NodeError {
    #[error("clone leaf node")]
    CloneLeafNode,
    #[error("empty child")]
    EmptyChild,
    #[error("left node key empty")]
    LeftNodeKeyEmpty,
    #[error("right node key empty")]
    RightNodeKeyEmpty,
    #[error("left hash is nil")]
    LeftHashIsNil,
    #[error("right hash is nil")]
    RightHashIsNil,
    #[error("invalid height")]
    InvalidHeight,
    #[error("invalid mode")]
    InvalidMode,
    #[error("invalid nonce")]
    InvalidNonce,
    #[error("decoding error: {0}")]
    DecodingError(&'static str),

    #[error("invalid node version")]
    InvalidNodeVersion,

    #[error("node db error")]
    NodeDBError(String),
}
