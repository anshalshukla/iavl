use bon::Builder;

use crate::{node::db::NodeDb, types::U63};

use super::node::ArlockNode;

#[derive(Debug, Builder)]
pub struct ImmutableTree<DB> {
    root: ArlockNode,
    ndb: NodeDb<DB>,
    version: U63,
}

impl<DB> ImmutableTree<DB> {
    pub fn root(&self) -> &ArlockNode {
        &self.root
    }

    pub fn ndb(&self) -> &NodeDb<DB> {
        &self.ndb
    }

    pub fn version(&self) -> U63 {
        self.version
    }
}
