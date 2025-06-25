use crate::types::{NonEmptyBz, U31, U63};

use super::NodeHash;

pub type Drafted = Drafter<NonEmptyBz>;

pub type Hashed<HAUX = ()> = Drafter<NonEmptyBz, Hasher<U63, NodeHash, HAUX>>;

pub type Saved<HAUX = (), SAUX = ()> =
    Drafter<NonEmptyBz, Hasher<U63, NodeHash, HAUX, Saver<U31, SAUX>>>;

#[derive(Debug)]
pub struct Drafter<K, STAGE = ()> {
    key: K,
    stage: STAGE,
}

#[derive(Debug)]
pub struct Hasher<VERSION, HASH, HAUX = (), STATUS = ()> {
    version: VERSION,
    hash: HASH,
    haux: HAUX,
    status: STATUS,
}

#[derive(Debug)]
pub struct Saver<NONCE, SAUX = ()> {
    nonce: NONCE,
    saux: SAUX,
}

impl<K, STAGE> Drafter<K, STAGE> {
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<K, VERSION, HASH, HAUX, STATUS> Drafter<K, Hasher<VERSION, HASH, HAUX, STATUS>> {
    pub fn version(&self) -> &VERSION {
        self.stage.version()
    }

    pub fn hash(&self) -> &HASH {
        self.stage.hash()
    }

    pub fn haux(&self) -> &HAUX {
        self.stage.haux()
    }
}

impl<K, VERSION, HASH, NONCE, HAUX, SAUX>
    Drafter<K, Hasher<VERSION, HASH, HAUX, Saver<NONCE, SAUX>>>
{
    pub fn nonce(&self) -> &NONCE {
        self.stage.nonce()
    }

    pub fn saux(&self) -> &SAUX {
        self.stage.saux()
    }
}

impl<K> Drafter<K> {
    pub fn new(key: K) -> Self {
        Self { key, stage: () }
    }

    pub fn into_hashed<VERSION, HASH, HAUX>(
        self,
        version: VERSION,
        hash: HASH,
        haux: HAUX,
    ) -> Drafter<K, Hasher<VERSION, HASH, HAUX>> {
        Drafter {
            key: self.key,
            stage: Hasher::new(version, hash, haux),
        }
    }
}

impl<K, VERSION, HASH, HAUX> Drafter<K, Hasher<VERSION, HASH, HAUX>> {
    pub fn into_saved<NONCE, SAUX>(
        self,
        nonce: NONCE,
        saux: SAUX,
    ) -> Drafter<K, Hasher<VERSION, HASH, HAUX, Saver<NONCE, SAUX>>> {
        Drafter {
            key: self.key,
            stage: self.stage.into_saved(nonce, saux),
        }
    }
}

impl<VERSION, HASH, HAUX, STATUS> Hasher<VERSION, HASH, HAUX, STATUS> {
    pub fn version(&self) -> &VERSION {
        &self.version
    }

    pub fn hash(&self) -> &HASH {
        &self.hash
    }

    pub fn haux(&self) -> &HAUX {
        &self.haux
    }
}

impl<VERSION, HASH, NONCE, HAUX, SAUX> Hasher<VERSION, HASH, HAUX, Saver<NONCE, SAUX>> {
    pub fn nonce(&self) -> &NONCE {
        self.status.nonce()
    }

    pub fn saux(&self) -> &SAUX {
        self.status.saux()
    }
}

impl<VERSION, HASH, HAUX> Hasher<VERSION, HASH, HAUX> {
    pub fn new(version: VERSION, hash: HASH, haux: HAUX) -> Self {
        Self {
            version,
            hash,
            haux,
            status: (),
        }
    }

    pub fn into_saved<NONCE, SAUX>(
        self,
        nonce: NONCE,
        saux: SAUX,
    ) -> Hasher<VERSION, HASH, HAUX, Saver<NONCE, SAUX>> {
        Hasher {
            version: self.version,
            hash: self.hash,
            haux: self.haux,
            status: Saver::new(nonce, saux),
        }
    }
}

impl<NONCE, SAUX> Saver<NONCE, SAUX> {
    pub fn new(nonce: NONCE, saux: SAUX) -> Self {
        Self { nonce, saux }
    }

    pub fn nonce(&self) -> &NONCE {
        &self.nonce
    }

    pub fn saux(&self) -> &SAUX {
        &self.saux
    }
}
