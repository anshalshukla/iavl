use core::error::Error;

use crate::types::NonEmptyBz;

pub trait MutKVStore {
    type Error: Error + Send + Sync + 'static;

    fn insert<K, V>(&self, key: &NonEmptyBz<K>, value: &NonEmptyBz<V>)
    -> Result<bool, Self::Error>;

    fn delete<K>(&self, key: &NonEmptyBz<K>) -> Result<(), Self::Error>;
}

pub trait KVStore {
    type Error: Error + Send + Sync + 'static;

    fn get<K>(&self, key: &NonEmptyBz<K>) -> Result<Option<NonEmptyBz>, Self::Error>;

    fn has<K>(&self, key: &NonEmptyBz<K>) -> Result<bool, Self::Error>;

    fn iterator<K1, K2>(
        &self,
        lo: &NonEmptyBz<K1>,
        hi: &NonEmptyBz<K2>,
    ) -> Result<impl KVIterator, Self::Error>;
}

pub trait KVIterator {
    type Error: Error + Send + Sync + 'static;

    fn next(&mut self) -> Result<Option<(NonEmptyBz, NonEmptyBz)>, Self::Error>;
}
