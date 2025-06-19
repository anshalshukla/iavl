use rocksdb::{DB, DBIterator, Direction, Error, IteratorMode, Options, WriteBatch, WriteOptions};
use std::{iter::Peekable, path::Path, sync::Arc};

use crate::{
    db::errors::BatchError,
    node_db::{Batch, Iterator},
};

pub struct RocksDB {
    db: DB,
}

impl RocksDB {
    pub fn open(path: &str) -> Result<Self, Error> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        Ok(Self {
            db: DB::open(&opts, path)?,
        })
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.db.get(key)?)
    }

    pub fn has(&self, key: &[u8]) -> Result<bool, Error> {
        Ok(self.db.get(key)?.is_some())
    }

    pub fn set(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        Ok(self.db.put(key, value)?)
    }

    pub fn set_sync(&self, key: &[u8], value: &[u8]) -> Result<(), Error> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        Ok(self.db.put_opt(key, value, &write_opts)?)
    }

    pub fn delete(&self, key: &[u8]) -> Result<(), Error> {
        Ok(self.db.delete(key)?)
    }

    pub fn delete_sync(&self, key: &[u8]) -> Result<(), Error> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(true);
        Ok(self.db.delete_opt(key, &write_opts)?)
    }

    pub fn new_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }

    pub fn write_batch(&self, batch: WriteBatch, sync: bool) -> Result<(), Error> {
        let mut write_opts = WriteOptions::default();
        write_opts.set_sync(sync);
        Ok(self.db.write_opt(batch, &write_opts)?)
    }

    pub fn iterator(&self, start: Option<Vec<u8>>, end: Option<Vec<u8>>) -> impl Iterator {
        let mut iter = match &start {
            Some(key) => self
                .db
                .iterator(IteratorMode::From(&key, Direction::Forward)),
            None => self.db.iterator(IteratorMode::Start),
        };

        RocksDBIterator::new(iter, start, end, Direction::Forward)
    }

    pub fn reverse_iterator(&self, start: Option<Vec<u8>>, end: Option<Vec<u8>>) -> impl Iterator {
        let mut iter = match &end {
            Some(key) => self
                .db
                .iterator(IteratorMode::From(&key, Direction::Reverse)),
            None => self.db.iterator(IteratorMode::End),
        };

        RocksDBIterator::new(iter, start, end, Direction::Reverse)
    }

    pub fn close(self) {
        drop(self);
    }
}

pub struct RocksDBIterator<'a> {
    source: Peekable<DBIterator<'a>>,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    direction: Direction,
    is_invalid: bool,
}

impl<'a> RocksDBIterator<'a> {
    pub fn new(
        iter: DBIterator<'a>,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
        direction: Direction,
    ) -> Self {
        Self {
            source: iter.peekable(),
            start,
            end,
            direction,
            is_invalid: false,
        }
    }

    pub fn peek(&mut self) -> Option<&Result<(Box<[u8]>, Box<[u8]>), Error>> {
        self.source.peek()
    }

    fn assert_valid(&mut self) {
        if !self.valid() {
            panic!("iterator is invalid");
        }
    }
}

impl<'a> Iterator for RocksDBIterator<'a> {
    type Error = Error;

    fn valid(&mut self) -> bool {
        if self.is_invalid {
            return false;
        }

        // Check if we have a valid next item
        let peek_result = match self.source.peek() {
            Some(Ok(pair)) => pair,
            _ => {
                self.is_invalid = true;
                return false;
            }
        };

        // Get the key from the peek result
        let key = &peek_result.0;

        // Compare based on direction
        match self.direction {
            Direction::Forward => {
                if let Some(end) = &self.end {
                    if key.as_ref() >= end.as_slice() {
                        self.is_invalid = true;
                        return false;
                    }
                }
            }
            Direction::Reverse => {
                if let Some(start) = &self.start {
                    if key.as_ref() < start.as_slice() {
                        self.is_invalid = true;
                        return false;
                    }
                }
            }
        }

        true
    }

    fn key(&mut self) -> Option<Vec<u8>> {
        self.assert_valid();
        match self.source.peek() {
            Some(Ok((key, _))) => Some(key.to_vec()),
            _ => None,
        }
    }

    fn value(&mut self) -> Option<Vec<u8>> {
        self.assert_valid();
        match self.source.peek() {
            Some(Ok((_, value))) => Some(value.to_vec()),
            _ => None,
        }
    }

    fn next(&mut self) -> Option<Result<(Box<[u8]>, Box<[u8]>), Error>> {
        self.assert_valid();

        // TODO: Confirms that the iterator is valid for reverse mode
        self.source.next()
    }

    fn domain(&self) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), Error> {
        Ok((self.start.clone(), self.end.clone()))
    }
}

pub struct RocksDBBatch {
    db: Arc<DB>,
    batch: Option<WriteBatch>,
}

impl RocksDBBatch {
    pub fn newRocksDBBatch(db: Arc<DB>) -> Self {
        Self {
            db,
            batch: Some(WriteBatch::default()),
        }
    }

    pub fn set(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<(), BatchError> {
        if key.is_empty() {
            return Err(BatchError::KeyEmpty);
        }
        if value.is_empty() {
            return Err(BatchError::ValueNil);
        }
        if let Some(batch) = &mut self.batch {
            batch.put(key, value);
            Ok(())
        } else {
            Err(BatchError::BatchNil)
        }
    }

    pub fn delete(&mut self, key: Vec<u8>) -> Result<(), BatchError> {
        if key.is_empty() {
            return Err(BatchError::KeyEmpty);
        }
        if let Some(batch) = &mut self.batch {
            batch.delete(key);
            Ok(())
        } else {
            Err(BatchError::BatchNil)
        }
    }

    pub fn write(&mut self) -> Result<(), BatchError> {
        self.write_inner(false)
    }

    pub fn write_sync(&mut self) -> Result<(), BatchError> {
        self.write_inner(true)
    }

    fn write_inner(&mut self, sync: bool) -> Result<(), BatchError> {
        // TODO: check if this is correct, as it moves the batch out of the struct
        match self.batch.take() {
            Some(batch) => {
                let mut opts = WriteOptions::default();
                opts.set_sync(sync);
                self.db.write_opt(batch, &opts)?;
                self.close();
                Ok(())
            }
            None => Err(BatchError::BatchNil),
        }
    }

    pub fn close(&mut self) {
        if let Some(mut batch) = self.batch.take() {
            batch.clear(); // not strictly needed
        }
    }

    pub fn get_byte_size(&self) -> Result<usize, BatchError> {
        if let Some(batch) = &self.batch {
            Ok(batch.data().len())
        } else {
            Err(BatchError::BatchNil)
        }
    }
}
