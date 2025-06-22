use std::{cmp::Ordering, ops::Sub};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct BoundedUint<U, I> {
    value: U,
    _phantom: std::marker::PhantomData<I>,
}

pub trait BoundedUintTrait<U, I> {
    fn new(x: U) -> Result<BoundedUint<U, I>, BoundedUintError>;
    fn as_signed(&self) -> I;
    fn to_be_bytes(&self) -> Vec<u8>;
    fn cmp(&self, x: BoundedUint<U, I>) -> Ordering;
    fn get(&self) -> U;
}

#[derive(Error, Debug)]
pub enum BoundedUintError {
    #[error("unsigned int exceeds the value of signed integer")]
    ExceedsSignedInteger,
}

macro_rules! impl_bounded_uint {
    ($u:ty, $i:ty) => {
        impl BoundedUintTrait<$u, $i> for BoundedUint<$u, $i> {
            fn new(x: $u) -> Result<Self, BoundedUintError> {
                if x > <$i>::MAX as $u {
                    return Err(BoundedUintError::ExceedsSignedInteger);
                }
                Ok(BoundedUint {
                    value: x,
                    _phantom: std::marker::PhantomData,
                })
            }

            fn as_signed(&self) -> $i {
                self.value as $i
            }

            fn to_be_bytes(&self) -> Vec<u8> {
                self.value.to_be_bytes().to_vec()
            }

            fn cmp(&self, x: BoundedUint<$u, $i>) -> Ordering {
                self.value.cmp(&x.value)
            }

            fn get(&self) -> $u {
                self.value
            }
        }
    };
}

impl_bounded_uint!(u8, i8);
impl_bounded_uint!(u32, i32);
impl_bounded_uint!(u64, i64);

// Type aliases for convenience
// Specially designed data structures to enfore compile time check on fields like (subtree_height, version, nonce)
// which are meant to be uint but are take int just to enable zigzag encoding
pub type U7 = BoundedUint<u8, i8>;
pub type U31 = BoundedUint<u32, i32>;
pub type U63 = BoundedUint<u64, i64>;

impl U7 {
    pub fn zero() -> Self {
        BoundedUint::<u8, i8>::new(0).unwrap()
    }
}

impl U31 {
    pub fn zero() -> Self {
        BoundedUint::<u32, i32>::new(0).unwrap()
    }
}

impl U63 {
    pub fn zero() -> Self {
        BoundedUint::<u64, i64>::new(0).unwrap()
    }
}

impl Sub<U63> for U63 {
    type Output = i64;

    fn sub(self, other: Self) -> Self::Output {
        // Secure subtraction with overflow checking
        // Since U63 guarantees values are <= i64::MAX, we can safely convert to i64
        let left = self.value as i64;
        let right = other.value as i64;

        // Use checked_sub to detect overflow
        match left.checked_sub(right) {
            Some(result) => result,
            None => {
                // Handle overflow - in this case, the result would be less than i64::MIN
                // Since we're subtracting unsigned values that fit in i64, this should be rare
                // but we'll return i64::MIN as a saturated result
                i64::MIN
            }
        }
    }
}
