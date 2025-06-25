use bytes::Bytes;

macro_rules! define_ux {
    ($name:ident, $unsigned:ty, $signed:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name($unsigned);

        impl $name {
            pub const MIN: Self = Self(<$unsigned>::MIN);
            pub const MAX: Self = Self(Self::INNER_MAX);
            pub const ONE: Self = Self(1);
            pub const TWO: Self = Self(2);

            const INNER_MAX: $unsigned = <$signed>::MAX as $unsigned;

            pub const fn new(x: $unsigned) -> Option<Self> {
                if x <= Self::INNER_MAX {
                    Some(Self(x))
                } else {
                    None
                }
            }

            pub const fn from_signed(x: $signed) -> Option<Self> {
                if x < 0 {
                    None
                } else {
                    Self::new(x as $unsigned)
                }
            }

            pub const fn get(self) -> $unsigned {
                self.0
            }

            pub const fn to_signed(self) -> $signed {
                self.get() as $signed
            }
        }
    };
}

define_ux!(U7, u8, i8);
define_ux!(U31, u32, i32);
define_ux!(U63, u64, i64);

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NonEmptyBz<T = Bytes>(T);

impl<T> NonEmptyBz<T> {
    /// If 'bz' is `[u8; N]` or `&[u8; N]`, consider using `NonEmptyBz::from_#_array()` constructors
    pub fn new(bz: T) -> Option<Self>
    where
        T: AsRef<[u8]>,
    {
        (!bz.as_ref().is_empty()).then_some(Self(bz))
    }

    pub const fn get(&self) -> &T {
        &self.0
    }

    pub fn len(&self) -> usize
    where
        T: AsRef<[u8]>,
    {
        self.0.as_ref().len()
    }

    pub fn is_empty(&self) -> bool
    where
        T: AsRef<[u8]>,
    {
        self.0.as_ref().is_empty()
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<const N: usize> NonEmptyBz<[u8; N]> {
    pub const fn from_owned_array(bz: [u8; N]) -> Self {
        const { assert!(N != 0) }
        Self(bz)
    }
}

impl<'a, const N: usize> NonEmptyBz<&'a [u8; N]> {
    pub const fn from_borrowed_array(bz: &'a [u8; N]) -> Self {
        const { assert!(N != 0) }
        Self(bz)
    }
}
