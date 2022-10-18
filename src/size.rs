use std::mem;

/// Trait indicating that an implementer can produce values about it's size.
pub trait Sizeable {
    /// Returns the approximate size of the object in bytes.
    fn get_approx_size(&self) -> usize;
}

impl Sizeable for Vec<u8> {
    fn get_approx_size(&self) -> usize {
        self.len()
    }
}

impl Sizeable for Vec<String> {
    fn get_approx_size(&self) -> usize {
        mem::size_of::<Vec<String>>() + self.iter().map(|s| s.len()).sum::<usize>()
    }
}

#[macro_export]
macro_rules! impl_sizeable {
    ($t:ty) => {
        impl Sizeable for $t {
            fn get_approx_size(&self) -> usize {
                mem::size_of::<$t>()
            }
        }
    };
}

impl_sizeable!(bool);

impl_sizeable!(u8);
impl_sizeable!(u16);
impl_sizeable!(u32);
impl_sizeable!(u64);
impl_sizeable!(u128);
impl_sizeable!(usize);

impl_sizeable!(i8);
impl_sizeable!(i16);
impl_sizeable!(i32);
impl_sizeable!(i64);
impl_sizeable!(i128);
impl_sizeable!(isize);

impl_sizeable!(f32);
impl_sizeable!(f64);

impl_sizeable!(char);
