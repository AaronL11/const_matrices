/// Helpful Vector Types
pub mod types;
use raat::{
    properties::*,
    structures::{algebraic::*, modules::*, vectorspaces::*},
};
use std::{convert, iter, ops};

#[derive(Copy, Clone, Debug, PartialEq)]
/// A vector, implemented via a static array.
///
/// It holds elements of a field `K` and has a size `N`.
pub struct Vector<K: Field, const N: usize>([K; N]);

impl<K: Field, const N: usize> Default for Vector<K, N> {
    fn default() -> Self {
        Self([K::add_id(); N])
    }
}

// Vector Implementations

impl<K: Field, const N: usize> Vector<K, N> {
    /// Creates a new vector of the additive identity of the field `K`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows for mapping indices to values.
    pub fn from_map<F>(f: F) -> Self
    where
        F: Fn(usize) -> K,
    {
        (0..N).map(f).collect::<Self>()
    }

    /// Returns the size of the vector.
    pub fn size(&self) -> usize {
        N
    }

    /// Returns the length of the vector
    /// similar to `size()`
    pub fn len(&self) -> usize {
        N
    }

    /// Returns an option holding a reference to the item if the index is less than `N`, else `None`.
    pub fn get(&self, idx: usize) -> Option<&K> {
        if idx < N {
            unsafe { Some(self.get_unchecked(idx)) }
        } else {
            None
        }
    }

    /// Returns an option holding a mutable reference to the item if the index is less than `N`, else `None`.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut K> {
        if idx < N {
            unsafe { Some(self.get_unchecked_mut(idx)) }
        } else {
            None
        }
    }

    /// Allows Unsafe indexing into the vector.
    ///
    /// # Safety
    ///
    /// The index must be less than `N`.
    pub unsafe fn get_unchecked(&self, idx: usize) -> &K {
        self.0.get_unchecked(idx)
    }

    /// Allows Unsafe indexing into the vector. Returning a mutable reference.
    ///
    /// # Safety
    ///
    /// The index must be less than `N`.
    pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut K {
        self.0.get_unchecked_mut(idx)
    }
}

// Vector Operations

impl<K: Field, const N: usize> ops::Index<usize> for Vector<K, N> {
    type Output = K;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

impl<K: Field, const N: usize> ops::IndexMut<usize> for Vector<K, N> {
    fn index_mut(&mut self, idx: usize) -> &mut K {
        &mut self.0[idx]
    }
}

impl<K: Field, const N: usize> ops::Neg for Vector<K, N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::from_map(|i| K::add_id() - self[i])
    }
}

impl<K: Field, const N: usize> ops::Add for Vector<K, N> {
    type Output = Self;
    fn add(self, rhs: Vector<K, N>) -> Self {
        Self::from_map(|i| self[i] + rhs[i])
    }
}

impl<K: Field, const N: usize> ops::AddAssign for Vector<K, N> {
    fn add_assign(&mut self, rhs: Vector<K, N>) {
        (0..N).for_each(|i| self[i] += rhs[i])
    }
}

impl<K: Field, const N: usize> ops::Sub for Vector<K, N> {
    type Output = Self;
    fn sub(self, rhs: Vector<K, N>) -> Self {
        Self::from_map(|i| self[i] - rhs[i])
    }
}

impl<K: Field, const N: usize> ops::SubAssign for Vector<K, N> {
    fn sub_assign(&mut self, rhs: Vector<K, N>) {
        (0..N).for_each(|i| self[i] -= rhs[i])
    }
}

impl<K: Field, const N: usize> ops::Mul<K> for Vector<K, N> {
    type Output = Self;
    fn mul(self, k: K) -> Self {
        Self::from_map(|i| self[i] * k)
    }
}

impl<K: Field, const N: usize> ops::MulAssign<K> for Vector<K, N> {
    fn mul_assign(&mut self, k: K) {
        (0..N).for_each(|i| self[i] *= k)
    }
}

impl<K: Field, const N: usize> ops::Mul<Self> for Vector<K, N> {
    type Output = K;
    fn mul(self, rhs: Vector<K, N>) -> K {
        (0..N).fold(K::add_id(), |sum, i| sum + (self[i] * rhs[i]))
    }
}

impl<K: Field, const N: usize> ops::Div<K> for Vector<K, N> {
    type Output = Self;
    fn div(self, k: K) -> Self::Output {
        Self::from_map(|i| self[i] / k)
    }
}

impl<K: Field, const N: usize> ops::DivAssign<K> for Vector<K, N> {
    fn div_assign(&mut self, k: K) {
        (0..N).for_each(|i| self[i] /= k)
    }
}

// Iterator Implementations

impl<K: Field, const N: usize> IntoIterator for Vector<K, N> {
    type Item = K;
    type IntoIter = std::array::IntoIter<K, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<K: Field, const N: usize> Vector<K, N> {
    /// Returns an iterator over the values of the vector.
    pub fn iter(&self) -> std::slice::Iter<K> {
        self.0.iter()
    }
    /// Returns a mutable iterator over the vector.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<K> {
        self.0.iter_mut()
    }
}

impl<K: Field, const N: usize> iter::FromIterator<K> for Vector<K, N> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut vec = [K::add_id(); N];
        iter.into_iter()
            .take(N)
            .enumerate()
            .for_each(|(i, x)| vec[i] = x);
        Self(vec)
    }
}

impl<'a, K: Field, const N: usize> convert::From<&'a [K]> for Vector<K, N> {
    fn from(slice: &[K]) -> Self {
        slice.iter().cloned().collect::<Self>()
    }
}

// Algebra Implementations

macro_rules! algebra {
    ($($tr:ty),*) => ($(impl<K: Field, const N: usize> $tr for Vector<K,N> {})*)
}

impl<K: Field, const N: usize> AddId for Vector<K, N> {
    fn add_id() -> Self {
        Default::default()
    }
}

algebra!(
    Set,
    Magma,
    Monoid,
    Group,
    Module<Vector<K, N>, K>,
    VectorSpace<Vector<K, N>, K, N>
);

macro_rules! float_impls {
    ($($f:ty),*) => ($(
        impl<const N: usize> NormedVectorSpace<Vector<$f,N>,$f,N> for Vector<$f,N> {
            fn norm(&self) -> $f {(*self**self).sqrt()}
        }

        impl<const N: usize> Vector<$f,N> {
            fn mag(&self) -> $f {self.norm()}
            fn hat(&self) -> Self {*self/self.mag()}
        }
    )*)
}

float_impls!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let u = (0i32..42).collect::<Vec<_>>();
        let v = Vector::<i32, 3>::from(&u[1..12]);
        dbg!(v);
    }

    #[test]
    fn test2() {
        let u = (0i32..42).collect::<Vec<_>>();
        let v = Vector::<i32, 12>::from(&u[1..=3]);
        dbg!(v);
    }

    #[test]
    fn test_collect() {
        dbg!(
            (0..12).collect::<Vector<_, 6>>(),
            (0..12).collect::<Vector<_, 12>>(),
            (0..12).collect::<Vector<_, 24>>(),
        );
    }
}
