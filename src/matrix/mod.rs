/// Types for square matrices
pub mod types;
use super::vector::Vector;
use raat::{
    properties::*,
    structures::{algebraic::*, modules::*},
};
use std::{iter, ops};
use types::SquareMatrix;

#[derive(Debug, Copy, Clone, PartialEq)]
/// A constant sized matrix, implemented via a static array.
///
/// It can be built from any field `K`, with `N` indicating the number of rows, and `M` indicating
/// the number of columns.
pub struct Matrix<K: Field, const N: usize, const M: usize>([[K; M]; N]);

impl<K: Field, const N: usize, const M: usize> Default for Matrix<K, N, M> {
    fn default() -> Self {
        Self([[K::add_id(); M]; N])
    }
}

impl<K: Field, const N: usize, const M: usize> Matrix<K, N, M> {
    /// Creates a new matrix filled with the additive identity of `K`.
    ///
    /// For numbers this is equivalent to a zero matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows the programmer to write a way to map individual indices to values.
    ///
    /// For example, a basic identity matrix could be defined as:
    /// ```rust
    /// # use types::Matrix4f64;
    /// # fn main() {
    /// let id = Matrix4f64::from_map(|i,j| if i==j { 1. } else { 0. });
    /// # }
    /// ```
    pub fn from_map<F>(f: F) -> Self
    where
        F: Fn(usize, usize) -> K,
    {
        let mut mat = [[K::add_id(); M]; N];
        (0..N).for_each(|i| (0..M).for_each(|j| mat[i][j] = f(i, j)));
        Self(mat)
    }

    /// Returns the identity matrix for the given size
    /// ```rust
    /// let id = Matrix4f64::id();
    /// let id2 = Matrix4f64::from([
    ///     [1.,0.,0.,0.],
    ///     [0.,1.,0.,0.],
    ///     [0.,0.,1.,0.],
    ///     [0.,0.,0.,1.],
    /// ]);
    /// assert_eq!(id,id2);
    /// ```
    pub fn id() -> Self {
        Self::from_map(|i, j| if i == j { K::mul_id() } else { K::add_id() })
    }

    /// Returns the size of the matrix as a tuple indicating `(rows, columns)`.
    /// ```rust
    /// let A = SquareMatrix<f64,5>::new();
    /// let B = Matrixf64<4,6>::new();
    /// assert_eq!(A.size(),(5,5));
    /// assert_eq!(B.size(),(4,6));
    /// ```
    pub fn size(&self) -> (usize, usize) {
        (N, M)
    }

    /// Returns the number of rows of a matrix.
    /// ```rust
    /// let A = SquareMatrix<f64,5>::new();
    /// let B = Matrixf64<4,6>::new();
    /// assert_eq!(A.rows(),5);
    /// assert_eq!(B.rows(),4);
    /// ```
    pub fn rows(&self) -> usize {
        N
    }

    /// Returns the number of columns of a matrix.
    /// ```rust
    /// let A = SquareMatrix<f64,5>::new();
    /// let B = Matrixf64<4,6>::new();
    /// assert_eq!(A.cols(),5);
    /// assert_eq!(B.cols(),6);
    /// ```
    pub fn cols(&self) -> usize {
        M
    }

    /// Iterator over the *i*-th row:
    /// ```rust
    /// let mat = Matrix4f64::id();
    /// let row = mat.row(3).collect::<Vec<_>>();
    /// assert_eq!(row, [0.,0.,0.,1.]);
    /// ```
    pub fn row(&self, idx: usize) -> impl Iterator<Item = &K> + '_ {
        self.row_iter().skip(N * idx).take(M)
    }

    /// Iterator over the *j*-th column:
    /// ```rust
    /// let mat = Matrix4f64::id();
    /// let col = mat.col(3).collect::<Vec<_>>();
    /// assert_eq!(col, [0.,0.,0.,1.]);
    /// ```
    pub fn col(&self, idx: usize) -> impl Iterator<Item = &K> + '_ {
        self.col_iter().skip(M * idx).take(N)
    }

    /// The transpose of the matrix
    fn t(&self) -> Matrix<K, M, N> {
        self.col_iter().cloned().collect::<Matrix<K, M, N>>()
    }

    /// Returns an option if there is an item at the index `(i,j)`.
    /// Similar to the standard library's `get()` function.
    pub fn get(&self, i: usize, j: usize) -> Option<&K> {
        self.0.get(i)?.get(j)
    }

    /// Unsafely returns the item at index `(i,j)`.
    /// Similar to the standard library's `get_unchecked()` function.
    ///
    /// # Safety
    ///
    /// `i` cannot be greater than `N` and `j` cannot be greater than `M`.
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &K {
        self.0.get_unchecked(i).get_unchecked(j)
    }

    /// Returns an option if there is an item at the index `(i,j)`.
    /// Similar to the standard library's `get_mut()` function.
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut K> {
        self.0.get_mut(i)?.get_mut(j)
    }

    /// Unsafely returns the item at index `(i,j)`.
    /// Similar to the standard library's `get_unchecked_mut()` function.
    ///
    /// # Safety
    ///
    /// `i` cannot be greater than `N` and `j` cannot be greater than `M`.
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &K {
        self.0.get_unchecked_mut(i).get_unchecked_mut(j)
    }
}

// Matrix Operations

impl<K: Field, const N: usize, const M: usize> ops::Index<(usize, usize)> for Matrix<K, N, M> {
    type Output = K;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.0[i][j]
    }
}

impl<K: Field, const N: usize, const M: usize> ops::IndexMut<(usize, usize)> for Matrix<K, N, M> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut K {
        &mut self.0[i][j]
    }
}

impl<K: Field, const N: usize, const M: usize> ops::Add for Matrix<K, N, M> {
    type Output = Self;
    /// Add two matrices together
    fn add(self, rhs: Matrix<K, N, M>) -> Self::Output {
        Self::from_map(|i, j| self[(i, j)] + rhs[(i, j)])
    }
}

impl<K: Field, const N: usize, const M: usize> ops::AddAssign for Matrix<K, N, M> {
    fn add_assign(&mut self, rhs: Matrix<K, N, M>) {
        *self = *self + rhs
    }
}

impl<K: Field, const N: usize, const M: usize> ops::Neg for Matrix<K, N, M> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::from_map(|i, j| K::add_id() - self[(i, j)])
    }
}

impl<K: Field, const N: usize, const M: usize> ops::Sub for Matrix<K, N, M> {
    type Output = Self;
    fn sub(self, rhs: Matrix<K, N, M>) -> Self::Output {
        Self::from_map(|i, j| self[(i, j)] - rhs[(i, j)])
    }
}

impl<K: Field, const N: usize, const M: usize> ops::SubAssign for Matrix<K, N, M> {
    fn sub_assign(&mut self, rhs: Matrix<K, N, M>) {
        *self = *self - rhs
    }
}

// Scalar Multiplication

impl<K: Field, const N: usize, const M: usize> ops::Mul<K> for Matrix<K, N, M> {
    type Output = Self;
    fn mul(self, k: K) -> Self::Output {
        Self::from_map(|i, j| self[(i, j)] * k)
    }
}

impl<K: Field, const N: usize, const M: usize> ops::MulAssign<K> for Matrix<K, N, M> {
    fn mul_assign(&mut self, k: K) {
        *self = *self * k
    }
}

impl<K: Field, const N: usize, const M: usize> ops::Div<K> for Matrix<K, N, M> {
    type Output = Self;
    fn div(self, k: K) -> Self::Output {
        Self::from_map(|i, j| self[(i, j)] / k)
    }
}

impl<K: Field, const N: usize, const M: usize> ops::DivAssign<K> for Matrix<K, N, M> {
    fn div_assign(&mut self, k: K) {
        *self = *self / k
    }
}

// Vector Multiplication

impl<K: Field, const N: usize, const M: usize> ops::Mul<Vector<K, M>> for Matrix<K, N, M> {
    type Output = Vector<K, N>;
    fn mul(self, rhs: Vector<K, M>) -> Self::Output {
        (0..N)
            .map(|i| {
                self.row(i)
                    .zip(rhs.iter())
                    .fold(K::add_id(), |sum, (&x, &y)| sum + x * y)
            })
            .collect::<Vector<K, N>>()
    }
}

impl<K: Field, const N: usize> ops::MulAssign<SquareMatrix<K, N>> for Vector<K, N> {
    fn mul_assign(&mut self, a: SquareMatrix<K, N>) {
        *self = a * *self
    }
}

// Matrix Multiplication

impl<K: Field, const N: usize, const P: usize, const Q: usize> ops::Mul<Matrix<K, N, Q>>
    for Matrix<K, P, N>
{
    type Output = Matrix<K, P, Q>;
    fn mul(self, rhs: Matrix<K, N, Q>) -> Self::Output {
        Matrix::from_map(|i, j| {
            self.row(i)
                .zip(rhs.col(j))
                .fold(K::add_id(), |sum, (&x, &y)| sum + x * y)
        })
    }
}

impl<K: Field, const N: usize> ops::MulAssign for SquareMatrix<K, N> {
    fn mul_assign(&mut self, rhs: SquareMatrix<K, N>) {
        *self = *self * rhs
    }
}

// Iterator Implementations

/// An iterator over the rows of the matrix:
/// ```rust
/// # fn main() {
/// let id = Matrix3f64::id();
/// let rows = id.row_iter().collect::<Vec<_>>();
/// assert_eq!(
///     rows,
///     [1.,0.,0.,
///      0.,1.,0.,
///      0.,0.,1.]);
/// # }
/// ```
pub struct RowIter<'a, K: Field, const N: usize, const M: usize>(&'a Matrix<K, N, M>, usize);

impl<'a, K: Field, const N: usize, const M: usize> RowIter<'a, K, N, M> {
    fn new(mat: &'a Matrix<K, N, M>) -> Self {
        Self(mat, 0)
    }
}

impl<'a, K: Field, const N: usize, const M: usize> Iterator for RowIter<'a, K, N, M> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        let r = self.0.get(self.1 / N, self.1 % M);
        self.1 += 1;
        r
    }
}

/// An iterator over the columns of a matrix:
/// ```rust
/// # fn main() {
/// let mat = Matrix3f64::from([
///     [1.,4.,7.],
///     [2.,5.,8.],
///     [3.,6.,9.]
///     ]);
/// let cols = mat.col_iter().collect::<Vec<_>>();
/// assert_eq!(
///     cols,
///     [1.,2.,3.,
///      4.,5.,6.,
///      7.,8.,9.]);
/// # }
/// ```
pub struct ColIter<'a, K: Field, const N: usize, const M: usize>(&'a Matrix<K, N, M>, usize);

impl<'a, K: Field, const N: usize, const M: usize> ColIter<'a, K, N, M> {
    fn new(mat: &'a Matrix<K, N, M>) -> Self {
        Self(mat, 0)
    }
}

impl<'a, K: Field, const N: usize, const M: usize> Iterator for ColIter<'a, K, N, M> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        let r = self.0.get(self.1 % N, self.1 / M);
        self.1 += 1;
        r
    }
}

impl<K: Field, const N: usize, const M: usize> Matrix<K, N, M> {
    /// Returns an iterator over the rows of the matrix.
    pub fn row_iter(&self) -> RowIter<K, N, M> {
        RowIter::new(self)
    }

    /// Returns an iterator over the columns of the matrix.
    pub fn col_iter(&self) -> ColIter<K, N, M> {
        ColIter::new(self)
    }
}

// From implementations

impl<K: Field, const N: usize, const M: usize> iter::FromIterator<K> for Matrix<K, N, M> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut mat = [[K::add_id(); M]; N];
        let mut iter = iter.into_iter();
        for i in 0..N {
            for j in 0..M {
                if let Some(x) = iter.next() {
                    mat[i][j] = x;
                } else {
                    break;
                }
            }
        }
        Self(mat)
    }
}

impl<K: Field, const N: usize, const M: usize> From<[[K; M]; N]> for Matrix<K, N, M> {
    fn from(mat: [[K; M]; N]) -> Self {
        Self::from_map(|i, j| mat[i][j])
    }
}

// algebra implementations

macro_rules! algebra {
    ($($tr:ty),*) => ($(impl<K: Field, const N: usize, const M: usize> $tr for Matrix<K,N,M> {})*)
}

macro_rules! algebra_square {
    ($($tr:ty),*) => ($(impl<K: Field, const N: usize> $tr for SquareMatrix<K,N> {})*)
}

impl<K: Field, const N: usize, const M: usize> AddId for Matrix<K, N, M> {
    fn add_id() -> Self {
        Default::default()
    }
}

impl<K: Field, const N: usize> MulId for SquareMatrix<K, N> {
    fn mul_id() -> Self {
        Self::id()
    }
}

algebra!(
    Set,
    Magma,
    Monoid,
    Group,
    Module<Matrix<K, N, M>, K>,
    // VectorSpace<Matrix<K, N, M>, K, { N * M }>,
    LinearTransformation<Vector<K, N>, Vector<K, M>, K, N, M>
);

algebra_square!(Ring);
