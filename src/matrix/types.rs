//! Helpful Matrix Types
//! Both 2x2 and 3x3 matrices implement a `det()` method which will calculate their determinants.

use super::Matrix;
use raat::structures::algebraic::Field;

/// A regular Square Matrix
pub type SquareMatrix<K, const N: usize> = Matrix<K, N, N>;

/// A 2x2 Matrix
pub type Matrix2<K> = SquareMatrix<K, 2>;
/// A 3x3 Matrix
pub type Matrix3<K> = SquareMatrix<K, 3>;
/// A 4x4 Matrix
pub type Matrix4<K> = SquareMatrix<K, 4>;

/// A single precision floating point matrix
pub type Matrixf32<const N: usize, const M: usize> = Matrix<f32, N, M>;
/// A double precision floating point matrix
pub type Matrixf64<const N: usize, const M: usize> = Matrix<f64, N, M>;

/// A 2x2 float matrix
pub type Matrix2f32 = Matrix2<f32>;
/// A 2x2 double matrix
pub type Matrix2f64 = Matrix2<f64>;
/// A 3x3 float matrix
pub type Matrix3f32 = Matrix3<f32>;
/// A 3x3 double matrix
pub type Matrix3f64 = Matrix3<f64>;
/// A 4x4 float matrix
pub type Matrix4f32 = Matrix4<f32>;
/// A 4x4 double matrix
pub type Matrix4f64 = Matrix4<f64>;

impl<K: Field> Matrix2<K> {
    /// Determinant for 2x2 matrices
    pub fn det(self) -> K {
        self[(0, 1)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }
}

impl<K: Field> Matrix3<K> {
    /// Determinant for 3x3 matrices
    pub fn det(&self) -> K {
        self[(0, 0)] * (self[(1, 1)] * self[(2, 2)] - self[(2, 1)] * self[(1, 2)])
            - self[(0, 1)] * (self[(1, 0)] * self[(2, 2)] - self[(2, 0)] * self[(1, 2)])
            + self[(0, 2)] * (self[(1, 0)] * self[(2, 1)] - self[(2, 0)] * self[(1, 1)])
    }
}
