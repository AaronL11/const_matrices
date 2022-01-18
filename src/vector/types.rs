//! Helpful vector types.
//!
//! Both `Vector2` and `Vector3` implement `ops::Rem` as the cross product operator.

use super::Vector;
use raat::structures::algebraic::Field;
use std::ops;

/// A standard two dimensional vector
pub type Vector2<K> = Vector<K, 2>;
/// A standard three dimensional vector
pub type Vector3<K> = Vector<K, 3>;

/// Single precision floating point vector
pub type Vectorf32<const N: usize> = Vector<f32, N>;
/// Double precision floating point vector
pub type Vectorf64<const N: usize> = Vector<f64, N>;

/// 2D float vector
pub type Vector2f32 = Vector2<f32>;
/// 3D float vector
pub type Vector3f32 = Vector3<f32>;
/// 2D double vector
pub type Vector2f64 = Vector2<f64>;
/// 3D double vector
pub type Vector3f64 = Vector3<f64>;

impl<K: Field> ops::Rem for Vector2<K> {
    type Output = Vector3<K>;
    fn rem(self, rhs: Vector2<K>) -> Vector3<K> {
        Vector([
            K::add_id(),
            K::add_id(),
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

impl<K: Field> ops::Rem for Vector3<K> {
    type Output = Self;
    fn rem(self, rhs: Vector3<K>) -> Self {
        Self([
            self[2] * rhs[1] - self[1] * rhs[2],
            self[0] * rhs[2] - self[2] * rhs[0],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

impl<K: Field> ops::RemAssign for Vector3<K> {
    fn rem_assign(&mut self, rhs: Vector3<K>) {
        *self = *self % rhs
    }
}
