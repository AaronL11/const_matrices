#![warn(missing_docs)]
#![allow(dead_code)]

//! This crate is an attempt to implement types for linear algebra using
//! Rust's new const generics feature.

// /// Traits for BLAS and LAPACK implementations
pub mod blapack;
/// Matrices using constant
pub mod matrix;
/// Vector Implementations
pub mod vector;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
