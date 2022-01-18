use raat::structures::{
    algebraic::{Field, Group},
    modules::Module,
    vectorspaces::VectorSpace,
};
use std::ops;

pub trait BLASL1<K, V, U, const N: usize, const M: usize>
where
    U: VectorSpace<U, K, N> + Group + ops::Mul<K> + ops::Div<K>,
    V: VectorSpace<V, K, N> + Group + ops::Mul<K> + ops::Div<K>,
    K: Field,
{
    fn rotg();

    fn rotgm();

    fn rot();

    fn axpy(&mut self, a: K, y: V);

    fn asum(self, y: V) -> K;
}
