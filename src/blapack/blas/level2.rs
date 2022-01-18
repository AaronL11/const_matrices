use raat::{
    properties::Lineartransformation,
    structures::{
        algebraic::{Field, Group},
        modules::Module,
        vectorspaces::VectorSpace,
    },
};
use std::ops;

pub trait BLASL2<K, V, U, L, const N: usize, const M: usize>
where
    L: Lineartransformation<K, N, M>,
    U: VectorSpace<U, K, N> + Group + ops::Mul<K> + ops::Div<K>,
    V: VectorSpace<V, K, N> + Group + ops::Mul<K> + ops::Div<K>,
    K: Field,
{
    fn gemv(&mut self,a:K,x:V,A: L,b);

    fn rotgm();

    fn rot();

    fn axpy(&mut self, a: K, y: V);

    fn asum(self, y: V) -> K;
}
