use std::ops::{Add, Sub, Mul};

use super::Matrix;

trait MatrixMultiply {
    fn multiply(&self, lhs: &Matrix) -> Matrix;    
}

impl MatrixMultiply for Matrix {
    fn multiply(&self, lhs: &Matrix) -> Matrix {
        assert!(self.shape().0 == lhs.shape().1);
        let mut temp = Matrix::new((lhs.shape().0, self.shape().1));
        
        temp
    }
}

impl<T: MatrixMultiply> Mul<T> for Matrix {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        todo!()        
    }
}

impl Add for Matrix {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape());
        self.params.iter_mut().zip(rhs.params).for_each(|(rowl, rowr)|{
            rowl.iter_mut().zip(rowr).for_each(|(l, r)| *l += r)
        });
        self
    }
}

impl std::ops::Neg for Matrix {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.params.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|ele| *ele = -*ele)
        });
        self
    }
}

impl Sub for Matrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.shape(), other.shape());
        self.params == other.params
    }
}
