use std::ops::{Add, Sub, Mul};

use super::NdArray;

fn universal_ops<F: FnMut((&mut f32, &f32)) -> ()>(lhs: &NdArray, rhs: &NdArray, mut ops: F) -> NdArray {
    assert!(NdArray::can_broadcast(&lhs.shape, &rhs.shape), "{:?} cannot be broadcasted by {:?}", lhs.shape, rhs.shape);
    let mut temp = lhs.clone();
    temp.reshape(rhs.shape.iter().fold(vec![-1], |mut s, i| {s.push(*i as i32); s}));
    for k in 0..temp.shape[0] {
        temp[k].iter_mut().zip(rhs.data.iter()).for_each(&mut ops);
    }
    temp.reshape(lhs.shape.clone());
    temp
}


impl PartialEq for NdArray {
    fn eq(&self, other: &Self) -> bool {
        if self.shape == other.shape {
            self.data == other.data
        } else {
            false
        }
    }
}

impl std::ops::Neg for NdArray {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.data.iter_mut().for_each(|i| *i = -*i);
        self
    }
}

impl std::ops::Index<usize> for NdArray {
    type Output = [f32];
    fn index(&self, index: usize) -> &Self::Output {
        let (s, e) = self.shape.iter().skip(1).fold((index, index+1), |s, i| (s.0*i, s.1 * i));
        &self.data[s..e]
    }
}

impl std::ops::IndexMut<usize> for NdArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (s, e) = self.shape.iter().skip(1).fold((index, index+1), |s, i| (s.0*i, s.1 * i));
        &mut self.data[s..e]
    }
}

impl std::ops::Add<&NdArray> for &NdArray {
    type Output = NdArray;
    fn add(self, rhs: &NdArray) -> Self::Output {
        universal_ops(self, rhs, |(a, b)| *a += b) 
    }
}

impl std::ops::Sub<&NdArray> for &NdArray {
    type Output = NdArray;
    fn sub(self, rhs: &NdArray) -> Self::Output {
        universal_ops(self, rhs, |(a, b)| *a -= b) 
    }
}

// multiply-------------------------------
trait NdArrayMultiplyTrait {
    fn multiply(&self, lhs: &NdArray) -> NdArray;
}

impl NdArrayMultiplyTrait for NdArray {

    fn multiply(&self, lhs: &NdArray) -> NdArray {
        assert!(self.dim() >= 2 && lhs.dim() >= 2, "only supports 2d matrix multiplication, but got left = {:?} right = {:?}, please reshape first", lhs.shape, self.shape);
        // check if could multiply
        assert!(lhs.shape[lhs.dim() -1] == self.shape[self.dim() - 2], "got left = {:?} right = {:?}, please check shape", lhs.shape, self.shape);
        // check if could broadcast despite the low-2 dimension shape
        let mut lhs_shape = lhs.shape.clone();
        lhs_shape.pop(); lhs_shape.pop();
        let mut rhs_shape = self.shape.clone();
        rhs_shape.pop(); rhs_shape.pop();
        // lhs.can_broadcast(rhs)


        todo!()
    }
}

impl NdArrayMultiplyTrait for f32 {
    fn multiply(&self, lhs: &NdArray) -> NdArray {
        let mut temp = lhs.clone();
        temp.data.iter_mut().for_each(|i| *i *= self);
        temp
    }
}

impl<T: NdArrayMultiplyTrait> Mul<T> for &NdArray {
    type Output = NdArray;
    fn mul(self, rhs: T) -> Self::Output {
        rhs.multiply(self)
    }
}
// multiply


#[cfg(test)]
mod test {
    use super::NdArray;

    #[test]
    fn test_index() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(vec![2, 3]);
        let mut d = &mut a[0];
        d.iter_mut().for_each(|i| *i = -*i);
        println!("{:?}", a);
    }

    #[test]
    fn test_ops() {
        // add
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        k.reverse();
        let b = NdArray::new(k);
        let c = &a + &b;
        println!("{:?}", c);

        a.reshape(vec![2, 3]);
        let b = NdArray::new(vec![-1.0, -2.0, -3.0]);
        println!("a = {:?}\nc = {:?}", a, b);
        // 0, 0, 0
        // 3 3 3


        let c = &a + &b;
        println!("a + c = {:?}", c);


        // sub
        println!("a - c = {:?}", &a - &b);

        // mul
        println!("a = {:?}", a);
        println!("a * 2.0 = {:?}", &a * 2.0);
    }
}