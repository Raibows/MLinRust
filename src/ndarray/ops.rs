use std::{ops::{Add, Sub, Mul}, vec};

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

macro_rules! zip {
    ($x: expr) => ($x);
    ($x: expr, $($y: expr), +) => (
        $x.iter().zip(
            zip!($($y), +))
    )
}

trait NdArrayMultiplyTrait {
    fn multiply(&mut self, lhs: &mut NdArray) -> NdArray;
}

impl NdArrayMultiplyTrait for &mut NdArray {

    fn multiply(&mut self, lhs: &mut NdArray) -> NdArray {
        assert!(self.dim() >= 2 && lhs.dim() >= 2, "only supports 2d matrix multiplication, but got left = {:?} right = {:?}, please reshape first", lhs.shape, self.shape);

        // check if could multiply
        assert!(lhs.shape[lhs.dim() -1] == self.shape[self.dim() - 2], "got left = {:?} right = {:?}, please check shape", lhs.shape, self.shape);

        // check if could broadcast despite the low-2 dimension shape
        let mut lhs_shape = lhs.shape.clone();
        lhs_shape.pop(); lhs_shape.pop();
        let mut rhs_shape = self.shape.clone();
        rhs_shape.pop(); rhs_shape.pop();
        assert!(NdArray::can_broadcast(&lhs_shape, &rhs_shape), "{:?} cannot be broadcasted by {:?}", lhs_shape, rhs_shape);

        // prepare the target
        let mut target_shape = lhs_shape.clone();
        let target_matrix_shape = vec![lhs.shape[lhs.dim() - 2], self.shape[self.dim() - 1]];
        target_shape.extend(target_matrix_shape.iter());
        let mut target = NdArray::new(vec![0.0; NdArray::total_num(&target_shape)]);
        
        // nest to the temporary shape for broadcast
        target.reshape(target_matrix_shape.iter().fold(vec![-1], |mut s, i| {s.push(*i as i32); s}));

        let ori_left_shape = lhs.shape.clone();
        let ori_right_shape = self.shape.clone();

        let left_matrix_shape = vec![lhs.shape[lhs.dim() - 2], lhs.shape[lhs.dim() - 1]];
        let right_matrix_shape = vec![self.shape[self.dim() - 2], self.shape[self.dim() - 1]];

        lhs.reshape(left_matrix_shape.iter().fold(vec![-1], |mut s, i| {s.push(*i as i32); s}));
        self.reshape(right_matrix_shape.iter().fold(vec![-1], |mut s, i| {s.push(*i as i32); s}));

        // ready to take multiplication
        let left_iters: Vec<usize> = (0..lhs.shape[0]).collect();
        let right_iters: Vec<usize> = (0..self.shape[0]).collect();
        let target_iters: Vec<usize> = (0..target.shape[0]).collect();
        assert!(left_iters.len() == target_iters.len());

        for (l, r) in left_iters.iter().zip(right_iters.iter().cycle()) { 
            let lm = &lhs[*l];
            let rm = &self[*r];
            let t = &mut target[*l];

            // println!("lm {} rm {} t {}", lm.len(), rm.len(), t.len());

            for j in 0..right_matrix_shape[1] {
                for li in 0..left_matrix_shape[0] {
                    let mut res = 0.0;
                    for i in 0..right_matrix_shape[0] {
                        res += lm[li * left_matrix_shape[1] + i] * rm[i * right_matrix_shape[1] + j];
                    }
                    t[li * target_matrix_shape[1] + j] = res;
                }
            }
        }

        // restore to the original shape
        lhs.reshape(ori_left_shape);
        self.reshape(ori_right_shape);
        target.reshape(target_shape);

        target    
    }
}

impl NdArrayMultiplyTrait for f32 {
    fn multiply(&mut self, lhs: &mut NdArray) -> NdArray {
        let mut temp = lhs.clone();
        temp.data.iter_mut().for_each(|i| *i *= *self);
        temp
    }
}

impl<T: NdArrayMultiplyTrait> Mul<T> for &mut NdArray {
    type Output = NdArray;
    fn mul(self, mut rhs: T) -> Self::Output {
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

        // mul float
        println!("a = {:?}", a);
        println!("a * 2.0 = {:?}", &mut a * 2.0);

        // mul 2d matrix
        let mut b = NdArray::new(vec![-1.0, -2.0, -3.0]);
        b.reshape(vec![3, 1]);
        println!("a = {:?}\nb = {:?}", a, b);
        let c =  &mut a * &mut b;
        println!("a * b = {:?}", c);
        let mut cc = NdArray::new(vec![-14.0, -32.0]);
        cc.reshape(vec![2, 1]);
        assert_eq!(c, cc);

        // mul broadcast
        // a = 2 x 3 x 2
        // b = 2 x 1
        let a: Vec<f32> = (0..12).into_iter().map(|i| i as f32).collect();
        let mut a = NdArray::new(a);
        a.reshape(vec![2, 3, 2]);
        let mut b = NdArray::new(vec![0.0, -1.0]);
        b.reshape(vec![2, 1]);

        println!("a = {:?}\nb = {:?}", a, b);
        println!("c = a * b = {:?}", &mut a * &mut b);
        let mut cc = NdArray::new(vec![-1.0, -3.0, -5.0, -7.0, -9.0, -11.0]);
        cc.reshape(vec![2, 3, 1]);
        assert_eq!(cc, &mut a * &mut b);

        // mul broadcast
        // a = 1 x 2 x 3 x 2
        // b = 2 x 2 x 1
        let a: Vec<f32> = (0..12).into_iter().map(|i| i as f32).collect();
        let mut a = NdArray::new(a);
        a.reshape(vec![1, 2, 3, 2]);
        let mut b = NdArray::new(vec![0.0, -1.0, 1.0, 0.0]);
        b.reshape(vec![2, 2, 1]);

        println!("a = {:?}\nb = {:?}", a, b);
        println!("c = a * b = {:?}", &mut a * &mut b);
        let mut cc = NdArray::new(vec![-1.0, -3.0, -5.0, 6.0, 8.0, 10.0]);
        cc.reshape(vec![1, 2, 3, 1]);
        assert_eq!(cc, &mut a * &mut b);

    }
}