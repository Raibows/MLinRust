use super::NdArray;

fn universal_ops<F: FnMut((&mut f32, &f32)) -> ()>(lhs: &NdArray, rhs: &NdArray, mut ops: F) -> NdArray {
    assert!(NdArray::can_broadcast(&lhs.shape, &rhs.shape), "{:?} cannot be broadcasted by {:?}", lhs.shape, rhs.shape);
    let mut temp = lhs.clone();
    temp.reshape(rhs.shape.iter().fold(vec![-1], |mut s, i| {s.push(*i as i32); s}));

    let base_sizes = NdArray::index_base_sizes(&temp.shape);

    temp.data.chunks_exact_mut(base_sizes[0]).for_each(|same_as_rhs| {
        same_as_rhs.iter_mut().zip(rhs.data.iter()).for_each(&mut ops);
    });
    
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

impl std::ops::Div<f32> for &NdArray {
    type Output = NdArray;
    fn div(self, rhs: f32) -> Self::Output {
        // avoid overflow
        let rhs = f32::max(1e-6, rhs); 
        (1.0 / rhs).multiply(self)
    }
}

// multiply-------------------------------
/* some naive versions of Matrix Multiplication
// naive version of multiply
for (l, r) in left_iters.iter().zip(right_iters.iter().cycle()) { 
    let lm = &lhs.data[l*left_matrix_ele_num..(l+1)*left_matrix_ele_num];
    let rm = &rhs.data[r*right_matrix_ele_num..(r+1)*right_matrix_ele_num];
    let t = &mut target.data[l*target_matrix_ele_num..(l+1)*target_matrix_ele_num];

    for j in 0..right_matrix_shape[1] {
        for li in 0..left_matrix_shape[0] {
            let mut res = 0.0;
            for i in 0..right_matrix_shape[0] {
                res += lm[li * left_matrix_shape[1] + i] * rm[i * right_matrix_shape[1] + j];
            }
            t[li * target_matrix_shape[1] + j] = res;
        }
    }

// reordering multiplication version
    for i in 0..left_matrix_shape[0] {
        for k in 0..left_matrix_shape[1] {
            for j in 0..right_matrix_shape[1] {
                t[i * target_matrix_shape[1] + j] += lm[i * left_matrix_shape[1] + k] * rm[k * right_matrix_shape[1] + j];
            }
        }
    }
}
*/

fn multiply(lhs: &NdArray, rhs: &NdArray) -> NdArray {
    // check if could multiply
    assert!(rhs.dim() >= 2 && lhs.dim() >= 2, "only supports 2d matrix multiplication, but got left = {:?} right = {:?}, please reshape first", lhs.shape, rhs.shape);

    let left_matrix_shape = vec![lhs.shape[lhs.dim() - 2], lhs.shape[lhs.dim() - 1]];
    let right_matrix_shape = vec![rhs.shape[rhs.dim() - 2], rhs.shape[rhs.dim() - 1]];

    assert!(left_matrix_shape[1] == right_matrix_shape[0], "got left = {:?} right = {:?}, please check shape", lhs.shape, rhs.shape);

    // check if could broadcast despite the low-2 dimension shape
    let lhs_broadcast_shape = lhs.shape.iter().take(lhs.dim() - 2).map(|i| *i).collect();
    let rhs_broadcast_shape = rhs.shape.iter().take(rhs.dim() - 2).map(|i| *i).collect();
    assert!(NdArray::can_broadcast(&lhs_broadcast_shape, &rhs_broadcast_shape), "{:?} cannot be broadcasted by {:?}", lhs_broadcast_shape, rhs_broadcast_shape);

    // prepare the target
    let mut target_shape = lhs_broadcast_shape.clone();
    let target_matrix_shape = vec![left_matrix_shape[0], right_matrix_shape[1]];
    target_shape.extend(target_matrix_shape.iter());
    let mut target = NdArray::new(target_shape);
    
    // prepare the index
    let left_matrix_ele_num = NdArray::total_num(&left_matrix_shape);
    let right_matrix_ele_num = NdArray::total_num(&right_matrix_shape);
    let target_matrix_ele_num = NdArray::total_num(&target_matrix_shape);

    // multiplication
    lhs.data.chunks_exact(left_matrix_ele_num)
    .zip(rhs.data.chunks_exact(right_matrix_ele_num).cycle())
    .zip(target.data.chunks_exact_mut(target_matrix_ele_num))
    .for_each(|((lm, rm), tm)| {
        tm.chunks_exact_mut(target_matrix_shape[1])
        .zip(lm.chunks_exact(left_matrix_shape[1]))
        .for_each(|(tmi, lmi)|{
            lmi.iter().zip(rm.chunks_exact(right_matrix_shape[1]))
            .for_each(|(lmik, rmk)|{
                tmi.iter_mut().zip(rmk.iter()).for_each(|(tmij, rmkj)| {
                    *tmij += lmik * rmkj;
                });
            });
        });
    });
    
    target
}

trait NdArrayMultiplyTrait {
    fn multiply(self, lhs: &NdArray) -> NdArray;
}

impl NdArrayMultiplyTrait for &NdArray {
    fn multiply(self, lhs: &NdArray) -> NdArray {
        multiply(lhs, self)
    }
}

impl NdArrayMultiplyTrait for &mut NdArray {
    fn multiply(self, lhs: &NdArray) -> NdArray {
        multiply(lhs, self)
    }
}

impl NdArrayMultiplyTrait for f32 {
    fn multiply(self, lhs: &NdArray) -> NdArray {
        let mut temp = lhs.clone();
        temp.data.iter_mut().for_each(|i| *i *= self);
        temp
    }
}

impl<T: NdArrayMultiplyTrait> std::ops::Mul<T> for &NdArray {
    type Output = NdArray;
    fn mul(self, rhs: T) -> Self::Output {
        rhs.multiply(self)
    }
}
// multiply-------------------------------

#[cfg(test)]
mod test {
    use super::NdArray;

    #[test]
    fn test_index() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(vec![2, 3]);
        let d = &mut a[0];
        d.iter_mut().for_each(|i| *i = -*i);
        println!("{:?}", a);
        let mut aa = NdArray::new(vec![-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]);
        aa.reshape(vec![2, 3]);
        assert_eq!(aa, a);
    }

    #[test]
    fn test_ops_add() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        k.reverse();
        let b = NdArray::new(k);
        let c = &a + &b;
        assert_eq!(c, NdArray::new(vec![7.0;6]));

        a.reshape(vec![2, 3]);
        let b = NdArray::new(vec![-1.0, -2.0, -3.0]);
        let c = &a + &b;
        let mut cc = NdArray::new(vec![0.0, 0.0, 0.0, 3.0, 3.0, 3.0]);
        cc.reshape(vec![2, 3]);
        println!("a = {:?}\nb = {:?}\na+b = {:?}", a, b, c);
        assert_eq!(cc, c);
    }

    #[test]
    fn test_ops_sub() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = NdArray::new(k);

        let c = &a - &b;
        let cc = NdArray::new(vec![0.0; 6]);
        assert_eq!(cc, c);
    }

    #[test]
    fn test_ops_multiply_by_float() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = &a * 2.0;
        let cc = NdArray::new(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c, cc)
    }

    #[test]
    fn test_ops_multiply_by_ndarray() {
        // example 1
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(vec![2, 3]);
        let mut b = NdArray::new(vec![-1.0, -2.0, -3.0]);
        b.reshape(vec![3, 1]);
        println!("a = {:?}\nb = {:?}", a, b);
        let c =  &a * &b;
        println!("a * b = {:?}", c);
        let mut cc = NdArray::new(vec![-14.0, -32.0]);
        cc.reshape(vec![2, 1]);
        assert_eq!(c, cc);

        // mul broadcast example 2
        // a = 2 x 3 x 2
        // b = 2 x 1
        let a: Vec<f32> = (0..12).into_iter().map(|i| i as f32).collect();
        let mut a = NdArray::new(a);
        a.reshape(vec![2, 3, 2]);
        let mut b = NdArray::new(vec![0.0, -1.0]);
        b.reshape(vec![2, 1]);
        println!("a = {:?}\nb = {:?}", a, b);
        let c = &a * &b;
        println!("a * b = {:?}", c);
        let mut cc = NdArray::new(vec![-1.0, -3.0, -5.0, -7.0, -9.0, -11.0]);
        cc.reshape(vec![2, 3, 1]);
        assert_eq!(cc, c);

        // mul broadcast example 3
        // a = 1 x 2 x 3 x 2
        // b = 2 x 2 x 1
        let a: Vec<f32> = (0..12).into_iter().map(|i| i as f32).collect();
        let mut a = NdArray::new(a);
        a.reshape(vec![1, 2, 3, 2]);
        let mut b = NdArray::new(vec![0.0, -1.0, 1.0, 0.0]);
        b.reshape(vec![2, 2, 1]);
        println!("a = {:?}\nb = {:?}", a, b);
        let c = &a * &b;
        println!("a * b = {:?}", c);
        let mut cc = NdArray::new(vec![-1.0, -3.0, -5.0, 6.0, 8.0, 10.0]);
        cc.reshape(vec![1, 2, 3, 1]);
        assert_eq!(cc, c);
    }

    #[test]
    fn test_multiply_profile() {
        use std::time::{Instant, Duration};
        const PSIZE: usize = 512 * 512 * 512;
        let mut a = NdArray::new((0..PSIZE).map(|i| i as f32).collect::<Vec<f32>>());
        a.reshape(vec![8, 128, 256, 512]);

        let mut b = NdArray::new((0..PSIZE).map(|i| i as f32).collect::<Vec<f32>>());
        b.reshape(vec![128, 512, 2048]);

        let start = Instant::now();
        let _ = &a * &b;
        let dur = start.elapsed();
        println!("execute {:?}", dur);
        assert!(dur < Duration::from_secs_f32(50.0));
    }
}