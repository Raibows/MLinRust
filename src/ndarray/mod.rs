use std::ops::Deref;

use self::utils::check_dim_is_legal;

mod ops;
pub mod utils;


#[derive(Debug, Clone)]
pub struct NdArray {
    pub shape: Vec<usize>,
    data: Vec<f32>,
}

pub trait NdArrayNewTrait {
    fn new(self) -> NdArray;
}

pub trait ReshapeTrait {
    fn reshape(&self, source: &Vec<usize>) -> Vec<usize>;
}
impl ReshapeTrait for Vec<i32> {
    fn reshape(&self, source: &Vec<usize>) -> Vec<usize> {
        let mut p = vec![];
        let mut index = None;
        for (i, e) in self.iter().enumerate() {
            if *e == -1 {
                if index.is_some() {
                    assert!(false, "-1 can only exist one but got {:?}", self);
                } else {
                    index = Some(i);
                }
            } else {
                assert!(*e > 0);
                p.push(*e as usize);
            }
        }
        let current_n = NdArray::total_num(source);
        let mut shape = p;
        if let Some(index) = index {
            let temp =  NdArray::total_num(&shape);
            assert!(current_n % temp == 0, "cannot reshape by using -1, current {:?} expect/-1 {:?}", source, shape);
            shape.insert(index, current_n / temp);
        } else {
            assert!(current_n == NdArray::total_num(&shape));
        }
        shape
    }
}
impl ReshapeTrait for Vec<usize> {
    fn reshape(&self, source: &Vec<usize>) -> Vec<usize> {
        assert!(NdArray::total_num(source) == NdArray::total_num(self));
        self.clone()
    }
}

impl ReshapeTrait for &Vec<usize> {
    fn reshape(&self, source: &Vec<usize>) -> Vec<usize> {
        assert!(NdArray::total_num(source) == NdArray::total_num(self));
        self.deref().clone()
    }
}

impl NdArrayNewTrait for Vec<usize> {
    // from the specify shape
    fn new(self) -> NdArray {
        let n = NdArray::total_num(&self);
        NdArray { shape: self, data: vec![0.0; n] }
    }
}

impl NdArrayNewTrait for Vec<f32> {
    // from vector
    fn new(self) -> NdArray {
        NdArray { shape: vec![self.len()], data: self }
    }
}

impl NdArrayNewTrait for Vec<Vec<f32>> {
    // from 2d array
    fn new(self) -> NdArray {
        NdArray { shape: vec![self.len(), self[0].len()], data: self.into_iter().fold(vec![], |mut s, row| {s.extend(row); s} ) }
    }
}

impl std::fmt::Display for NdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn print_row(f: &mut std::fmt::Formatter<'_>, row: &[f32]) -> std::fmt::Result {
            write!(f, "[")?;
            if row.len() <= 20 {
                for i in row.iter().take(row.len() - 1) {
                    write!(f, "{}, ", i)?;
                }
            } else {
                for i in row.iter().take(10) {
                    write!(f, "{}, ", i)?;
                }
                write!(f, "..., ")?;
                for i in row.iter().skip(row.len() - 9) {
                    write!(f, "{}, ", i)?;
                }
            }
            write!(f, "{}]", row[row.len() - 1])?;
            Ok(())
        }

        fn recursive_print(data: &[f32], shape: &Vec<usize>, cursor: usize, f: &mut std::fmt::Formatter<'_>, width: usize) -> std::fmt::Result {
            let blank = " ".repeat(width * cursor);
            if cursor == shape.len() - 1 {
                write!(f, "{}", blank)?;
                print_row(f, data)?;
                if cursor > 0 {
                    write!(f, "\n")?;
                }
            } else {
                write!(f, "{}[\n", blank)?;
                let base: usize = shape.iter().skip(cursor + 1).fold(1, |s, i| s * i);
                for i in 0..shape[cursor] - 1 {
                    recursive_print(&data[i*base..(i+1)*base], shape, cursor+1, f, width)?;
                }
                let i = shape[cursor] - 1;
                recursive_print(&data[i*base..(i+1)*base], shape, cursor+1, f, width)?;
                if cursor == 0 {
                    write!(f, "{}]", blank)?;
                } else {
                    write!(f, "{}],\n", blank)?;
                }
            }
            Ok(())
        }

        recursive_print(&self.data[..], &self.shape, 0, f, 2)?;
        write!(f, ", NdArray::Shape: {:?}", self.shape)?;
        Ok(())
    }
}

impl NdArray {
    pub fn new<T: NdArrayNewTrait>(arg: T) -> Self {
        arg.new()
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape<T: ReshapeTrait>(&mut self, shape: T) {
        self.shape = shape.reshape(&self.shape);
    }

    pub fn total_num(v: &Vec<usize>) -> usize {
        v.iter().fold(1, |s, i| s * i)
    }

    fn can_broadcast(lhs: &Vec<usize>, rhs: &Vec<usize>) -> bool {
        if lhs.len() < rhs.len() ||
        Self::total_num(lhs) < Self::total_num(rhs) {
            false
        } else {
            let mut judge = true;
            let mut m = lhs.iter().rev();
            for rev in rhs.iter().rev() {
                if rev != m.next().unwrap() {
                    judge = false;
                }
            }
            judge
        }
    }

    pub fn index_base_sizes(shape: &Vec<usize>) -> Vec<usize> {
        // help you to calculate the index of data
        let mut sizes: Vec<usize> = shape.iter().rev().fold(vec![1], |mut s, i| {
            s.push(s.last().unwrap() * i);
            s    
        });
        sizes.pop();
        sizes.reverse();
        sizes
    }

    pub fn permute(&self, order: Vec<usize>) -> NdArray {
        // check whether the order is valid
        // e.g., transpose [0, 1] -> [1, 0]
        assert!(order.len() == self.dim());
        let mut tgt_shape = vec![0usize; self.dim()];
        let mut check = vec![false; self.dim()];
        let mut src_tgt_map = vec![0usize; self.dim()];
        for (i, item) in order.iter().enumerate() {
            if check.get(*item).is_none() || check[*item] {
                panic!("Permute Error, check target order {:?}", order);
            } else {
                check[*item] = true;
                tgt_shape[i] = self.shape[*item];
                src_tgt_map[*item] = i;
            }
        }

        // prepare indexs to move data
        let mut target_data = vec![0.0f32; NdArray::total_num(&self.shape)];

        let src_sizes = NdArray::index_base_sizes(&self.shape);
        let tgt_sizes = NdArray::index_base_sizes(&tgt_shape);

        // start moving
        fn recursive_move(dim: usize, pos: usize, src_data_i: usize, tgt_data_i: usize, src_sizes: &Vec<usize>, tgt_sizes: &Vec<usize>, src_data: &Vec<f32>, tgt_data: &mut Vec<f32>, src_shapes: &Vec<usize>, tgt_shapes: &Vec<usize>, src_tgt_pos_map: &Vec<usize>) {
            if pos == dim - 1 {
                for i in 0..src_shapes[pos] {
                    let ti = tgt_data_i + tgt_sizes[src_tgt_pos_map[pos]] * i;
                    let si = src_data_i + i * src_sizes[pos];
                    // println!("src_sizes = {:?}, tgt_sizes = {:?} \nti = {}, si = {}", src_sizes, tgt_sizes, ti, si);
                    tgt_data[ti] = src_data[si];
                }
            } else {
                for i in 0..src_shapes[pos] {
                    let ti = tgt_data_i + tgt_sizes[src_tgt_pos_map[pos]] * i;
                    let si = src_data_i + src_sizes[pos] * i;
                    recursive_move(dim, pos + 1, si, ti, src_sizes, tgt_sizes, src_data, tgt_data, src_shapes, tgt_shapes, src_tgt_pos_map);
                }
            }
        }
        
        recursive_move(self.dim(), 0, 0, 0, &src_sizes, &tgt_sizes, &self.data, &mut target_data, &self.shape, &tgt_shape, &src_tgt_map);

        let mut target = NdArray::new(target_data);
        target.reshape(tgt_shape);

        target
    }

    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|i| *i = 0.0);
    }

    pub fn squeeze(&mut self, dim: i32) {
        let udim = check_dim_is_legal(dim, self.dim());
        assert!(self.shape[udim] == 1, "the shape is {:?}, shape[dim={}] = {} != 1", self.shape, udim, self.shape[udim]);
        if self.dim() > 1 {
            self.shape.remove(udim);
        }
    }

    pub fn destroy(self) -> (Vec<usize>, Vec<f32>) {
        (self.shape, self.data)
    }

    pub fn data_as_mut_vector(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn data_as_vector(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod test {
    use super::NdArray;

    #[test]
    fn test_new() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0]);
        println!("a.shape = {:?}", a.shape);
        a.reshape(vec![-1, 1]);
        println!("a.shape = {:?}", a.shape);
        let mut b = NdArray::new(vec![4, 5, 3, 1]);
        b.reshape(vec![10, -1, 3, 1]);
        println!("b.shape = {:?}", b.shape);
        println!("broadcast b + a = {}", NdArray::can_broadcast(&b.shape, &a.shape));
        assert!(NdArray::can_broadcast(&b.shape, &a.shape));
    }

    #[test]
    fn test_fmt_display() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        println!("{}", a);

        a.reshape(vec![2, 3]);
        println!("{}", a);

        a.reshape(vec![3, 2]);
        println!("{}", a);

        let mut a = NdArray::new((0..16).map(|i| i as f32).collect::<Vec<f32>>());
        a.reshape(vec![2, 2, 4]);
        println!("{}", a);

        let mut a = NdArray::new((0..128).map(|i| i as f32).collect::<Vec<f32>>());
        a.reshape(vec![2, 2, 32]);
        println!("{}", a);
    }

    #[test]
    fn test_permute() {
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(vec![2, 3]);
        println!("{}", a);
        let b = a.permute(vec![1, 0]);
        println!("after permute transpose");
        println!("{}", b);
        let bb = NdArray::new(
            vec![
                vec![1.0, 4.0], 
                vec![2.0, 5.0], 
                vec![3.0, 6.0],
            ]
        );
        assert_eq!(bb, b);
        assert_eq!(a, bb.permute(vec![1, 0]));

        println!("here-----------------\n\n");
        let mut a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        a.reshape(vec![1, 2, 3]);
        let b = a.permute(vec![2, 0, 1]);
        println!("{}", b);
        let mut bb = NdArray::new(
            vec![
                vec![1.0, 4.0], 
                vec![2.0, 5.0], 
                vec![3.0, 6.0],
            ]
        );
        bb.reshape(vec![3, 1, 2]);
        assert_eq!(bb, b);
    }
}