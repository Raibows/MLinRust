mod ops;


#[derive(Debug, Clone)]
struct NdArray {
    shape: Vec<usize>,
    data: Vec<f32>,
}

trait NdArrayNewTrait {
    fn new(self) -> NdArray;
}

trait ReshapeTrait {
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

        fn recursive_print(data: &[f32], shape: &Vec<usize>, cursor: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let width = " ";
            if cursor == shape.len() - 1 {
                write!(f, "{}[", width.repeat(4 * (cursor)))?;
                print_row(f, data)?;
                write!(f, "]")?;
            } else if cursor == shape.len() - 2 {
                // print it, since it is a vector now
                write!(f, "{}[\n", width.repeat(4 * cursor))?;
                let row_size = shape.last().unwrap();
                for i in 0..shape[cursor] {
                    write!(f, "{}[", width.repeat(4 * (cursor + 1)))?;
                    print_row(f, &data[i*row_size..(i+1)*row_size])?;
                    write!(f, "],\n")?;
                }
                write!(f, "{}]\n", width.repeat(4 * cursor))?;
            } else {
                let res_size: usize = shape.iter().skip(cursor+1).sum();
                for i in 0..shape[cursor] {
                    write!(f, "{}[\n", width.repeat(4 * cursor))?;
                    recursive_print(&data[i*res_size..(i+1)*res_size], shape, cursor+1, f)?;
                    write!(f, "{}]\n", width.repeat(4 * cursor))?;
                }
            }
            Ok(())
        }


        

        fn recursive2_print(data: &[f32], shape: &Vec<usize>, cursor: usize, f: &mut std::fmt::Formatter<'_>, width: usize) -> std::fmt::Result {
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
                    recursive2_print(&data[i*base..(i+1)*base], shape, cursor+1, f, width)?;
                }
                let i = shape[cursor] - 1;
                recursive2_print(&data[i*base..(i+1)*base], shape, cursor+1, f, width)?;
                if cursor == 0 {
                    write!(f, "{}]", blank)?;
                } else {
                    write!(f, "{}],\n", blank)?;
                }
            }
            Ok(())
        }

        recursive2_print(&self.data[..], &self.shape, 0, f, 2)?;
        write!(f, ", ndarray: {:?}", self.shape)?;
        Ok(())
    }
}

impl NdArray {
    fn new<T: NdArrayNewTrait>(arg: T) -> Self {
        arg.new()
    }

    fn dim(&self) -> usize {
        self.shape.len()
    }

    fn reshape<T: ReshapeTrait>(&mut self, shape: T) {
        self.shape = shape.reshape(&self.shape);
    }

    fn total_num(v: &Vec<usize>) -> usize {
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
}