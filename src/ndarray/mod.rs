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
        let n = self.iter().fold(1, |s, i| s * *i);
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
        todo!()
    }
}

impl NdArray {
    fn new<T: NdArrayNewTrait>(arg: T) -> Self {
        arg.new()
    }

    fn dim(&self) -> usize {
        self.shape.len()
    }

    // fn index(&self, idx: usize) -> Self {
    //     // take a copy
    //     assert!(idx < self.shape[0]);
    //     let (s, e) = self.shape.iter().skip(1).fold((idx , idx+1), |s, i| (s.0*i, s.1 * i));
    //     let data: Vec<f32> = self.data.iter().skip(s).take(e - s).map(|i| *i).collect();
    //     Self::new(data)    
    // }

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
    }
}