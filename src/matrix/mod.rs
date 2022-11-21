pub mod ops;

#[derive(Debug)]
struct Matrix {
    params: Vec<Vec<f32>>,
}

trait MatrixNewArg {
    fn new(self) -> Matrix;
}

impl MatrixNewArg for (usize, usize) {
    fn new(self) -> Matrix {
        let params = vec![vec![0.0f32; self.1]; self.0];
        Matrix { params: params }
    }
}

impl MatrixNewArg for Vec<Vec<f32>> {
    fn new(self) -> Matrix {
        Matrix { params: self }
    }
}



impl Matrix {
    fn new<T: MatrixNewArg>(arg: T) -> Self {
        arg.new()
    }

    fn shape(&self) -> (usize, usize) {
        (self.params.len(), self.params[0].len())
    }

    fn get_row(&mut self, row: usize) -> &mut Vec<f32> {
        &mut self.params[row]
    }

    fn get_column(&mut self, column: usize) -> Vec<&mut f32> {
        self.params.iter_mut().map(|row| &mut row[column]).collect()
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.params.iter() {
            write!(f, "{:?}\n", row)?;
        }
        write!(f, "shape {}x{}", self.shape().0, self.shape().1)
    }
}

#[cfg(test)]
mod test {
    use super::Matrix;

    #[test]
    fn test_matrix_index() {
        let mut m = Matrix::new((2, 3));
        m.params = vec![vec![0.0, 0.1], vec![1.0, 1.1]];
        // println!("{:?}", m.get_row(0));
        // println!("{:?}", m.get_row(0));
        println!("{}", m);
        let mut column = m.get_column(1);
        column.iter_mut().for_each(|i| **i += 1.0);
        println!("{}", m);
    }
}