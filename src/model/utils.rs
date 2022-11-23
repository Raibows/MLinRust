fn softmax(mut x: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let constant: Vec<f32> = x.iter().map(|row| row.iter().cloned().reduce(f32::max).unwrap()).collect();
    x.iter_mut().zip(constant).for_each(|(row, cc)| {
        row.iter_mut().for_each(|item| {*item = (*item-cc).exp();})
    });
    let s: Vec<f32> = x.iter().map(|row| row.iter().sum()).collect();
    x.iter_mut().zip(s.iter()).for_each(|(row, s)| {
        row.iter_mut().for_each(|item| *item /= s)
    });
    x
}

#[cfg(test)]
mod test {
    use std::vec;
    use super::softmax;


    #[test]
    fn test_softmax() {
        let mut x = vec![vec![1.0; 3]];
        x = softmax(x);
        assert_eq!(&x[0][..], &[1.0/3.0; 3]);
        let mut x = vec![vec![1.1, -3.7, 341.23, 46.6], vec![3.23, 6.2, 0.4, -2.87]];
        x = softmax(x);
        assert_eq!(&x[0][..], &[0.0, 0.0, 1.0, 0.0])
    }
}