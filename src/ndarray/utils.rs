use super::NdArray;

pub fn softmax(x: &mut NdArray, dim: i32) {
    // check if dim is legal
    assert!((dim >= 0 && dim < x.dim() as i32) || 
    (dim < 0 && dim >= - (x.dim() as i32)));
    let dim: usize = if dim < 0 {
        (x.dim() as i32 + dim) as usize
    } else {
        dim as usize
    };

    let index_base_sizes = NdArray::index_base_sizes(&x.shape);

    // retrieval from the specified dim 
    fn retrieval_by_recursive(pos: usize, max_dim: usize, softmax_dim: usize, idx: usize, index_base_sizes: &Vec<usize>, shapes: &Vec<usize>, data: &mut Vec<f32>) {
        if pos == max_dim - 1 {
            let last_size = if softmax_dim == pos {
                1
            } else {
                shapes[pos]
            };
            (0..last_size).for_each(|i| {
                let idxs: Vec<usize> = (0..shapes[softmax_dim]).map(|sd_i| idx + i + index_base_sizes[softmax_dim] * sd_i).collect();
                let mut softmax_data: Vec<f32> = idxs.iter().map(|ii| data[*ii]).collect();

                let constant = softmax_data.iter().cloned().reduce(f32::max).unwrap();
                softmax_data.iter_mut().for_each(|x| {
                    *x = (*x - constant).exp();
                });
                let sum: f32 = softmax_data.iter().sum();
                softmax_data.iter_mut().for_each(|x| *x /= sum);
                idxs.into_iter().zip(softmax_data.into_iter()).for_each(|(sd_i, d)| {
                    data[sd_i] = d;
                })
            });
        } else if pos == softmax_dim {
            retrieval_by_recursive(pos + 1, max_dim, softmax_dim, idx, index_base_sizes, shapes, data);
        } else {
            for i in 0..shapes[pos] {
                retrieval_by_recursive(pos + 1, max_dim, softmax_dim, idx + i * index_base_sizes[pos], index_base_sizes, shapes, data);
            }
        }
    }

    // start
    retrieval_by_recursive(0, x.dim(), dim, 0, &index_base_sizes, &x.shape, &mut x.data);
}

#[cfg(test)]
mod test {
    use super::{softmax};
    use super::NdArray;

    #[test]
    fn test_softmax() {
        // example 1
        let mut x = NdArray::new(vec![vec![1.0; 3]; 2]);
        softmax(&mut x, -1);
        println!("{}", x);

        // example 2
        softmax(&mut x, -2);
        println!("{}", x);

        // example 3
        let mut x = NdArray::new(vec![vec![1.1, -3.7, 341.23, 46.6], vec![3.23, 6.2, 0.4, -2.87]]);
        softmax(&mut x, -1);
        let xx = NdArray::new(vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.048654296, 0.94836545, 0.002871229, 0.000109125154]
          ]);
        println!("{}", x);
        assert_eq!(xx, x);

    }
}