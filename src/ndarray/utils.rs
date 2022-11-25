use super::NdArray;

pub fn check_dim_is_legal(dim: i32, max_dim: usize) -> usize {
    // check if dim is legal
    assert!((dim >= 0 && dim < max_dim as i32) || 
    (dim < 0 && dim >= - (max_dim as i32)));
    if dim < 0 {
        (max_dim as i32 + dim) as usize
    } else {
        dim as usize
    }
}

fn collect_by_recursive_then_gather_to<F>(pos: usize, max_dim: usize, softmax_dim: usize, src_idx: usize, tgt_idx: usize, src_base_sizes: &Vec<usize>, tgt_base_sizes: &Vec<usize>, shapes: &Vec<usize>, src_data: &Vec<f32>, tgt_data: &mut Vec<f32>, gather: &F) 
where F: Fn(&Vec<f32>, &Vec<usize>) -> f32 {
    if pos == max_dim - 1 {
        let last_size = if softmax_dim == pos {
            1
        } else {
            shapes[pos]
        };
        (0..last_size).for_each(|i| {
            let idxs: Vec<usize> = (0..shapes[softmax_dim]).map(|sd_i| src_idx + i + src_base_sizes[softmax_dim] * sd_i).collect();
            // let sum_value = idxs.iter().fold(0.0, |s, ii| s + src_data[*ii]);
            tgt_data[tgt_idx + i] = gather(&src_data, &idxs);
        });
    } else if pos == softmax_dim {
        collect_by_recursive_then_gather_to(pos + 1, max_dim, softmax_dim, src_idx, tgt_idx, src_base_sizes, tgt_base_sizes, shapes, src_data, tgt_data, gather);
    } else {
        for i in 0..shapes[pos] {
            collect_by_recursive_then_gather_to(pos + 1, max_dim, softmax_dim, src_idx + i * src_base_sizes[pos], tgt_idx + i * tgt_base_sizes[pos], src_base_sizes, tgt_base_sizes, shapes, src_data, tgt_data, gather);
        }
    }
}

fn gather_by_a_specific_dim_and_do(x: &NdArray, dim: i32, gather: &dyn Fn(&Vec<f32>, &Vec<usize>) -> f32) -> NdArray {
    // squeeze this dim, and gather [....] to f32
    // e.g., sum by dim, argmax by dim, etc.
    let dim = check_dim_is_legal(dim, x.dim());
    let tgt_shape = x.shape.iter().enumerate().fold(vec![], |mut s, (i, item)| {
        if i == dim {
            s.push(1);
            s
        } else {
            s.push(*item);
            s
        }
    });
    let mut target = NdArray::new(tgt_shape);

    collect_by_recursive_then_gather_to(0, x.dim(), dim, 0, 0, &NdArray::index_base_sizes(&x.shape), &NdArray::index_base_sizes(&target.shape), &x.shape, &x.data, &mut target.data, &gather);

    target.squeeze(dim as i32);

    target
}


pub fn softmax(x: &mut NdArray, dim: i32) {
    let dim = check_dim_is_legal(dim, x.dim());

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

pub fn sum_ndarray(x: &NdArray, dim: i32) -> NdArray {
    fn sum_value(src_data: &Vec<f32>, idxs: &Vec<usize>) -> f32 {
        idxs.iter().fold(0.0, |s, i| s + src_data[*i])
    }
    gather_by_a_specific_dim_and_do(x, dim, &sum_value)
}

pub fn argmax(x: &NdArray, dim: i32) -> NdArray {
    // since NdArray has not implemented the template for usize, so we have to return NdArray<f32> instead
    // todo
    fn get_arg_by_max(src_data: &Vec<f32>, idxs: &Vec<usize>) -> f32 {
        let res = idxs.iter().enumerate().fold((f32::MIN, -1.0), |(s, si), (i, ii)| {
            if src_data[*ii] > s {
                (src_data[*ii], i as f32)
            } else {
                (s, si)
            }
        });
        res.1
    }
    gather_by_a_specific_dim_and_do(x, dim, &get_arg_by_max)
}

#[cfg(test)]
mod test {
    use super::{softmax, sum_ndarray, argmax};
    use super::NdArray;

    #[test]
    fn test_argmax() {
        let mut x = NdArray::new(vec![1.0, -123.0, 5.8, 2.3, 11.3, 5.0]);
        x.reshape(vec![2, 3]);
        let t = argmax(&x, -1);
        println!("argmax {}", x);
        let tt = NdArray::new(vec![2.0, 1.0]);
        assert!(tt == t);
        println!("{}", t);
    }


    #[test]
    fn test_sum_ndarray() {
        let x = NdArray::new(vec![vec![1.0; 3]; 2]);
        let t = sum_ndarray(&x, 0);
        assert_eq!(NdArray::new(vec![2.0, 2.0, 2.0]), t);
        println!("{}", t);

        let mut x = NdArray::new((0..12).map(|i| i as f32).collect::<Vec<f32>>());
        x.reshape(vec![2, 3, 2]);
        let t = sum_ndarray(&x, 1);
        assert!(t.shape == vec![2, 2]);
        println!("{}", t);
    }

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