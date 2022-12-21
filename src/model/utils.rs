use crate::ndarray::NdArray;

/// known as:
/// * p = 1.0: manhattan distance
/// * p = 2.0: euclidean distance
/// * p ~ 80.0: infinity
pub fn minkowski_distance(x: &Vec<f32>, y: &Vec<f32>, p: f32) -> f32 {
    assert!(x.len() == y.len());
    // infinite
    if p > f32::MAX.log2() / x.len() as f32 {
        x.iter().zip(y.iter()).fold(0.0, |s, (i, j)| {
            f32::max((i - j).abs(), s)
        })
    } else {
        x.iter().zip(y.iter()).fold(0.0, |s, (i, j)| {
            s + (i - j).abs().powf(p)
        }).powf(1.0 / p)
    }
    
}

#[derive(Clone, Copy, Debug)]
pub enum Penalty {
    LassoL1(f32),
    RidgeL2(f32),
}

#[derive(Debug, Clone, Copy)]
pub enum NormType {
    L1(f32),
    L2(f32),
    Inf(f32),
}

pub fn gradient_clip(grad: &mut NdArray, norm_type: &NormType) {
    let (total_norm, max_norm) = match norm_type {
        NormType::L1(max_norm) => {
            (grad.data_as_vector().iter().fold(0.0, |s, i| s + i.abs()), max_norm)
        },
        NormType::L2(max_norm) => {
            (grad.data_as_vector().iter().fold(0.0, |s, i| s + i * i).sqrt(), max_norm)
        },
        NormType::Inf(max_norm) => {
            (grad.data_as_vector().iter().fold(0.0, |s, i| f32::max(s, i.abs())), max_norm)
        }
    };
    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef < 1.0 {
        grad.data_as_mut_vector().iter_mut().for_each(|i| *i *= clip_coef);
    }
}

pub fn calculate_penalty_value(var: &NdArray, penalty: Penalty) -> f32 {
    // return penalty value
    match penalty {
        Penalty::LassoL1(ratio) => {
            ratio * var.data_as_vector().iter().fold(0.0, |s, i| s + i.abs())
        },
        Penalty::RidgeL2(ratio) => {
            ratio * (var.data_as_vector().iter().fold(0.0, |s, i| s + i * i)).sqrt()
        },
    }
}

pub fn calculate_penalty_grad(var: &NdArray, penalty: Penalty) -> NdArray {
    // return the grad caused by penalty
    let g = match penalty {
        Penalty::LassoL1(ratio) => {
            let g: Vec<f32> = var.data_as_vector().iter().map(|i| i / f32::max(1e-6, i.abs()) * ratio).collect();
            g
        },
        Penalty::RidgeL2(ratio) => {
            let t = 0.5 * 1.0 / f32::max(1e-6, var.data_as_vector().iter().fold(0.0, |s, i| s + i * i).sqrt());
            let g: Vec<f32> = var.data_as_vector().iter().map(|i| i / f32::max(1e-6, i.abs()) * 2.0 * ratio * t).collect();
            g
        },
    };
    let mut g = NdArray::new(g);
    g.reshape(&var.shape);
    g
}