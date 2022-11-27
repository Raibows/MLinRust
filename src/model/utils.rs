use crate::ndarray::NdArray;

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
            (grad.data_as_vector().iter().fold(f32::MIN, |s, i| f32::max(s, *i)), max_norm)
        }
    };
    let clip_coef = max_norm / (total_norm + 1e-6);
    if clip_coef < 1.0 {
        grad.data_as_vector().iter_mut().for_each(|i| *i *= clip_coef);
    }
}