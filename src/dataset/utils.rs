use crate::ndarray::NdArray;
use crate::ndarray::utils::{std, mean, min, max};

use super::{Dataset, TaskLabelType};


pub enum ImputeType {
    Mean,
    Zero,
    Value(f32),
}

/// impute missing values in the dataset
/// * res: should be the results of the dataset preprocessing
/// * filled: ImputeType {Mean, Zero, Specific Value f32}
pub fn impute_missing_values(res: Vec<Vec<Result<f32, Box<dyn std::error::Error>>>>, filled: ImputeType) -> Vec<Vec<f32>> {
    match filled {
        ImputeType::Mean => {
            let counter = res.iter().fold(vec![(0.0f32, 0.0f32); res[0].len()], |mut fold, item| {
                item.iter().enumerate().for_each(|(i, item)| {
                    if let Ok(v) = item {
                        fold[i].0 += 1.0;
                        fold[i].1 += v;
                    }
                });
                fold
            });
            let counter: Vec<f32> = counter.into_iter().map(|(num, sum)| sum / num).collect();
            res.into_iter().map(|item| {
                item.into_iter().enumerate().map(|(i, e)| {
                    if let Ok(v) = e {
                        v
                    } else {
                        counter[i]
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
        ImputeType::Zero => {
            res.into_iter().map(|item| {
                item.into_iter().map(|e| {
                    if let Ok(v) = e {
                        v
                    } else {
                        0.0
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
        ImputeType::Value(value) => {
            res.into_iter().map(|item| {
                item.into_iter().map(|e| {
                    if let Ok(v) = e {
                        v
                    } else {
                        value
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
    }
}

/// scaler for data normalization
pub enum ScalerType {
    Standard,
    MinMax,
}

pub fn normalize_dataset<T: TaskLabelType + Copy>(dataset: &mut Dataset<T>, scaler: ScalerType) {
    let feature_len = dataset.feature_len();
    let data = std::mem::replace(&mut dataset.features, vec![]);
    let data = NdArray::new(data); // [num, feature_len]
    let data = match scaler {
        ScalerType::MinMax => {
            let min = min(&data, 0);
            let numerator = &data - &min;

            let mut denominator = -min + max(&data, 0);
            denominator.data_as_mut_vector().iter_mut().for_each(|i| *i = 1.0 / f32::max(1e-6, *i));

            numerator.point_multiply(&denominator).destroy().1
        },
        ScalerType::Standard => {
            let mu = mean(&data, 0);
            // the dataset should have infinite data idealy
            let mut sigma = std(&data, 0, false);
            sigma.data_as_mut_vector().iter_mut().for_each(|i| *i = 1.0 / f32::max(1e-6, *i));
            (data - mu).point_multiply(&sigma).destroy().1            
        },
    };
    let data: Vec<Vec<f32>> = data.chunks_exact(feature_len).map(|v| v.into()).collect();
    dataset.features = data;
}

#[cfg(test)]
mod test {
    use crate::{ndarray::{NdArray, utils::{mean, std}}, utils::RandGenerator, dataset};

    use super::normalize_dataset;

    #[test]
    fn test_normalize_dataset() {
        let mut rng = RandGenerator::new(0);
        let data = (0..4).map(|_| (0..2).map(|_| rng.gen_f32()).collect()).collect();
        let mut dataset = dataset::Dataset::new(data, vec![rng.gen_f32(); 4], None);

        println!("{:?}", dataset.features);

        normalize_dataset(&mut dataset, super::ScalerType::MinMax);
        println!("min_max{:?}", dataset.features);


        normalize_dataset(&mut dataset, super::ScalerType::Standard);
        let mut a = NdArray::new(dataset.features.clone());
        a.reshape(vec![dataset.len(), dataset.feature_len()]);
        println!("std{:?}\nmean:{}, std:{}", dataset.features, mean(&a, 0), std(&a, 0, false));
    }

    #[test]
    fn test() {
        let a = NdArray::random(vec![4, 2], None);
        let b = NdArray::random(vec![1, 2], Some(2));
        println!("a can broadcast b {}", a + b);
    }
}