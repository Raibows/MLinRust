use super::{Model, utils::{NormType, Penalty}, nn::{linear::Linear, criterion::{MeanSquaredError}}};
use crate::{ndarray::{NdArray}, model::{nn::NNBackPropagation}};

pub struct LinearRegression {
    feature_size: usize,
    linear: Linear,
    criterion: MeanSquaredError,
}

impl LinearRegression {
    pub fn new<F>(feature_size: usize, penalty: Option<Penalty>, weight_init_fn: F) -> Self
    where F: Fn(&mut [f32])
    {   
        let mut linear = Linear::new(feature_size, 1, penalty);
        weight_init_fn(linear.weight_mut_borrow());
        Self { feature_size: feature_size, linear: linear, criterion: MeanSquaredError::new() }
    }

    pub fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        assert!(input.dim() == 2);
        assert!(input.shape[1] == self.feature_size);
        self.linear.forward(input, required_grad)
    }

    /// return: predictions(after softmax), avg_loss
    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<f32>, lr: f32, gradient_clip_by_norm: Option<NormType>) -> f32 {
        let logits = self.forward(&feature, true);

        let bp_grad = self.criterion.forward(logits, label);

        self.linear.backward(bp_grad);
        self.linear.step(feature.shape[0], lr, gradient_clip_by_norm);

        self.criterion.avg_loss
    }
}

impl Model<f32> for LinearRegression {
    fn predict(&self, feature: &Vec<f32>) -> f32 {
        let input = vec![feature.clone()];
        let predicts = self.linear.forward_as_borrow(&NdArray::new(input));
        predicts[0][0]
    }
}

#[cfg(test)]
mod test {

    use super::{NdArray, LinearRegression};
    use crate::model::Model;

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![5.8, -6.7];
        let mut model = LinearRegression::new(batch_of_feature.shape[1], None, |_| {});

        for _ in 0..2 {
            let loss = model.one_step(&batch_of_feature, &label, 1e-3, None);
            println!("loss {loss:?} {} {}", (model.predict(&datas[0]) - label[0]).abs(), (model.predict(&datas[1]) - label[1]).abs());
        }
    }
}