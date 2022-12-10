use super::{Model, utils::{NormType, Penalty}, nn::{linear::Linear, criterion::CrossEntropyLoss}};
use crate::{ndarray::{NdArray, utils::{argmax}}, model::{nn::NNBackPropagation}};

pub struct LogisticRegression {
    feature_size: usize,
    linear: Linear,
    criterion: CrossEntropyLoss,
}

impl LogisticRegression {
    pub fn new<F>(feature_size: usize, class_num: usize, penalty: Option<Penalty>, weight_init_fn: F) -> Self
    where F: Fn(&mut [f32])
    {   
        let mut linear = Linear::new(feature_size, class_num, penalty);
        weight_init_fn(linear.weight_mut_borrow());
        Self { feature_size: feature_size, linear: linear, criterion: CrossEntropyLoss::new() }
    }

    pub fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        assert!(input.dim() == 2);
        assert!(input.shape[1] == self.feature_size);
        self.linear.forward(input, required_grad)
    }

    /// return: predictions(after softmax), avg_loss
    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<usize>, lr: f32, gradient_clip_by_norm: Option<NormType>) -> f32 {
        let logits = self.forward(&feature, true);

        let bp_grad = self.criterion.forward(logits, label);

        self.linear.backward(bp_grad);
        self.linear.step(feature.shape[0], lr, gradient_clip_by_norm);

        self.criterion.avg_loss
    }
}

impl Model<usize> for LogisticRegression {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let input = vec![feature.clone()];
        let logits = self.linear.forward_as_borrow(&NdArray::new(input));
        let (_, predictions) = argmax(&logits, -1).destroy();
        let predictions: Vec<usize> = predictions.into_iter().map(|i| i as usize).collect();
        predictions[0]
    }
}

#[cfg(test)]
mod test {
    
    use super::{NdArray, LogisticRegression};
    use crate::model::Model;

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![0, 1];
        let mut model = LogisticRegression::new(batch_of_feature.shape[1], 2, None,  |_| {});
        // step 1
        for _ in 0..5 {
            let loss = model.one_step(&batch_of_feature, &label, 1e-1, None);
            println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
        }
    }
}