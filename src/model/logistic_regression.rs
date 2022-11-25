use super::Model;
use crate::ndarray::{NdArray, utils::{softmax, sum_ndarray, argmax}};


struct LogitRegression {
    weight: NdArray,
    bias: NdArray,
    feature_size: usize,
    class_num: usize,
    grad_w: NdArray,
    grad_b: NdArray,
}

impl LogitRegression {
    pub fn new<F>(feature_size: usize, class_num: usize, weight_init_fn: F) -> Self
    where F: Fn(Vec<Vec<f32>>) -> Vec<Vec<f32>>
    {
        let weight = vec![vec![0.0f32; feature_size]; class_num];
        let bias = NdArray::new(vec![class_num]);
        let weight = NdArray::new(weight_init_fn(weight));
        Self { weight: weight, bias: bias, feature_size: feature_size, class_num: class_num, grad_w: NdArray::new(vec![class_num, feature_size]), grad_b: NdArray::new(vec![class_num]) }
    }

    pub fn forward(&self, input: &NdArray) -> NdArray {
        assert!(input.dim() == 2);
        assert!(input.shape[1] == self.feature_size);
        // weight: [class_num, feature_size]
        // input: [bsz, feature_size]
        // bias: [class_num]
        // return: [bsz, class_num] logits
        &(input * &self.weight.permute(vec![1, 0])) + &self.bias
    }

    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<usize>) -> Vec<f32> {
        assert!(feature.dim() == 2); // feature: [bsz, feature_size]
        
        // after softmax logits
        let mut predicts = self.forward(feature); //
        softmax(&mut predicts, -1);

        // loss
        let entropy_loss: Vec<f32> = label.iter().enumerate().map(|(i, l)| - predicts[i][*l].ln()).collect();

        // calculate gradient
        label.iter().enumerate().for_each(|(i, l)| {
            predicts[i][*l] -= 1.0;
        });
        self.grad_w = &self.grad_w + &(&(&predicts.permute(vec![1, 0]) * feature) / feature.shape[0] as f32);
        self.grad_b = &self.grad_b + &sum_ndarray(&predicts, 0);

        entropy_loss
    }

    pub fn backward(&mut self) {
        self.weight = &self.weight - &self.grad_w;
        self.bias = &self.bias - &self.grad_b;
        self.grad_w.clear();
        self.grad_b.clear();
    }

}

impl Model<usize> for LogitRegression {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let input = vec![feature.clone()];
        let logits = self.forward(&NdArray::new(input));
        let (_, predictions) = argmax(&logits, -1).destroy();
        let predictions: Vec<usize> = predictions.into_iter().map(|i| i as usize).collect();
        println!("p len = {}", predictions.len());
        predictions[0]
    }
}


#[cfg(test)]
mod test {
    use crate::model::Model;

    use super::{NdArray, LogitRegression};

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![0, 1];
        let mut model = LogitRegression::new(batch_of_feature.shape[1], 2, |i| i);
        let loss = model.one_step(&batch_of_feature, &label);
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
        model.backward();

        let loss = model.one_step(&batch_of_feature, &label);
        model.backward();
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
    }

    #[test]
    fn test_something() {
        let x:f32 = 1.0 / 0.0;
        println!("{}", 0.0 == 0.0);
    }

}
