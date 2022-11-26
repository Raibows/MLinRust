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

    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<usize>, required_grad: Option<bool>) -> Vec<f32> {
        assert!(feature.dim() == 2); // feature: [bsz, feature_size]
        let required_grad = required_grad.unwrap_or(true);
        
        // after softmax logits
        let mut predicts = self.forward(feature); //
        softmax(&mut predicts, -1);

        // loss
        let entropy_loss: Vec<f32> = label.iter().enumerate().map(|(i, l)| - predicts[i][*l].ln()).collect();

        // calculate gradient
        if required_grad {
            label.iter().enumerate().for_each(|(i, l)| {
                predicts[i][*l] -= 1.0;
            });
            self.grad_w = &self.grad_w + &(&(&predicts.permute(vec![1, 0]) * feature) / feature.shape[0] as f32);
            self.grad_b = &self.grad_b + &sum_ndarray(&predicts, 0);
        }

        entropy_loss
    }

    pub fn backward(&mut self, lr: f32) {
        self.weight = &self.weight - &(&self.grad_w * lr);
        self.bias = &self.bias - &(&self.grad_b * lr);
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
        predictions[0]
    }
}


#[cfg(test)]
mod test {
    use crate::{
        model::Model, 
        dataset::{Dataset, FromPathDataset, DatasetName, dataloader::Dataloader}, utils::evaluate};
    use super::{NdArray, LogitRegression};

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let lr = 1e-1;
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![0, 1];
        let mut model = LogitRegression::new(batch_of_feature.shape[1], 2, |i| i);
        let loss = model.one_step(&batch_of_feature, &label, None);
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
        model.backward(lr);

        let loss = model.one_step(&batch_of_feature, &label, Some(false));
        model.backward(lr);
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
    }

    #[test]
    fn test_with_mobile_phone_price_dataset() {
        let path = ".data/MobilePhonePricePredict/train.csv";
        let mut dataset: Dataset<usize> = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset);
        dataset.shuffle(0);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let mut model = LogitRegression::new(train_dataset.feature_len(), train_dataset.class_num(), |i| i);

        let mut train_dataloader = Dataloader::new(train_dataset, 16, true);

        for epoch in 0..200 {
            let mut losses = vec![];
            let lr = match epoch {
                i if i < 100 => 1e-6,
                i if i < 200 => 1e-6,
                i => 1e-6 * (i - 200 + 1) as f32,
            };
            for (feature, label) in &mut train_dataloader {
                let loss: f32 = model.one_step(&feature, &label, None).iter().sum::<f32>() / feature.shape[0] as f32;
                model.backward(lr);
                print!("\repoch {} loss {:.3}", epoch, loss);
                losses.push(loss);
            }
            println!("\nepoch {} avg.loss = {:.3}", epoch, losses.iter().sum::<f32>() / losses.len() as f32);
        }

        println!("stating testing");
        let (correct, acc) = evaluate(&test_dataset, &model);
        println!("correct {} / {}, acc {:.5}", correct, test_dataset.len(), acc);


    }


    #[test]
    fn test_something() {
        let x:f32 = 1.0 / 0.0;
        println!("{}", 0.0 == 0.0);
    }

}
