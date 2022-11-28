use super::{Model, utils::{NormType, Penalty, gradient_clip}};
use crate::{ndarray::{NdArray, utils::sum_ndarray}, model::utils::calculate_penalty_value};

pub struct LinearRegression {
    weight: NdArray,
    bias: NdArray,
    feature_size: usize,
    grad_w: NdArray,
    grad_b: NdArray,
    penalty: Option<Penalty>,
}

impl LinearRegression {
    pub fn new<F>(feature_size: usize, penalty: Option<Penalty>, weight_init_fn: F) -> Self
    where F: Fn(Vec<Vec<f32>>) -> Vec<Vec<f32>>
    {
        let weight = vec![vec![0.0f32; feature_size]];
        let bias = NdArray::new(vec![1usize]);
        let weight = NdArray::new(weight_init_fn(weight));
        Self { weight: weight, bias: bias, feature_size: feature_size, grad_w: NdArray::new(vec![1, feature_size]), grad_b: NdArray::new(vec![1]), penalty: penalty}
    }

    pub fn forward(&self, input: &NdArray) -> NdArray {
        assert!(input.dim() == 2);
        assert!(input.shape[1] == self.feature_size);
        // weight: [1, feature_size]
        // input: [bsz, feature_size]
        // bias: [1]
        // return: [bsz, 1]
        &(input * &self.weight.permute(vec![1, 0])) + &self.bias
    }

    fn calculate_gradient(&mut self, feature: &NdArray, label: &Vec<f32>, mut predicts: NdArray) {
        // p = prediction, e = Error, d = partial derivative
        label.iter().zip(predicts.data_as_mut_vector().iter_mut()).for_each(|(l, p)| {
            *p = 2.0 * (*p - *l);
        });
        let mut temp_grad_w = &predicts.permute(vec![1, 0]) * feature;
        temp_grad_w = &temp_grad_w / feature.shape[0] as f32;

        let temp_grad_b = &sum_ndarray(&predicts, 0) / feature.shape[0] as f32;

        self.grad_w = &self.grad_w + &temp_grad_w;
        self.grad_b = &self.grad_b + &temp_grad_b;
    }

    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<f32>, required_grad: Option<bool>) -> Vec<f32> {
        assert!(feature.dim() == 2); // feature: [bsz, feature_size]
        let required_grad = required_grad.unwrap_or(true);
        
        // predicts [bsz, 1]
        let predicts = self.forward(feature);

        // loss
        let penalty_v = if self.penalty.is_some() {
            calculate_penalty_value(&self.weight, self.penalty.unwrap())
        } else {
            0.0
        };
        let mean_squared_error: Vec<f32> = label.iter().enumerate().map(|(i, l)| (predicts[i][0] - l).powf(2.0) + penalty_v).collect();

        
        // calculate gradient
        if required_grad {
            self.calculate_gradient(feature, label, predicts);
        }

        mean_squared_error
    }

    pub fn backward(&mut self, lr: f32, gradient_clip_by_norm: Option<NormType>) {
        if gradient_clip_by_norm.is_some() {
            gradient_clip(&mut self.grad_w, gradient_clip_by_norm.as_ref().unwrap());
            gradient_clip(&mut self.grad_b, gradient_clip_by_norm.as_ref().unwrap());
        }

        self.weight = &self.weight - &(&self.grad_w * lr);
        self.bias = &self.bias - &(&self.grad_b * lr);
        self.grad_w.clear();
        self.grad_b.clear();
    }
}

impl Model<f32> for LinearRegression {
    fn predict(&self, feature: &Vec<f32>) -> f32 {
        let input = vec![feature.clone()];
        let predicts = self.forward(&NdArray::new(input));
        predicts[0][0]
    }
}

#[cfg(test)]
mod test {
    use std::io::{stdout, Write};

    use crate::{
        model::{Model, utils::Penalty, utils::NormType}, 
        dataset::{Dataset, FromPathDataset, DatasetName, dataloader::Dataloader}, utils::{evaluate_regression}};
    use super::{NdArray, LinearRegression};

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let lr = 1e-3;
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![5.8, -6.7];
        let mut model = LinearRegression::new(batch_of_feature.shape[1], None, |i| i);
        let loss = model.one_step(&batch_of_feature, &label, None);
        println!("loss {:?} {} {}", loss, (model.predict(&datas[0]) - label[0]).abs(), (model.predict(&datas[1]) - label[1]).abs());
        model.backward(lr, None);

        let loss = model.one_step(&batch_of_feature, &label, Some(false));
        model.backward(lr, None);
        println!("loss {:?} {} {}", loss, (model.predict(&datas[0]) - label[0]).abs(), (model.predict(&datas[1]) - label[1]).abs());
    }

    #[test]
    fn test_with_tianchi_car_price_dataset() -> std::io::Result<()> {
        let path = ".data/TianchiCarPriceRegression/train_5w.csv";
        let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let mut model = LinearRegression::new(train_dataset.feature_len(), Some(Penalty::RidgeL2(0.1)), |i| i);

        let mut train_dataloader = Dataloader::new(train_dataset, 64, true);

        const EPOCH: usize = 10;
        let mut error_records = vec![];
        for ep in 0..EPOCH {
            let mut losses = vec![];
            for (feature, label) in &mut train_dataloader {
                let loss = model.one_step(&feature, &label, None).iter().sum::<f32>() / feature.shape[0] as f32;
                model.backward(1e-2, Some(NormType::L2(1.0)));
                losses.push(loss);
            }
            let mean_abs_error = evaluate_regression(&test_dataset, &model);
            error_records.push(mean_abs_error);
            let width = ">".repeat(ep * 50 / EPOCH);
            print!("\r{:-<50}\t{:.3}\t{:.3}", width, losses.iter().sum::<f32>() / losses.len() as f32, mean_abs_error);
            stdout().flush()?;
        }
        let (best_ep, best_error) = error_records.iter().enumerate().fold((0, f32::MAX), |s, (i, e)| {
            if *e < s.1 {
                (i, *e)
            } else {
                s
            }
        });
        println!("\n{:?}\nbest ep {} best mean abs error {:.5}", error_records, best_ep, best_error);

        
        Ok(())
    }
}