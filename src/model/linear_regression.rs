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
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![5.8, -6.7];
        let mut model = LinearRegression::new(batch_of_feature.shape[1], None, |_| {});

        for _ in 0..2 {
            let loss = model.one_step(&batch_of_feature, &label, 1e-3, None);
            println!("loss {:?} {} {}", loss, (model.predict(&datas[0]) - label[0]).abs(), (model.predict(&datas[1]) - label[1]).abs());
        }
    }

    #[test]
    fn test_with_tianchi_car_price_dataset() -> std::io::Result<()> {
        let path = ".data/TianchiCarPriceRegression/train_5w.csv";
        let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let mut model = LinearRegression::new(train_dataset.feature_len(), Some(Penalty::RidgeL2(0.1)), |_| {});

        let mut train_dataloader = Dataloader::new(train_dataset, 64, true, None);

        const EPOCH: usize = 10;
        let mut error_records = vec![];
        for ep in 0..EPOCH {
            let mut losses = vec![];
            for (feature, label) in train_dataloader.iter_mut() {
                let loss = model.one_step(&feature, &label, 1e-2, Some(NormType::L2(1.0)));
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