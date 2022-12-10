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
    use std::io::{stdout, Write};

    use crate::{
        model::{Model, utils::Penalty, utils::NormType}, 
        dataset::{Dataset, FromPathDataset, DatasetName, dataloader::Dataloader}, utils::evaluate, ndarray::utils::argmax};
    use super::{NdArray, LogisticRegression};

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

    #[test]
    fn test_with_mobile_phone_price_dataset() -> std::io::Result<()> {
        let path = ".data/MobilePhonePricePredict/train.csv";
        let mut dataset: Dataset<usize> = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
        dataset.shuffle(0);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let mut model = LogisticRegression::new(train_dataset.feature_len(), train_dataset.class_num(), Some(Penalty::RidgeL2(1e-1)),|_| {});

        let mut train_dataloader = Dataloader::new(train_dataset, 8, true, None);

        const EPOCH: usize = 3000;
        let mut best_acc = vec![];
        for ep in 0..EPOCH {
            let mut losses = vec![];
            let lr = match ep {
                i if i < 200 => 1e-3,
                _ => 2e-3,
            };
            for (feature, label) in train_dataloader.iter_mut() {
                let loss = model.one_step(&feature, &label, lr, Some(NormType::L2(1.0)));
                losses.push(loss);
            }
            let (_, acc) = evaluate(&test_dataset, &model);
            best_acc.push(acc);
            let width = ">".repeat(ep * 100 / EPOCH);
            print!("\r{:-<100}\t{:.3}\t{:.3}", width, losses.iter().sum::<f32>() / losses.len() as f32, acc);
            stdout().flush()?;
        }
        let acc = best_acc.iter().fold(0.0, |s, i| f32::max(s, *i));
        let best_ep = argmax(&NdArray::new(best_acc), 0);
        println!("\nbest acc = {} ep {}", acc, best_ep[0][0]);
        assert!(acc > 0.75); // gradient clip greatly helps it

        Ok(())

    }
}