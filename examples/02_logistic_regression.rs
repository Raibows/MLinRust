use std::io::{stdout, Write};
use mlinrust::model::logistic_regression::LogisticRegression;
use mlinrust::model::utils::{NormType, Penalty};
use mlinrust::dataset::{Dataset, DatasetName, FromPathDataset, utils::{normalize_dataset, ScalerType}};
use mlinrust::dataset::dataloader::Dataloader;
use mlinrust::ndarray::utils::argmax;
use mlinrust::ndarray::NdArray;
use mlinrust::utils::{evaluate, RandGenerator};

fn main() -> std::io::Result<()> {
    let path = ".data/MobilePhonePricePredict/train.csv";
        let mut dataset: Dataset<usize> = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);

        normalize_dataset(&mut dataset, ScalerType::Standard);

        let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        
        let mut rng = RandGenerator::new(0);

        // initing weights is also important, or the gradients may not be good
        let mut model = LogisticRegression::new(train_dataset.feature_len(), train_dataset.class_num(), Some(Penalty::RidgeL2(1e-1)),|item| item.iter_mut().for_each(move |i| {*i = rng.gen_f32()}));

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
            print!("\r{width:-<100}\t{:.3}\t{acc:.3}", losses.iter().sum::<f32>() / losses.len() as f32);
            stdout().flush()?;
        }
        let acc = best_acc.iter().fold(0.0, |s, i| f32::max(s, *i));
        let best_ep = argmax(&NdArray::new(best_acc), 0);
        println!("\nbest acc = {acc} ep {}", best_ep[0][0]);
        assert!(acc > 0.9); // gradient clip greatly helps it

        Ok(())
}