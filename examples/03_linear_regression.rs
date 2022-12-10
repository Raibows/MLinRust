use std::io::{stdout, Write};
use mlinrust::model::linear_regression::LinearRegression;
use mlinrust::model::utils::{NormType, Penalty};
use mlinrust::dataset::{Dataset, DatasetName, FromPathDataset};
use mlinrust::dataset::dataloader::Dataloader;
use mlinrust::utils::evaluate_regression;


fn main() -> std::io::Result<()> {
    let path = ".data/TianchiCarPriceRegression/train_5w.csv";
        let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
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