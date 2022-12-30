use std::time::{Instant};
use std::io::{Write, stdout};
use mlinrust::dataset::{Dataset, DatasetName, FromPathDataset, dataloader::{Dataloader}};
use mlinrust::model::nn::{NeuralNetwork, NNmodule, NNBackPropagation};
use mlinrust::model::utils::{NormType, Penalty};
use mlinrust::model::nn::criterion::MeanSquaredError;
use mlinrust::utils::evaluate_regression;

fn main() -> std::io::Result<()> {

    let path = ".data/TianchiCarPriceRegression/train_5w.csv";
    let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
    let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
    let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

    let blocks = vec![
        NNmodule::Linear(train_dataset.feature_len(), 512, Some(Penalty::RidgeL2(1e-1))), 
        NNmodule::Tanh,
        NNmodule::Linear(512, 128, Some(Penalty::RidgeL2(1e-1))), 
        NNmodule::Tanh,
        NNmodule::Linear(128, 128, Some(Penalty::RidgeL2(1e-1))), 
        NNmodule::Relu,
        NNmodule::Linear(128, 16, Some(Penalty::RidgeL2(1e-1))), 
        NNmodule::Tanh,
        NNmodule::Linear(16, 1, None)
    ];
    let mut model = NeuralNetwork::new(blocks);
    model.weight_init(None);
    let mut criterion = MeanSquaredError::new();

    let mut train_dataloader = Dataloader::new(train_dataset, 64, true, Some(0));
    let mut test_dataloader = Dataloader::new(test_dataset, 128, false, None);

    const EPOCH: usize = 10;
    let mut error_records = vec![];

    for ep in 0..EPOCH {
        let mut losses = vec![];
        let start = Instant::now();
        for (feature, label) in train_dataloader.iter_mut() {
            let logits = model.forward(&feature, true);
            let grad = criterion.forward(logits, &label);
            model.backward(grad);
            model.step(label.len(), 1e-1, Some(NormType::L2(1.0)));
            losses.push(criterion.avg_loss);
        }
        let train_time = Instant::now() - start;
        let start = Instant::now();
        let mean_abs_error = evaluate_regression(&mut test_dataloader, &model);
        let test_time = Instant::now() - start;
        error_records.push(mean_abs_error);
        let width = ">".repeat(ep * 50 / EPOCH);
        print!("\r{width:-<50}\t{:.3}\t{mean_abs_error:.3}\t", losses.iter().sum::<f32>() / losses.len() as f32);
        println!("\ntime cost train {:.3} test {:.3}", train_time.as_secs_f64(), test_time.as_secs_f64());
        stdout().flush()?;
    }
    let (best_ep, best_error) = error_records.iter().enumerate().fold((0, f32::MAX), |s, (i, e)| {
        if *e < s.1 {
            (i, *e)
        } else {
            s
        }
    });
    println!("\n{error_records:?}\nbest ep {best_ep} best mean abs error {best_error:.5}");

    Ok(())
}