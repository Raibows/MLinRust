use rand::Rng;

use crate::ndarray::NdArray;
use super::Model;
use super::utils::{NormType, Penalty};

pub mod linear;
pub mod activation;
pub mod criterion;

pub trait NNBackPropagation {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray;
    
    fn forward_as_borrow(&self, input: &NdArray) -> NdArray;

    fn backward(&mut self, bp_grad: NdArray) -> NdArray;

    fn step(&mut self, _reduction: usize, _lr: f32, _gradient_clip_by_norm: Option<NormType>) {
        
    }

    fn weight_mut_borrow(&mut self) -> &mut [f32] {
        todo!()
    }
}

/// build your DNN from supported NN modules
#[derive(Debug, Clone, Copy)]
pub enum NNmodule {
    /// (insize, outsize, penalty)
    Linear(usize, usize, Option<Penalty>),
    Relu,
    Sigmoid,
    Tanh,
}


pub struct NeuralNetwork {
    layers: Vec<Box<dyn NNBackPropagation>>,
}

impl NeuralNetwork {
    pub fn new(blocks: Vec<NNmodule>, ) -> Self {
        assert!(blocks.len() > 0);
        let mut layers = vec![];
        for item in blocks {
            layers.push(Self::create_layer(item));
        }
        Self { layers: layers }
    }

    fn create_layer(module: NNmodule) -> Box<dyn NNBackPropagation> {
        match module {
            NNmodule::Linear(insize, outsize, penalty) => Box::new(linear::Linear::new(insize, outsize, penalty)),
            NNmodule::Relu => Box::new(activation::Relu::new()),
            NNmodule::Sigmoid => Box::new(activation::Sigmoid::new()),
            NNmodule::Tanh => Box::new(activation::Tanh::new()),
        }
    }

    pub fn weight_init(&mut self) {
        let mut rng = rand::thread_rng();
        for nn in self.layers.iter_mut() {
            nn.weight_mut_borrow().iter_mut().for_each(|i| *i = rng.gen());
        }
    }
}

impl NNBackPropagation for NeuralNetwork {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        let mut out = self.layers[0].forward(input, required_grad);
        for nn in self.layers.iter_mut().skip(1) {
            out = nn.forward(&out, required_grad);
        }
        out
    }

    fn forward_as_borrow(&self, input: &NdArray) -> NdArray {
        let mut out = self.layers[0].forward_as_borrow(input);
        for nn in self.layers.iter().skip(1) {
            out = nn.forward_as_borrow(&out);
        }
        out
    }

    /// backward chain, calculate gradient for each module
    /// bp_grad: usually be the output of criterion.forward
    fn backward(&mut self, mut bp_grad: NdArray) -> NdArray {
        // let mut back_grad;
        for nn in self.layers.iter_mut().rev() {
            bp_grad = nn.backward(bp_grad);
        }
        bp_grad
    }

    fn step(&mut self, reduction: usize, lr: f32, gradient_clip_by_norm: Option<NormType>) {
        for nn in self.layers.iter_mut() {
            nn.step(reduction, lr, gradient_clip_by_norm);
        }
    }
}

impl Model<f32> for NeuralNetwork {
    fn predict(&self, feature: &Vec<f32>) -> f32 {
        let input = vec![feature.clone()];
        let input = NdArray::new(input);
        let predicts = self.forward_as_borrow(&input);
        predicts[0][0]
    }
}


#[cfg(test)]
mod test {
    use std::io::{stdout, Write};
    use crate::dataset::{Dataset, DatasetName, dataloader, FromPathDataset};
    use crate::model::nn::criterion::MeanSquaredError;
    use crate::model::utils::{Penalty, NormType};
    use crate::ndarray::NdArray;
    use crate::utils::evaluate_regression;
    use super::{NeuralNetwork, NNBackPropagation, NNmodule, criterion};

    #[test]
    fn test_nn_forward_backward() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let datas = NdArray::new(datas);
        let label = vec![0usize, 1];
        let mut model = NeuralNetwork::new(vec![NNmodule::Linear(5, 3, None), NNmodule::Relu, NNmodule::Linear(3, 2, None)]);
        model.weight_init();

        let mut criterion = criterion::CrossEntropyLoss::new();

        for _ in 0..10 {
            let logits = model.forward(&datas, true);
            let grad = criterion.forward(logits, &label);
            model.backward(grad);
            model.step(2, 1e-1, None);
            println!("avg loss {} loss {:?}", criterion.avg_loss, criterion.loss);
        }
    }

    #[test]
    fn test_with_regression_task() -> std::io::Result<()> {
        use std::time::{Instant};

        let path = ".data/TianchiCarPriceRegression/train_5w.csv";
        let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let blocks = vec![
            NNmodule::Linear(train_dataset.feature_len(), 128, Some(Penalty::RidgeL2(1e-1))), 
            NNmodule::Relu, 
            NNmodule::Linear(128, 16, Some(Penalty::RidgeL2(1e-1))), 
            NNmodule::Relu,
            NNmodule::Linear(16, 1, Some(Penalty::RidgeL2(1e-1)))
        ];
        let mut model = NeuralNetwork::new(blocks);
        model.weight_init();
        let mut criterion = MeanSquaredError::new();

        let mut train_dataloader = dataloader::Dataloader::new(train_dataset, 64, true);

        const EPOCH: usize = 10;
        let mut error_records = vec![];
        for ep in 0..EPOCH {
            let mut losses = vec![];
            let start = Instant::now();
            for (feature, label) in &mut train_dataloader {
                let logits = model.forward(&feature, true);
                let grad = criterion.forward(logits, &label);
                model.backward(grad);
                model.step(label.len(), 1e-1, Some(NormType::L2(1.0)));
                losses.push(criterion.avg_loss);
            }
            let train_time = Instant::now() - start;
            let start = Instant::now();
            let mean_abs_error = evaluate_regression(&test_dataset, &model);
            let test_time = Instant::now() - start;
            error_records.push(mean_abs_error);
            let width = ">".repeat(ep * 50 / EPOCH);
            print!("\r{:-<50}\t{:.3}\t{:.3}\t", width, losses.iter().sum::<f32>() / losses.len() as f32, mean_abs_error);
            println!("\ntime cost train {:.1} test {:.1}", train_time.as_secs_f32(), test_time.as_secs());
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