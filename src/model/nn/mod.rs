use crate::ndarray::NdArray;
use crate::utils::RandGenerator;
use super::Model;
use super::utils::{NormType, Penalty};

pub mod linear;
pub mod activation;
pub mod criterion;

pub trait NNBackPropagation {
    /// nn forward
    /// * require_grad: if it is set to true, it will save the necessary grad graph, and it is also the reason for &mut self
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray;
    
    /// forward but without graph and it only requires immutable reference of self
    fn forward_as_borrow(&self, input: &NdArray) -> NdArray;

    /// calculate the gradidents and save them to grad_w or grad_b
    fn backward(&mut self, bp_grad: NdArray) -> NdArray;

    /// update the weights and bias with the grad_w and grad_b, respectively
    /// * reduction:
    ///     * len(batch): average over the batch
    ///     * 1: sum the batch
    /// * lr: learning rate
    /// * gradient_clip_by_norm: gradient clippling by NormType, default is None
    fn step(&mut self, _reduction: usize, _lr: f32, _gradient_clip_by_norm: Option<NormType>) {
        
    }

    /// mutablly borrow the raw data of the weights
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
    /// create a neural network from the given ModuleList
    /// * blocks: vector of enum NNModule
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

    /// init the weights with random number from \[0, 1.0\]
    /// * seed: default is set to 0
    pub fn weight_init(&mut self, seed: Option<usize>) {
        let mut rng = RandGenerator::new(seed.unwrap_or(0));
        for nn in self.layers.iter_mut() {
            nn.weight_mut_borrow().iter_mut().for_each(|i| *i = rng.gen_f32());
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

    fn predict_with_batch(&self, features: &NdArray) -> Vec<f32> {
        let res = self.forward_as_borrow(features);
        res.destroy().1
    }
}


#[cfg(test)]
mod test {
    use crate::ndarray::NdArray;
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
        model.weight_init(None);

        let mut criterion = criterion::CrossEntropyLoss::new();

        for _ in 0..10 {
            let logits = model.forward(&datas, true);
            let grad = criterion.forward(logits, &label);
            model.backward(grad);
            model.step(2, 1e-1, None);
            println!("avg loss {} loss {:?}", criterion.avg_loss, criterion.loss);
        }
    }
}