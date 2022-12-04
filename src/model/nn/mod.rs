use crate::ndarray::NdArray;
use super::Model;
use super::utils::{NormType, Penalty};

pub mod linear;
pub mod activation;
pub mod criterion;

pub trait NNBackPropagation {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray;
    
    fn backward(&mut self, bp_grad: &NdArray) -> NdArray;

    fn step(&mut self, reduction: usize, lr: f32, gradient_clip_by_norm: Option<NormType>) {
        
    }
}


/// build your DNN from supported NN modules
#[derive(Debug, Clone, Copy)]
pub enum NNmodule {
    /// (insize, outsize, penalty)
    Linear((usize, usize, Option<Penalty>)),
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
            NNmodule::Linear((insize, outsize, penalty)) => Box::new(linear::Linear::new(insize, outsize, penalty)),
            NNmodule::Relu => Box::new(activation::Relu::new()),
            NNmodule::Sigmoid => Box::new(activation::Sigmoid::new()),
            NNmodule::Tanh => Box::new(activation::Tanh::new()),
        }
    }
}

impl NNBackPropagation for NeuralNetwork {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        // let out = self.layers[0].
        // for nn in self.layers.iter_mut() {
        //     out = nn.forward(&out, required_grad);
        // }
        // out
        todo!()
    }

    fn backward(&mut self, bp_grad: &NdArray) -> NdArray {
        // let mut back_grad;
        for nn in self.layers.iter_mut().rev() {
            // back_grad = nn.backward(bp_grad)
        }

        todo!()
    }
}


#[cfg(test)]
mod test {
    use std::collections::LinkedList;

    use crate::ndarray::NdArray;

    use super::{NeuralNetwork, NNBackPropagation};

    #[test]
    fn test_nn_forward() {
        let mut model = NeuralNetwork::new(vec![]);
        // model.backward(&NdArray::default());
    }

    #[test]
    fn test_linkedlist() {
        struct Node {
            v: &'static str,
        }

        let mut header = LinkedList::new();
        // let mut footer = LinkedList::new();
        header.push_back(Node {v: "a"});
        header.push_back(Node {v: "b"});
        
        let item = header.back().unwrap();
        println!("last {}", item.v);

        for item in header.iter() {
            
        }
    }
}