use crate::ndarray::NdArray;

pub mod linear;
pub mod activation;


pub trait NNBackPropagation {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray;
    
    fn backward(&mut self, bp_grad: &NdArray) -> NdArray;

    fn step(&mut self) {
        
    }
}