use crate::ndarray::{NdArray};

use super::NNBackPropagation;

/// Linear is a nn module, similar to torch.nn.Linear
pub struct Linear {
    weight: NdArray,
    bias: NdArray,
    grad_w: NdArray,
    grad_b: NdArray,
    // current node gradient
    temp_grad_w: Option<NdArray>,
    temp_grad_b: Option<NdArray>,
}

impl NNBackPropagation for Linear {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        // input should be [bsz, hidden]
        
        // forward to get outputs
        let outputs = input * &self.weight.permute(vec![1, 0])  + &self.bias;

        // calculate current node gradient
        if required_grad {
            self.calculate_current_node_gradient(input);
        }

        outputs
    }

    fn backward(&mut self, bp_grad: &NdArray) -> NdArray {
        // bp [bsz, outsize]
        // temp [bsz, insize]
        self.grad_w = bp_grad.permute(vec![1, 0]) * &self.temp_grad_w.take().unwrap();

        self.grad_b = &self.temp_grad_b.take().unwrap() * bp_grad;
        self.grad_b.squeeze(0);

        bp_grad * &self.weight
    }
}


impl Linear {
    /// create a matrix with shape \[n, m\] + bias \[n\]
    /// 
    /// insize: m
    /// 
    /// outsize: n
    pub fn new(insize: usize, outsize: usize) -> Self {
        let weight = NdArray::new(vec![outsize, insize]);
        let bias = NdArray::new(vec![outsize]);
        Self { weight: weight.clone(), bias: bias.clone(), grad_w: weight, grad_b: bias, temp_grad_w: None, temp_grad_b: None }
    }

    fn calculate_current_node_gradient(&mut self, input: &NdArray) {
        // input [bsz, hidden]
        // no average here. should I take average of the gradient for every step? or just in the final step?

        self.temp_grad_w = Some(input.clone());
        self.temp_grad_b = Some(NdArray::new(vec![vec![1.0; input.shape[0]]]));
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use crate::ndarray::NdArray;

    use super::{Linear, NNBackPropagation};

    #[test]
    fn test_back_propagation() {
        let mut x = NdArray::new(vec![1.0; 12]);
        let mut target = NdArray::new(vec![-1.0; 4]);
        x.reshape(vec![4usize, 3]);
        target.reshape(vec![4, -1]);
        
        let mut l1 = Linear::new(3, 5);
        let mut l2 = Linear::new(5, 1);

        let outs = l1.forward(&x, true);
        println!("out1 {}", outs);
        let outs = l2.forward(&outs, true);
        println!("out2 {}", outs);

        let loss = (&target - &outs).point_multiply(&(target - outs));
        let bp_grad = l2.backward(&loss);
        println!("bp_grad1 {}", bp_grad);
        let bp_grad = l1.backward(&bp_grad);
        println!("bp_grad2 {}", bp_grad);

        println!("l1 weight {}", l1.weight);
    }
}