use crate::{ndarray::{NdArray}, model::utils::{Penalty, NormType, calculate_penalty_grad, gradient_clip}};

use super::{NNBackPropagation};


/// Linear is a nn module, similar to torch.nn.Linear
pub struct Linear {
    weight: NdArray,
    bias: NdArray,
    grad_w: NdArray,
    grad_b: NdArray,
    // current node gradient
    temp_grad_w: Option<NdArray>,
    temp_grad_b: Option<NdArray>,
    penalty: Option<Penalty>
}

impl NNBackPropagation for Linear {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        // input should be [bsz, hidden]
        
        // forward to get outputs
        let outputs = input * &self.weight.permute(vec![1, 0])  + &self.bias;

        // calculate current node gradient
        if required_grad {
            self.save_grad_graph(input);
        }

        outputs
    }

    fn forward_as_borrow(&self, input: &NdArray) -> NdArray {
        // input should be [bsz, hidden]
        
        // forward to get outputs
        input * &self.weight.permute(vec![1, 0])  + &self.bias
    }

    fn backward(&mut self, bp_grad: NdArray) -> NdArray {
        // bp [bsz, outsize]
        // temp [bsz, insize]
        // println!("first loss {}", bp_grad);

        self.grad_w += bp_grad.permute(vec![1, 0]) * &self.temp_grad_w.take().unwrap();

        let mut temp_grad_b = &self.temp_grad_b.take().unwrap() * &bp_grad;
        temp_grad_b.squeeze(0);
        self.grad_b += temp_grad_b;

        bp_grad * &self.weight
    }

    fn step(&mut self, reduction: usize, lr: f32, gradient_clip_by_norm: Option<NormType>) {

        if self.penalty.is_some() {
            self.grad_w += calculate_penalty_grad(&self.weight, self.penalty.unwrap());
        }

        self.grad_w /= reduction as f32;
        self.grad_b /= reduction as f32;

        if gradient_clip_by_norm.is_some() {
            gradient_clip(&mut self.grad_w, gradient_clip_by_norm.as_ref().unwrap());
            gradient_clip(&mut self.grad_b, gradient_clip_by_norm.as_ref().unwrap());
        }
        // println!("weight grad {}", self.grad_w);
        // println!("bias grad {}", self.grad_b);

        self.weight -= &self.grad_w * lr;
        self.bias -= &self.grad_b * lr;

        self.grad_w.clear();
        self.grad_b.clear();
    }

    fn weight_mut_borrow(&mut self) -> &mut [f32] {
        self.weight.data_as_mut_vector()
    }
}


impl Linear {
    /// create a matrix with shape \[n, m\] + bias \[n\]
    /// * insize: m
    /// * outsize: n
    pub fn new(insize: usize, outsize: usize, penalty: Option<Penalty>) -> Self {
        let weight = NdArray::new(vec![outsize, insize]);
        let bias = NdArray::new(vec![outsize]);
        Self { weight: weight.clone(), bias: bias.clone(), grad_w: weight, grad_b: bias, temp_grad_w: None, temp_grad_b: None, penalty: penalty }
    }

    /// save the activation values
    fn save_grad_graph(&mut self, input: &NdArray) {
        // input [bsz, hidden]

        self.temp_grad_w = Some(input.clone());
        self.temp_grad_b = Some(NdArray::new(vec![vec![1.0; input.shape[0]]]));
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use crate::{ndarray::NdArray, model::{nn::criterion::MeanSquaredError}};

    use super::{Linear, NNBackPropagation};

    #[test]
    fn test_back_propagation() {
        let mut x = NdArray::new(vec![3.5; 12]);
        let target = vec![-1.0; 4];
        x.reshape(vec![4usize, 3]);
        
        let mut l1 = Linear::new(3, 5, None);
        let mut l2 = Linear::new(5, 1, None);
        for _ in 0..5 {
            let mut criterion = MeanSquaredError::new();
            let outs = l1.forward(&x, true);
            println!("out1 {outs}");
            let outs = l2.forward(&outs, true);
            println!("out2 {outs}");

            let loss = criterion.forward(outs, &target);
            println!("loss {loss}");
            let bp_grad = l2.backward(loss);
            println!("bp_grad1 {bp_grad}");
            let bp_grad = l1.backward(bp_grad);
            println!("bp_grad2 {bp_grad}");
            println!("l1 weight grad {}", l2.grad_w);
            println!("l1 bias grad {}", l2.grad_b);
            l2.step(4, 1e-1, None);
            l1.step(4, 1e-1, None);
        }

        println!("l1 weight {} bias {}", l1.weight, l1.bias);
        println!("l2 weight {} bias {}", l2.weight, l2.bias);
        // the bp is not working, since the initial weights are exactly zero
        // neither l1 nor l2 could get useful gradient
        // so it is important to init weights

    }
}