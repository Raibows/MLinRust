use super::{NNBackPropagation};
use crate::ndarray::{NdArray, utils::{sigmoid, tanh, relu}};


macro_rules! nn_no_param_initialization {
    ($name:tt) => {
        pub struct $name {
            temp_grad: Option<NdArray>,
        }

        impl $name {
            pub fn new() -> Self {
                Self {temp_grad: None}
            }
        }
    }
}

macro_rules! nn_write_backward {
    () => {
        fn backward(&mut self, bp_grad: &NdArray) -> NdArray {
            bp_grad.point_multiply(&self.temp_grad.take().unwrap())
        }
    };
}

nn_no_param_initialization!(Relu);
nn_no_param_initialization!(Sigmoid);
nn_no_param_initialization!(Tanh);


impl NNBackPropagation for Relu {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        let out = relu(input);

        if required_grad {
            let mut temp_grad = NdArray::new(vec![1.0; NdArray::total_num(&input.shape)]);
            out.data_as_vector().iter().zip(temp_grad.data_as_mut_vector().iter_mut())
            .for_each(|(o, g)| {
                if *o == 0.0 {
                    *g = 0.0;
                }
            });
            self.temp_grad = Some(temp_grad);
        }
            
        out
    }
    
    nn_write_backward!();
}

impl NNBackPropagation for Sigmoid {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        let out = sigmoid(input);

        if required_grad {
            let mut grad = out.clone();
            grad.data_as_mut_vector().iter_mut().for_each(|i| {
                // sigmoid(x)(1 - sigmoid(x))
                *i *= 1.0 - *i;
            });
            self.temp_grad = Some(grad);
        }

        out
    }

    nn_write_backward!();
}

impl NNBackPropagation for Tanh {
    fn forward(&mut self, input: &NdArray, required_grad: bool) -> NdArray {
        let out = tanh(input);

        if required_grad {
            let mut grad = out.clone();
            grad.data_as_mut_vector().iter_mut().for_each(|i| {
                // 1 - tanh^2(x)
                *i = 1.0 - i.powi(2);
            });
        }

        out
    }

    nn_write_backward!();
}