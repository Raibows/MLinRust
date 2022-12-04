use crate::{ndarray::{NdArray, utils::softmax}};

pub struct CrossEntropyLoss {
    pub loss: Vec<f32>,
    pub avg_loss: f32,
}


impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self { loss: vec![], avg_loss: 0.0 }
    }

    /// calculate the loss
    /// 
    /// logits: the direct output of forward (without softmax)
    ///  
    /// return: grad for back propagation
    pub fn forward(&mut self, mut logits: NdArray, label: &Vec<usize>) -> NdArray {
        softmax(&mut logits, -1);

        self.loss = label.iter().enumerate().map(|(i, l)| - f32::max(logits[i][*l], 1e-9).ln()).collect();
        self.avg_loss = self.loss.iter().sum::<f32>() / label.len() as f32;

        // need logits after softmax
        label.iter().enumerate().for_each(|(i, l)| {
            logits[i][*l] -= 1.0;
        });

        logits
    }
}

pub struct MeanSquaredError {
    pub loss: Vec<f32>,
    pub avg_loss: f32,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self { loss: vec![], avg_loss: 0.0 }
    }

    /// calculate the mean squared error
    /// 
    /// logits: the direct output of forward (you don't need it in training, so take the owner here)
    ///  
    /// return: grad for back propagation
    pub fn forward(&mut self, mut logits: NdArray, label: &Vec<f32>) -> NdArray {
        self.loss = label.iter().enumerate().map(|(i, l)| (logits[i][0] - l).powf(2.0)).collect();
        self.avg_loss = self.loss.iter().fold(0.0, |s, i| s + *i) / label.len() as f32;

        label.iter().zip(logits.data_as_mut_vector().iter_mut()).for_each(|(l, p)| {
            *p = 2.0 * (*p - *l);
        });

        logits // is the bp_grad
    }
}