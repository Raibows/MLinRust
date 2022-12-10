use crate::{ndarray::{NdArray, utils::sum_ndarray}, dataset::{Dataset, dataloader::Dataloader}};
use super::{utils::{Penalty, calculate_penalty_grad, calculate_penalty_value}, Model};


#[derive(Clone, Copy, Debug)]
pub enum SVMLoss {
    Hinge, // max(0, 1 - z)
    Exponential, // exp(-z)
    Logistic, // log (1 + exp(-z))
}

pub struct SupportVectorMachine { 
    // linear SVM, for kernel functions, it is too complicated for programming😥
    lambda: f32,
    lr: f32,
    bsz: Option<usize>,
    weight: NdArray,
    bias: NdArray,
}


impl SupportVectorMachine {
    pub fn new(class_num: usize, feature_num: usize, lr: f32,  bsz: Option<usize>, lambda: f32) -> Self {
        // lambda * (regularization(w) or marginal width) + 1/batch Loss
        assert_eq!(class_num, 2, "only supports binary classification task but got {} classes", class_num);
        Self { lambda: lambda, lr: lr, bsz: bsz, weight: NdArray::new(vec![1, feature_num]), bias: NdArray::new(vec![1]) }
    }

    fn forward(&self, x: &NdArray) -> NdArray {
        // x [bsz, feature]
        // w [class_num, feature]
        // return [bsz, 1]
        let res = x * self.weight.permute(vec![1, 0]) + &self.bias;
        res
    }

    fn gradient_backward(&mut self, loss: SVMLoss, x: &NdArray, y: &Vec<usize>) -> f32 {
        // return average loss
        // update gradient
        let bsz = x.shape[0];
        let y: Vec<f32> = y.iter().map(|i| if *i == 0 {-1.0} else {1.0}).collect();
        let mut z = self.forward(x);

        z.data_as_mut_vector().iter_mut().zip(y.iter()).for_each(|(p, y)| {
            *p = *p * *y - 1.0;
        });
        
        // the loss part
        let (losses, grad_w, grad_b) = match loss {
            SVMLoss::Hinge => {
                let losses: Vec<f32> = z.data_as_mut_vector().iter().map(|i| {f32::max(0.0, 1.0 - i)}).collect();
                let partial_mask: Vec<f32> = losses.iter().zip(y.iter()).map(|(l, y)| {
                    if *l > 0.0 {
                        -y
                    } else {
                        0.0
                    }
                }).collect();
                let mut partial_mask = NdArray::new(partial_mask);
                partial_mask.reshape(vec![1, -1]);
                
                let grad_b = sum_ndarray(&partial_mask, 1);
                let grad_w = partial_mask * x;

                (losses, grad_w, grad_b)
            },
            _ => todo!(),
        };

        // lambda * regularization or marginal width
        let structural_risk = calculate_penalty_grad(&self.weight, Penalty::LassoL1(self.lambda));
        let margin = calculate_penalty_value(&self.weight, Penalty::LassoL1(self.lambda));

        // reduction
        let grad_w = structural_risk + grad_w / bsz as f32;
        let grad_b = grad_b / bsz as f32;
        let avg_loss = losses.iter().fold(0.0, |s, i| s + i) / bsz as f32;
        self.weight = &self.weight - grad_w * self.lr;
        self.bias = &self.bias - grad_b * self.lr;

        avg_loss + margin
    }

    pub fn train(&mut self, dataset: Dataset<usize>, epoch: usize, loss: SVMLoss, early_stop: bool) {
        assert!(dataset.class_num() == 2);
        let bsz = self.bsz.unwrap_or(dataset.len());
        let mut dataloader = Dataloader::new(dataset, bsz, true, None);

        let mut early_stop_loss: Vec<f32> = vec![];

        for ep in 0..epoch {
            let mut avg_loss = vec![];
            for (x, y) in dataloader.iter_mut() {
                let loss = self.gradient_backward(loss, &x, &y);
                avg_loss.push(loss);
            }
            let avg_loss = avg_loss.iter().sum::<f32>() / avg_loss.len() as f32;
            println!("ep {}/{}\t{:?} {:.5}", ep, epoch, loss, avg_loss);

            if ! early_stop{
                continue;
            }

            early_stop_loss.push(avg_loss);
            if ep > usize::min(10, epoch / 3) &&
            early_stop_loss.len() > 10 {
                if self.lr < 1e-5 {
                    println!("early stopping!");
                    break;
                } else {
                    let decrease = early_stop_loss.windows(2).rev().take(10).fold(0.0, |s, i| {
                        s + (i[0] - i[1]) / i[1]
                    });
                    if decrease > -0.05 {
                        let old = self.lr;
                        self.lr *= 0.9; // discount
                        println!("adjust lr from {} to {}", old, self.lr);
                        early_stop_loss.clear();
                    } else {
                        let old = self.lr;
                        self.lr *= 1.0 + f32::min(f32::max(-decrease - 0.2, 0.0), 0.5); // < 1.5
                        println!("adjust lr from {} to {}", old, self.lr);
                        early_stop_loss.clear();
                    }
                }
            }
        }
    }
}

impl Model<usize> for SupportVectorMachine {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let mut x = NdArray::new(feature.clone());
        x.reshape(vec![1, -1]);
        let p = self.forward(&x);
        if p[0][0] < 0.0 {
            0
        } else {
            1
        }
    }
}



#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::{dataset::Dataset, utils::evaluate};

    use super::{SupportVectorMachine, SVMLoss};

    #[test]
    fn test_with_2d_points() {
        let datas = vec![
            vec![1.0, 3.0],
            vec![2.0, 3.0],
            vec![1.0, 2.0],
            vec![4.0, 0.0],
            vec![3.0, 0.0],
            vec![3.0, -1.0],
            vec![3.0, 0.5],
        ];
        let labels = vec![0usize, 0, 0, 1, 1, 1, 1];
        let mut label_map = HashMap::new();
        label_map.insert(0, "up left".to_string());
        label_map.insert(1, "down right".to_string());
        let dataset = Dataset::new(datas, labels, Some(label_map));

        let mut model = SupportVectorMachine::new(2, dataset.feature_len(), 1e-1, None, 1.0);
        model.train(dataset.clone(), 1000, SVMLoss::Hinge, true);

        fn get_2d_line_w_b(model: &SupportVectorMachine) {
            let w = -model.weight[0][0] / model.weight[0][1];
            let b = -model.bias[0][0] / model.weight[0][1];
            println!("{}x + {}", w, b);
        }
        get_2d_line_w_b(&model);

        let (correct, acc) = evaluate(&dataset, &model);
        println!("correct {} acc {}", correct, acc);
    }
}
