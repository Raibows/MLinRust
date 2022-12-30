use std::io::{stdout, Write};

use crate::{ndarray::{NdArray, utils::sum_ndarray}, dataset::{Dataset, dataloader::Dataloader}, utils::evaluate};
use super::{utils::{Penalty, calculate_penalty_grad, calculate_penalty_value}, Model};

/// soft constraints for SVM
/// * Hinge Loss, max(0, 1 - z)
/// * Exponential, exp(-z)
/// * Log, log (1 + exp(-z))
#[derive(Clone, Copy, Debug)]
pub enum SVMLoss {
    Hinge,
    Exponential,
    Logistic,
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
    /// Support Vector Machine for binary classification
    /// 
    /// only supports linear kernel, using SGD to optimize hinge loss
    /// 
    /// update = lambda * (regularization(w) or marginal width or so called structure loss) + 1/batch * Hinge Loss
    /// 
    /// * bsz: default will be len(dataset)
    pub fn new(class_num: usize, feature_num: usize, lr: f32,  bsz: Option<usize>, lambda: f32) -> Self {
        
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

    /// update weight and bias
    /// 
    /// * gradient calculation is by average
    /// * return: (margin/structure risk, avg_hinge_loss)
    fn gradient_backward(&mut self, loss: SVMLoss, x: &NdArray, y: &Vec<usize>) -> (f32, f32) {
        let y: Vec<f32> = y.iter().map(|i| if *i == 0 {-1.0} else {1.0}).collect();
        let mut z = self.forward(x);

        z.data_as_mut_vector().iter_mut().zip(y.iter()).for_each(|(p, y)| {
            *p = *p * *y;
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
            _ => panic!("not supporting other kernels except linear kernel"),
        };

        // lambda * regularization or marginal width
        let structural_risk = calculate_penalty_grad(&self.weight, Penalty::RidgeL2(self.lambda));
        let margin = calculate_penalty_value(&self.weight, Penalty::RidgeL2(self.lambda));

        // reduction
        let bsz = x.shape[0] as f32;
        let grad_w = structural_risk + grad_w / bsz;
        let grad_b = grad_b / bsz;
        let avg_loss = losses.iter().fold(0.0, |s, i| s + i) / bsz;
        self.weight = &self.weight - grad_w * self.lr;
        self.bias = &self.bias - grad_b * self.lr;

        (margin, avg_loss)
    }

    /// train the SVM model
    /// * verbose: whether show the training info
    pub fn train(&mut self, dataset: Dataset<usize>, epoch: usize, loss: SVMLoss, early_stop: bool, verbose: bool) -> std::io::Result<()> {
        assert!(dataset.class_num() == 2);
        let bsz = self.bsz.unwrap_or(dataset.len());
        let mut dataloader: Dataloader<usize, Dataset<usize>> = Dataloader::new(dataset, bsz, true, None);

        let mut last_eps_record: Vec<f32> = vec![];

        for ep in 0..epoch {
            let mut avg_loss = vec![];
            for (x, y) in dataloader.iter_mut() {
                let loss = self.gradient_backward(loss, &x, &y);
                avg_loss.push(loss);
            }
            let num_batch = avg_loss.len() as f32;
            let avg_loss = avg_loss.iter().fold((0.0, 0.0), |s, i| (s.0 + i.0, s.1 + i.1));
            let (structure_loss, hinge_loss) = (avg_loss.0 / num_batch, avg_loss.1 / num_batch );

            if verbose {
                let width = ">".repeat(ep * 50 / epoch);
                let (_, acc) = evaluate(&mut dataloader, self);
                print!("\r{width:-<50}epoch\t{ep}/{epoch}\ttrain_acc {acc:.3}\tmargin {:.3}\tloss {hinge_loss:.3}", 1.0/structure_loss);
                stdout().flush()?;
            }
            
            last_eps_record.push(hinge_loss);

            if ep > usize::min(10, epoch / 3) && last_eps_record.len() > 10 {
                let decrease = last_eps_record.windows(2).rev().take(10).fold(0.0, |s, i| {
                    s + (i[0] - i[1]) / i[1]
                });
                let old = self.lr;
                if early_stop && (decrease > 0.5 || self.lr < 1e-5) {
                    println!("\nearly stopping! Last 10 epoch loss {last_eps_record:?}");
                    break;
                } else if decrease > -0.05 {
                    self.lr *= 0.9; // discount lr
                } else {
                    // increase lr
                    self.lr *= 1.0 + f32::min(f32::max(-decrease - 0.2, 0.0), 0.5);
                }
                last_eps_record.clear();
                if verbose && old != self.lr {
                    println!("\nadjust lr from {old:.3e} to {:.3e}", self.lr);
                }
            }
        }
        Ok(())
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
    fn test_with_2d_points() -> std::io::Result<()> {
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
        model.train(dataset.clone(), 1000, SVMLoss::Hinge, true, true)?;

        fn get_2d_line_w_b(model: &SupportVectorMachine) {
            let w = -model.weight[0][0] / model.weight[0][1];
            let b = -model.bias[0][0] / model.weight[0][1];
            println!("{w}x + {b}");
        }
        get_2d_line_w_b(&model);

        let (correct, acc) = evaluate(&dataset, &model);
        println!("correct {correct} acc {acc}");

        Ok(())
    }
}
