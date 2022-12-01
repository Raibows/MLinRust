use super::{Model, utils::{NormType, Penalty, gradient_clip, calculate_penalty_grad}};
use crate::{ndarray::{NdArray, utils::{softmax, sum_ndarray, argmax}}, model::utils::calculate_penalty_value};


pub struct LogisticRegression {
    weight: NdArray,
    bias: NdArray,
    feature_size: usize,
    grad_w: NdArray,
    grad_b: NdArray,
    penalty: Option<Penalty>,
}

impl LogisticRegression {
    pub fn new<F>(feature_size: usize, class_num: usize, penalty: Option<Penalty>, weight_init_fn: F) -> Self
    where F: Fn(Vec<Vec<f32>>) -> Vec<Vec<f32>>
    {
        let weight = vec![vec![0.0f32; feature_size]; class_num];
        let bias = NdArray::new(vec![class_num]);
        let weight = NdArray::new(weight_init_fn(weight));
        Self { weight: weight, bias: bias, feature_size: feature_size, grad_w: NdArray::new(vec![class_num, feature_size]), grad_b: NdArray::new(vec![class_num]), penalty: penalty}
    }

    pub fn forward(&self, input: &NdArray) -> NdArray {
        assert!(input.dim() == 2);
        assert!(input.shape[1] == self.feature_size);
        // weight: [class_num, feature_size]
        // input: [bsz, feature_size]
        // bias: [class_num]
        // return: [bsz, class_num] logits
        input * self.weight.permute(vec![1, 0]) + &self.bias
    }


    fn calculate_gradient(&mut self, feature: &NdArray, label: &Vec<usize>, mut predicts: NdArray) {
        label.iter().enumerate().for_each(|(i, l)| {
            predicts[i][*l] -= 1.0;
        });

        let mut temp_grad_w = predicts.permute(vec![1, 0]) * feature;
        if self.penalty.is_some() {
            temp_grad_w = &temp_grad_w + calculate_penalty_grad(&temp_grad_w, self.penalty.unwrap());
        }

        temp_grad_w = temp_grad_w / feature.shape[0] as f32;
        self.grad_w = &self.grad_w + temp_grad_w;
        
        let temp_grad_b = sum_ndarray(&predicts, 0) / feature.shape[0] as f32;
        self.grad_b = &self.grad_b + temp_grad_b;

    }

    pub fn one_step(&mut self, feature: &NdArray, label: &Vec<usize>, required_grad: Option<bool>) -> Vec<f32> {
        assert!(feature.dim() == 2); // feature: [bsz, feature_size]
        let required_grad = required_grad.unwrap_or(true);
        
        // after softmax logits
        let mut predicts = self.forward(feature); //
        softmax(&mut predicts, -1);
        // println!("softmax {}", predicts);

        // loss
        let penalty_v = if self.penalty.is_some() {
            calculate_penalty_value(&self.weight, self.penalty.unwrap())
        } else {
            0.0
        };
        let entropy_loss: Vec<f32> = label.iter().enumerate().map(|(i, l)| - f32::max(predicts[i][*l], 1e-9).ln() + penalty_v).collect();
        

        // calculate gradient
        if required_grad {
            self.calculate_gradient(feature, label, predicts);
        }
        // println!("{:?}", entropy_loss);
        entropy_loss
    }

    pub fn backward(&mut self, lr: f32, gradient_clip_by_norm: Option<NormType>) {
        if gradient_clip_by_norm.is_some() {
            gradient_clip(&mut self.grad_w, gradient_clip_by_norm.as_ref().unwrap());
            gradient_clip(&mut self.grad_b, gradient_clip_by_norm.as_ref().unwrap());
        }

        self.weight = &self.weight - &self.grad_w * lr;
        self.bias = &self.bias - &self.grad_b * lr;
        self.grad_w.clear();
        self.grad_b.clear();
    }

}

impl Model<usize> for LogisticRegression {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let input = vec![feature.clone()];
        let logits = self.forward(&NdArray::new(input));
        let (_, predictions) = argmax(&logits, -1).destroy();
        let predictions: Vec<usize> = predictions.into_iter().map(|i| i as usize).collect();
        predictions[0]
    }
}


#[cfg(test)]
mod test {
    use std::io::{stdout, Write};

    use crate::{
        model::{Model, utils::Penalty, utils::NormType}, 
        dataset::{Dataset, FromPathDataset, DatasetName, dataloader::Dataloader}, utils::evaluate, ndarray::utils::argmax};
    use super::{NdArray, LogisticRegression};

    #[test]
    fn test_with_synthetic_data() {
        let datas = vec![
            vec![0.0, 1.5, 2.3, 4.5, 5.8],
            vec![-6.7, 5.1, 2.3, 4.5, 1.2],
        ];
        let lr = 1e-1;
        let batch_of_feature = NdArray::new(datas.clone());
        let label = vec![0, 1];
        let mut model = LogisticRegression::new(batch_of_feature.shape[1], 2, None,  |i| i);
        let loss = model.one_step(&batch_of_feature, &label, None);
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
        model.backward(lr, None);

        let loss = model.one_step(&batch_of_feature, &label, Some(false));
        model.backward(lr, None);
        println!("loss {:?} {} {}", loss, model.predict(&datas[0]) == label[0], model.predict(&datas[1]) == label[1]);
    }

    #[test]
    fn test_with_mobile_phone_price_dataset() -> std::io::Result<()> {
        let path = ".data/MobilePhonePricePredict/train.csv";
        let mut dataset: Dataset<usize> = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
        dataset.shuffle(0);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));

        let mut model = LogisticRegression::new(train_dataset.feature_len(), train_dataset.class_num(), Some(Penalty::RidgeL2(1e-1)), |i| i);

        let mut train_dataloader = Dataloader::new(train_dataset, 8, true);

        const EPOCH: usize = 3000;
        let mut best_acc = vec![];
        for ep in 0..EPOCH {
            let mut losses = vec![];
            let lr = match ep {
                i if i < 200 => 1e-3,
                _ => 2e-3,
            };
            for (feature, label) in &mut train_dataloader {
                let loss: f32 = model.one_step(&feature, &label, None).iter().sum::<f32>() / feature.shape[0] as f32;
                model.backward(lr, Some(NormType::L2(1.0)));
                losses.push(loss);
            }
            let (_, acc) = evaluate(&test_dataset, &model);
            best_acc.push(acc);
            let width = ">".repeat(ep * 100 / EPOCH);
            print!("\r{:-<100}\t{:.3}\t{:.3}", width, losses.iter().sum::<f32>() / losses.len() as f32, acc);
            stdout().flush()?;
        }
        let acc = best_acc.iter().fold(0.0, |s, i| f32::max(s, *i));
        let best_ep = argmax(&NdArray::new(best_acc), 0);
        println!("\nbest acc = {} ep {}", acc, best_ep[0][0]);
        assert!(acc > 0.75); // gradient clip greatly helps it

        Ok(())

    }
}
