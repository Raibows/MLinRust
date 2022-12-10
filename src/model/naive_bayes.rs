use std::{vec, collections::{HashMap, HashSet}};

use crate::{dataset::Dataset};

use super::Model;

#[derive(Clone, Debug)]
enum ProbFeatureType {
    Discrete(HashMap<i32, f32>),
    Continuous((Vec<f32>, f32, f32))
}


pub struct NaiveBayes {
    is_discrete_feature: Vec<bool>,
    prior_prob: Vec<f32>,
    // for discrete feature x_i
    // p(x_i | c) = D_c_x_i / D
    // for continuous feature x_i
    // p(x_i | c) = N_\mu_\sigma()
    // (c, i(feature idx), v)
    class_condition_prob: Vec<Vec<ProbFeatureType>>,
    unknown_feature_prob: Vec<Vec<f32>>,
    laplace_alpha: f32
}

impl NaiveBayes {
    pub fn new(class_num: usize, is_discrete_feature: Vec<bool>, laplace_alpha: Option<f32>) -> Self {
        let class_condition_prob = is_discrete_feature.iter().map(|i| {
            if *i {
                ProbFeatureType::Discrete(HashMap::new())
            } else {
                ProbFeatureType::Continuous((vec![], 0.0, 0.0))
            }
        }).collect::<Vec<ProbFeatureType>>();
        let laplace_alpha = laplace_alpha.unwrap_or(1.0);

        Self { is_discrete_feature: is_discrete_feature, prior_prob: vec![laplace_alpha; class_num], class_condition_prob:  vec![class_condition_prob; class_num], laplace_alpha: laplace_alpha, unknown_feature_prob: vec![vec![]; class_num]}
    }

    fn gaussian_dist_pdf(&self, x: f32, mu: f32, sigma: f32) -> f32 {
        1.0 / f32::max(1e-6, (2.0 * std::f32::consts::PI).sqrt() * sigma) *
        std::f32::consts::E.powf(
            - (x - mu).powf(2.0) / f32::max(1e-6, 2.0 * sigma.powf(2.0))
        )
    }

    pub fn train(&mut self, dataset: &Dataset<usize>) {
        assert!(dataset.feature_len() == self.is_discrete_feature.len());
        assert!(dataset.class_num() == self.prior_prob.len());
        
        let discrete_num = self.is_discrete_feature.iter().fold(0, |s, i| s + if *i {1} else {0});
        let mut discrete_feature_n = vec![HashSet::<i32>::new(); discrete_num];

        dataset.into_iter().for_each(|(x, y)| {
            // prior prob
            self.prior_prob[y] += 1.0;
            // likelihood p(x|y)
            let mut idx = 0;
            x.iter().zip(self.class_condition_prob[y].iter_mut()).for_each(|(xi, d)| {
                match d {
                    ProbFeatureType::Discrete(item) => {
                        *item.entry(*xi as i32).or_insert(0.0) += 1.0;
                        discrete_feature_n[idx].insert(*xi as i32); // statistic the max unique value of a discrete feature
                        idx += 1;
                    },
                    ProbFeatureType::Continuous((item, _, _)) => {
                        item.push(*xi);
                    },
                }
            });
        });

        // calculate prior prob
        let total: f32 = self.prior_prob.iter().sum();
        self.prior_prob.iter_mut().for_each(|i| *i /= total);

        // calculate likelihood
        self.class_condition_prob.iter_mut()
        .zip(self.unknown_feature_prob.iter_mut())
        .for_each(|(values, unknown)| {
            let mut idx = 0;
            values.iter_mut().for_each(|item| {
                match item {
                    ProbFeatureType::Discrete(cnt) => {
                        let sum: f32 = cnt.values().sum::<f32>() + self.laplace_alpha * (discrete_feature_n[idx].len() + 1) as f32; // 1 is for the unknown
                        cnt.values_mut().for_each(|i| *i = (*i + self.laplace_alpha) / sum); // laplace smoothing
                        unknown.push(self.laplace_alpha / sum);
                        idx += 1;
                    },
                    ProbFeatureType::Continuous((data, mu, sigma)) => {
                        *mu = data.iter().sum::<f32>() / data.len() as f32;
                        *sigma = data.iter().fold(0.0, |s, i| s + (*mu - *i) * (*mu - *i)) / data.len() as f32;
                        *sigma = sigma.sqrt();
                    },
                }
            });
        });
    }
}

impl Model<usize> for NaiveBayes {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        assert!(feature.len() == self.is_discrete_feature.len());
        let posterior_probs: Vec<f32> = self.class_condition_prob.iter().zip(self.prior_prob.iter()).zip(self.unknown_feature_prob.iter()).map(|((likelihood, prior), unknown)| {
            // println!("unknown feature prob {:?}", unknown);
            let mut unk = unknown.iter();
            likelihood.iter().zip(feature.iter()).fold(*prior, |s, (record, xi)| {
                s * match record {
                    ProbFeatureType::Discrete(freqs) => {
                        *freqs.get(&(*xi as i32)).unwrap_or(unk.next().unwrap())
                    },
                    ProbFeatureType::Continuous((_, mu, sigma)) => {
                        self.gaussian_dist_pdf(*xi, *mu, *sigma)    
                    },
                }
            }) 
        }).collect();

        // argmax
        // println!("class condition prob {:?}", self.class_condition_prob);
        // println!("posterior {:?}", posterior_probs);
        posterior_probs.iter().enumerate().fold((0, f32::MIN), |s, i| {
            if *i.1 > s.1 {
                (i.0, *i.1)
            } else {
                s
            }
        }).0
    }
}


#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::NaiveBayes;
    use super::{Dataset};
    use crate::dataset::{DatasetName, FromPathDataset};
    use crate::model::Model;
    use crate::utils::evaluate;

    #[test]
    fn test_gaussian_pdf() {
        let model = NaiveBayes::new(2, vec![true], None);
        println!("{}", model.gaussian_dist_pdf(0.697, 0.574, 0.129)); // 1.959
        println!("{}", model.gaussian_dist_pdf(0.697, 0.496, 0.195)); // 1.203
    }

    #[test]
    fn test_unknown_feature() {
        let is_discrete_feature = vec![true, false];
        let data = vec![
            vec![0.0, 0.5],
            vec![1.0, 0.6],
        ];
        let label = vec![0, 1];
        let mut label_map = HashMap::new();
        label_map.insert(0, "0".to_string());
        label_map.insert(1, "1".to_string());
        let dataset = Dataset::new(data, label, Some(label_map));
        let mut model = NaiveBayes::new(dataset.class_num(), is_discrete_feature, Some(1.0));
        model.train(&dataset);

        let test_sample = vec![2.0, 0.589];
        let res = model.predict(&test_sample);
        println!("note that the sigma of the second feature is zero! so the prob for every class is equal to ZERO!");
        assert!(res == 0);
    }

    #[test]
    fn test_with_iris_dataset() {
        let is_discrete_feature = vec![false,true,false,true,true,true,false,true,false,true,true,false,false,false,true,true,true,true,true,true]; // from oracle pandas unique
        let path = ".data/MobilePhonePricePredict/train.csv";

        let dataset = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
        let mut temp =  dataset.split_dataset(vec![0.8, 0.2], 0);
        let (train_dataset, test_dataset) = (temp.remove(0), temp.remove(0));

        let discrete_feature_is: Vec<bool> = (0..train_dataset.feature_len()).map(|i| {
            train_dataset.get_unique_feature_values(i).len() < 25
        }).collect();

        assert!(is_discrete_feature == discrete_feature_is);

        println!("train {} : test {}", train_dataset.len(), test_dataset.len());
        assert_eq!(is_discrete_feature.len(), train_dataset.feature_len());

        let mut model = NaiveBayes::new(train_dataset.class_num(), is_discrete_feature, Some(1.0));
        model.train(&train_dataset);

        let (correct, acc) = evaluate(&train_dataset, &model);
        println!("train set correct {} / {}, acc = {}", correct, train_dataset.len(), acc);

        let (correct, acc) = evaluate(&test_dataset, &model);
        println!("test set correct {} / {}, acc = {}", correct, test_dataset.len(), acc);
        assert!(acc > 0.75);
    }
}