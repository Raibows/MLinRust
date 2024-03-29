#![allow(dead_code)]

use std::{collections::{HashMap}};
use crate::dataset::{Dataset, TaskLabelType};
use super::Model;

/// calculate the info gains by
/// * Gini
/// * Entropy
/// * Variation
#[derive(Debug)]
pub enum InfoGains {
    Gini,
    Entropy,
    Variation,
}

#[derive(Debug)]
pub struct Node<T> {
    value: Option<T>,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
    feature_idx: Option<usize>,
    threshold: Option<f32>,
    info_gain: Option<f32>,
}

#[derive(Debug)]
pub struct DecisionTree<T: TaskLabelType> {
    root: Option<Box<Node<T>>>,
    min_sample_split: usize,
    max_depth: usize,
    info_gain_type: InfoGains,
}

pub trait TaskConditionedReturn<T: TaskLabelType> {
    /// return the ensemble results
    fn get_leaf_value(&self) -> T;

    fn calculate_information(&self, info_gain_type: &InfoGains) -> f32;
}

impl TaskConditionedReturn<usize> for Dataset<usize> {
    fn get_leaf_value(&self) -> usize {
        let mut rations_by_label: HashMap<usize, f32> = HashMap::new();
        for idx in 0..self.len() {
            let (_, label) = self.get(idx);
            *rations_by_label.entry(*label).or_insert(0.0) += 1.0;
        }
        let item = rations_by_label.iter().max_by(|x, y| (*x.1).total_cmp(y.1)).unwrap();

        *item.0
    }

    fn calculate_information(&self, info_gain_type: &InfoGains) -> f32 {
        let mut rations_by_label: HashMap<usize, f32> = HashMap::new();
        for idx in 0..self.len() {
            let (_, label) = self.get(idx);
            *rations_by_label.entry(*label).or_insert(0.0) += 1.0;
        }
        rations_by_label.iter_mut().for_each(|n| {
            *n.1 /= self.len() as f32;
        });
        
        let info = match info_gain_type {
            InfoGains::Gini => rations_by_label.values().fold(1.0, |gini, x| gini - x * x),
            InfoGains::Entropy => rations_by_label.values().fold(0.0, |entropy, x| entropy - x * x.log2()),
            _ => {assert!(false, "you should use gini or entropy!"); 0.0},
        };
        if cfg!(test) {
            println!("{rations_by_label:?} {info}");
        }
        info
    }
}

impl TaskConditionedReturn<f32> for Dataset<f32> {
    fn get_leaf_value(&self) -> f32 {
        self.labels.iter().sum::<f32>() / self.len() as f32
    }

    fn calculate_information(&self, info_gain_type: &InfoGains) -> f32 {
        match info_gain_type {
            InfoGains::Variation => {
                let mean = self.labels.iter().sum::<f32>() / self.len() as f32;
                let var = self.labels.iter().fold(0.0, |acc, item| acc + (item - mean) * (item - mean)) / self.len() as f32;
                var
            },
            _ => panic!("you should use Variation for InfoGains but got {:?}", info_gain_type)
        }    
    }
}


impl<T: TaskLabelType + Copy> Model<T> for DecisionTree<T> {
    fn predict(&self, feature: &Vec<f32>) -> T {
        let mut node = self.root.as_ref().unwrap().clone();
        // let mut parent;
        while node.value.is_none() {
            if feature[node.feature_idx.unwrap()] < node.threshold.unwrap() {
                node = &node.left.as_ref().unwrap().clone();
            } else {
                node = &node.right.as_ref().unwrap().clone();
            }
        }
        node.value.unwrap()
    }
}


impl<T: TaskLabelType + Copy + std::fmt::Display> DecisionTree<T> {
    /// define a decision tree (not built yet), f32 for regression task, usize for classification task
    /// * min_sample_split: it will not try to split the tree if it reaches the min_sample_split
    /// * max_depth: max depth of the tree
    /// * info_gain: calculate the info gains by InfoGains(Gini, Entropy, Variation)
    pub fn new(min_sample_split: usize, max_depth: usize, info_gain: InfoGains) -> Self {
        DecisionTree { root: None, min_sample_split: min_sample_split, max_depth: max_depth, info_gain_type: info_gain}
    }

    /// build the tree by the train dataset
    pub fn train(&mut self, dataset: Dataset<T>) 
    where Dataset<T>: TaskConditionedReturn<T> 
    {
        self.root = Some(Box::new(self.build_trees(dataset, 0)));
    }

    fn build_trees(&mut self, dataset: Dataset<T>, current_depth: usize) -> Node<T> where Dataset<T>: TaskConditionedReturn<T>
    {
        let sample_num = dataset.len();
        // let feature_num = dataset.feature_len();
        if sample_num > self.min_sample_split && current_depth < self.max_depth {
            let (info_gain, feature_idx, threshold, left_dataset, right_dataset) = self.get_best_split(&dataset);
            if info_gain > 0.0 {
                let left_tree = self.build_trees(left_dataset.unwrap(), current_depth + 1);
                let right_tree = self.build_trees(right_dataset.unwrap(), current_depth + 1);
                return Node {feature_idx: Some(feature_idx), info_gain: Some(info_gain), left: Some(Box::new(left_tree)), right: Some(Box::new(right_tree)), threshold: Some(threshold), value: None};
            }
        }

        let leaf_value = dataset.get_leaf_value();

        Node {feature_idx: None, info_gain: None, left: None, right: None, threshold: None, value: Some(leaf_value)}
    }

    /// traverse all possible feature values to find the best split (the max info gain)
    fn get_best_split(&self, dataset: &Dataset<T>) -> (f32, usize, f32, Option<Dataset<T>>, Option<Dataset<T>>)
    where Dataset<T>: TaskConditionedReturn<T>
    {
        // info_gain, feature_idx, threshold, left_dataset, right_dataset
        let mut best_splits = (f32::MIN, 0, f32::MIN, None, None);
        let parent_info = dataset.calculate_information(&self.info_gain_type);
        if parent_info <= 0.0 {
            return best_splits;
        }
        for fi in 0..dataset.feature_len() {
            let feature_values = dataset.get_feature_by_idx(fi);
            for threshold in dataset.get_unique_feature_values(fi) {
                let (left_dataset, right_dataset) = self.split_dataset_by(dataset, &feature_values, *threshold);

                if left_dataset.len() > 0 && right_dataset.len() > 0 {
                    let current_info_gains = parent_info - 
                    (left_dataset.len() as f32 / dataset.len() as f32) * left_dataset.calculate_information(&self.info_gain_type) - 
                    (right_dataset.len() as f32 / dataset.len() as f32) * right_dataset.calculate_information(&self.info_gain_type);
                    if cfg!(test) {
                        println!("debug current info gain {current_info_gains}");
                    }
                    if current_info_gains > best_splits.0 {
                        best_splits = (current_info_gains, fi, *threshold, Some(left_dataset), Some(right_dataset));
                    }
                }
            }
        }
        best_splits
    }

    /// split the dataset into two subsets by the given (feature_idx, feature_value)
    fn split_dataset_by(&self, dataset: &Dataset<T>, feature_values: &Vec<&f32>, threshold: f32) -> (Dataset<T>, Dataset<T>) {
        let mut left_datas = vec![];
        let mut right_datas = vec![];
        for (i, fv) in feature_values.iter().enumerate() {
            if **fv < threshold {
                left_datas.push(dataset.get(i));
            } else {
                right_datas.push(dataset.get(i));
            }
        }

        (Dataset::from(left_datas), Dataset::from(right_datas))
    }

    
    fn recursive_print(&self, node: &Option<Box<Node<T>>>, depth: usize) {
        if node.is_some() {
            let width = " ".repeat(depth * 7) + "-------";
            let t = node.as_ref().unwrap();
            if t.value.is_some() {
                println!("{depth} {width} VALUE: {}", t.value.unwrap());
            } else {
                println!("{depth} {width} LEFT : F{:0>3} < {}", t.feature_idx.unwrap(), t.threshold.unwrap());
                self.recursive_print(&node.as_ref().unwrap().as_ref().left, depth + 1);
                println!("{depth} {width} RIGHT: F{:0>3} ≥ {}", t.feature_idx.unwrap(), t.threshold.unwrap());
                self.recursive_print(&node.as_ref().unwrap().as_ref().right, depth + 1);
            }
        }
    }

    /// print the structure of the built tree
    pub fn print_self(&self) {
        self.recursive_print(&self.root, 0);
    }
}



#[cfg(test)]
mod test {
    use crate::dataset::{DatasetName, FromPathDataset};
    use crate::utils::{evaluate, evaluate_regression};
    use super::{Dataset};
    use super::{DecisionTree, InfoGains};


    #[test]
    fn test_synthetic_samples() {
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.0, 3.1, 4.0],
            vec![1.1, 2.0, 3.0, 4.0],
        ];
    
        let y = vec![0, 1, 2];
        let temp_dataset = Dataset::new(x, y, None);
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini);
        dct.train(temp_dataset);
        dct.print_self();
    }

    #[test]
    fn test_iris_dataset() {
        let path = ".data/IRIS.csv";
        let dataset = Dataset::<usize>::from_name(path, DatasetName::IrisDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini);
        dct.train(train_dataset);

        dct.print_self();

        let (correct, acc) = evaluate(&test_dataset, &dct);
        println!("correct {correct} / test {}, acc = {acc}", test_dataset.len());

        assert!(acc > 0.7);
    }

    #[test]
    fn test_mobile_phone_price_dataset() {
        let path = ".data/MobilePhonePricePredict/train.csv";
        let dataset = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini);
        dct.train(train_dataset);

        dct.print_self();

        let (correct, acc) = evaluate(&test_dataset, &dct);
        println!("correct {correct} / test {}, acc = {acc}", test_dataset.len());

        assert!(acc > 0.7)
    }

    #[test]
    fn test_car_price_regression_dataset() {
        let path = ".data/TianchiCarPriceRegression/train_5w.csv";
        let dataset = Dataset::<f32>::from_name(path, DatasetName::CarPriceRegressionDataset, None);
        let mut res = dataset.split_dataset(vec![0.8, 0.2], 0);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());

        let mut dct = DecisionTree::new(100, 3, InfoGains::Variation);
        dct.train(train_dataset);
        dct.print_self();

        let abs_error = evaluate_regression(&test_dataset, &dct);
        println!("mean absolute error {abs_error:.5}");
    }
}