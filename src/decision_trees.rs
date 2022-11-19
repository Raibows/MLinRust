#![allow(dead_code)]

use std::{collections::{HashMap}};
use crate::dataset::{Dataset, Task, TaskLabelType};

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
pub struct DecisionTree<T> {
    pub root: Option<Box<Node<T>>>,
    min_sample_split: usize,
    max_depth: usize,
    info_gain_type: InfoGains,
    task: Task
}

pub trait TaskConditionedReturn<T: TaskLabelType> {
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
            println!("{:?} {}", rations_by_label, info);
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
            _ => {assert!(false, "you should use variation!"); 0.0},
        }    
    }
}



impl<T: TaskLabelType + Copy + std::cmp::PartialEq + std::fmt::Display> DecisionTree<T> {
    pub fn new(min_sample_split: usize, max_depth: usize, info_gain:InfoGains, task: Task) -> Self {
        DecisionTree { root: None, min_sample_split: min_sample_split, max_depth: max_depth, info_gain_type: info_gain, task: task}
    }

    pub fn build_trees(&mut self, dataset: Dataset<T>, current_depth: usize) -> Node<T> where Dataset<T>: TaskConditionedReturn<T>
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

        let leaf_value = self.get_leaf_value(&dataset);

        Node {feature_idx: None, info_gain: None, left: None, right: None, threshold: None, value: Some(leaf_value)}
    }

    fn get_best_split(&self, dataset: &Dataset<T>) -> (f32, usize, f32, Option<Dataset<T>>, Option<Dataset<T>>)
    where Dataset<T>: TaskConditionedReturn<T>
    {
        // info_gain, feature_idx, threshold, left_dataset, right_dataset
        let mut best_splits = (f32::MIN, 0, f32::MIN, None, None);
        for fi in 0..dataset.feature_len() {
            let feature_values = dataset.get_feature_by_idx(fi);
            for threshold in dataset.get_unique_feature_values(fi) {
                let (left_dataset, right_dataset) = self.split_dataset_by(dataset, &feature_values, *threshold);

                if left_dataset.len() > 0 && right_dataset.len() > 0 {
                    let current_info_gains = dataset.calculate_information(&self.info_gain_type) - 
                    (left_dataset.len() as f32 / dataset.len() as f32) * left_dataset.calculate_information(&self.info_gain_type) - 
                    (right_dataset.len() as f32 / dataset.len() as f32) * right_dataset.calculate_information(&self.info_gain_type);
                    if cfg!(test) {
                        println!("debug current info gain {}", current_info_gains);
                    }
                    if current_info_gains > best_splits.0 {
                        best_splits = (current_info_gains, fi, *threshold, Some(left_dataset), Some(right_dataset));
                    }
                }
            }
        }
        best_splits
    }

    fn get_leaf_value<E: TaskConditionedReturn<T>>(&self, dataset: &E) -> T {
        dataset.get_leaf_value()
    }

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

    fn judge(&self, features: &Vec<f32>, node: &Box<Node<T>>) -> T {
        if node.value.is_some() {
            node.value.unwrap()
        } else {
            if features[node.feature_idx.unwrap()] < node.threshold.unwrap() {
                self.judge(features, &node.as_ref().left.as_ref().unwrap())
            } else {
                self.judge(features, &node.as_ref().right.as_ref().unwrap())
            }
        }
    }

    // pub fn predict(&self, item: (&Vec<f32>, &T)) -> bool {
    //     self.judge(item.0, self.root.as_ref().unwrap()) == *item.1
    // }

    pub fn predict(&self, item: (&Vec<f32>, &T)) -> bool {
        let mut node = self.root.as_ref().unwrap().clone();
        // let mut parent;
        let (feature, label) = item;
        while node.value.is_none() {
            if feature[node.feature_idx.unwrap()] < node.threshold.unwrap() {
                node = &node.left.as_ref().unwrap().clone();
            } else {
                node = &node.right.as_ref().unwrap().clone();
            }
        }
        node.value.unwrap() == *label
    }

    pub fn print_self(&self, node: &Option<Box<Node<T>>>, depth: usize) {
        if node.is_some() {
            let width = " ".repeat(depth * 7) + "-------";
            let t = node.as_ref().unwrap();
            if t.value.is_some() {
                println!("{} {} VALUE: {}", depth, width, t.value.unwrap());
            } else {
                println!("{} {} LEFT : F{:0>3} < {}", depth, width, t.feature_idx.unwrap(), t.threshold.unwrap());
                self.print_self(&node.as_ref().unwrap().as_ref().left, depth + 1);
                println!("{} {} RIGHT: F{:0>3} ≥ {}", depth, width, t.feature_idx.unwrap(), t.threshold.unwrap());
                self.print_self(&node.as_ref().unwrap().as_ref().right, depth + 1);
            }
        }
    }
}



#[cfg(test)]
mod test {
    use crate::dataset::{DatasetName, FromPathDataset};
    use super::{Dataset, Task};
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
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini, Task::Classification);
        dct.root = Some(Box::new(dct.build_trees(temp_dataset, 0)));
        dct.print_self(&dct.root, 0);

        assert!(false);
    }

    #[test]
    fn test_iris_dataset() {
        let path = ".data/IRIS.csv";
        let dataset = Dataset::<usize>::from_name(path, DatasetName::IrisDataset);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini, Task::Classification);
        dct.root = Some(Box::new(dct.build_trees(train_dataset, 0)));


        dct.print_self(&dct.root, 0);


        let mut correct = 0;
        for i in 0..test_dataset.len() {
            match dct.predict(test_dataset.get(i)) {
                true => {correct += 1},
                false => {},
            }
        }
        let acc = correct as f32 / test_dataset.len() as f32;
        println!("correct {} / test {}, acc = {}", correct, test_dataset.len(), acc);

        assert!(acc > 0.7);
    }

    #[test]
    fn test_mobile_phone_price_dataset() {
        let path = ".data/MobilePhonePricePredict/train.csv";
        let dataset = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset);
        let mut res = dataset.split_dataset(vec![0.8, 0.2]);
        let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
        println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());
        let mut dct = DecisionTree::<usize>::new(1, 3, InfoGains::Gini, Task::Classification);
        dct.root = Some(Box::new(dct.build_trees(train_dataset, 0)));


        dct.print_self(&dct.root, 0);


        let mut correct = 0;
        for i in 0..test_dataset.len() {
            match dct.predict(test_dataset.get(i)) {
                true => {correct += 1},
                false => {},
            }
        }
        let acc = correct as f32 / test_dataset.len() as f32;
        println!("correct {} / test {}, acc = {}", correct, test_dataset.len(), acc);

        assert!(acc > 0.7)
    }
}