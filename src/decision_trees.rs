use std::collections::HashMap;

use crate::utils;

struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
    feature_name: String,
    threshold: f32,
    info_gain: f32,
}

struct DecisionTree<T> {
    root: Option<Box<Node<T>>>,
    min_sample_split: usize,
    max_depth: usize,
    info_gain: String,
}

impl<T> DecisionTree<T> {
    pub fn new(min_sample_split: usize, max_depth: usize, info_gain: &str) -> DecisionTree<T> {
        DecisionTree::<T> { root: None, min_sample_split: min_sample_split, max_depth: max_depth, info_gain: info_gain.to_string() }
    }

    pub fn build_trees(&mut self, dataset: utils::Dataset, current_depth: usize) {
        let sample_num = dataset.len();
        let feature_num = if let Some(item) = features.get(0) {
            item.len()    
        } else {
            0
        };
        if sample_num > self.min_sample_split && current_depth < self.max_depth {
            let best_split = self.get_best_split(features, labels);
        }
    }

    fn get_best_split(&self, features: Vec<HashMap<String, f32>>, labels: Vec<HashMap<String, f32>>) -> () {
        todo!()
    }
}