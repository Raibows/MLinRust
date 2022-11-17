
pub mod decision_trees;
pub mod utils;


use std::{collections::HashMap};
use crate::utils::FromPathDataset;

// struct Arr<T> {
//     value: T
// }




fn main() {
    
    println!("Hello, world!");
    let path = ".data/IRIS.csv";
    let dataset = utils::Dataset::<usize>::from_name(utils::DatasetName::IrisDataset(path));
    let mut dct: decision_trees::DecisionTree<usize> = decision_trees::DecisionTree::new(3, 10, decision_trees::InfoGains::Entropy, utils::Task::Classification);
    dct.root = Some(Box::new(dct.build_trees(dataset, 0)));



    let dataset = utils::Dataset::<usize>::from_name(utils::DatasetName::IrisDataset(path));
    println!("{:?}", dct.predict(dataset.get(0)));
    



    // if dataset.is_err() {
    //     println!("error opening {}", path);
    // } else {
    //     let dataset = dataset.unwrap();
    //     println!("{:?}", dataset.labels);
        
    // }



    let v = vec!["12".to_string(), "4".to_string(), "5".to_string()];
    for item in v {
        println!("{}", item);
    }
    // for item in v {
    //     println!("{}", item);

    // }
    let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    let mut fp = HashMap::new();
    fp.insert(1usize, 1.0f32);
    fp.insert(2usize, 2.0f32);
    fp.insert(19usize, 2.8f32);
    println!("{}", fp.values().fold(1.0, |acc, x| acc - *x));
    println!("{}", v.iter().fold(1.0, |acc, x| acc - x));


    let argmax = fp.iter().max_by(|(_, a), (_, b)| a.total_cmp(b));
    println!("{:?}", argmax);

}
