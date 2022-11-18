
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
    let mut dataset = utils::Dataset::<usize>::from_name(utils::DatasetName::IrisDataset(path));
    dataset.shuffle(0);
    let mut res = dataset.split_dataset(vec![0.8, 0.2]);
    let (train_dataset, test_dataset) = (res.remove(0), res.remove(0));
    println!("split dataset train {} : test {}", train_dataset.len(), test_dataset.len());
    println!("{:?}", train_dataset.labels);


    let mut dct: decision_trees::DecisionTree<usize> = decision_trees::DecisionTree::new(1, 3, decision_trees::InfoGains::Gini, utils::Task::Classification);


    let x = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.1, 2.0, 3.1, 4.0],
        vec![1.1, 2.0, 3.0, 4.0],
    ];
    
    let y = vec![0, 1, 2];
    let temp_dataset = utils::Dataset::new(x, y, None);



    dct.root = Some(Box::new(dct.build_trees(train_dataset, 0)));

    let mut correct = 0;
    for i in 0..test_dataset.len() {
        match dct.predict(test_dataset.get(i)) {
            true => {correct += 1},
            false => {},
        }
    }
    dct.print_self(&dct.root, 1);
    println!("correct {} / test {}, acc = {}", correct, test_dataset.len(), correct as f32 / test_dataset.len() as f32);


    // println!("{:?}", dct.predict(dataset.get(0)));
    



    // if dataset.is_err() {
    //     println!("error opening {}", path);
    // } else {
    //     let dataset = dataset.unwrap();
    //     println!("{:?}", dataset.labels);
        
    // }



    // let v = vec!["12".to_string(), "4".to_string(), "5".to_string()];
    // for item in v {
    //     println!("{}", item);
    // }
    // // for item in v {
    // //     println!("{}", item);

    // // }
    // let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    // let mut fp = HashMap::new();
    // fp.insert(1usize, 1.0f32);
    // fp.insert(2usize, 2.0f32);
    // fp.insert(19usize, 2.8f32);
    // println!("{}", fp.values().fold(1.0, |acc, x| acc - *x));
    // println!("{}", v.iter().fold(1.0, |acc, x| acc - x));


    // let argmax = fp.iter().max_by(|(_, a), (_, b)| a.total_cmp(b));
    // println!("{:?}", argmax);

}
