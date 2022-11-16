
pub mod decision_trees;
pub mod utils;


use std::collections::HashMap;

// struct Arr<T> {
//     value: T
// }




fn main() {
    
    println!("Hello, world!");
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
    println!("{}", fp.values().fold(1.0, |acc, x| acc - *x));
    println!("{}", v.iter().fold(1.0, |acc, x| acc - x));


    let argmax = v.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b));
    println!("{:?}", argmax);

}
