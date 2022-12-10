use std::collections::HashMap;
use mlinrust::model::svm::{SupportVectorMachine, SVMLoss};
use mlinrust::dataset::{Dataset};
use mlinrust::utils::evaluate;

fn main() -> std::io::Result<()> {
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

    let (correct, acc) = evaluate(&dataset, &model);
    println!("correct {}/{} acc {}", correct, dataset.len(), acc);

    Ok(())
}