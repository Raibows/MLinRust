use mlinrust::dataset::{Dataset, FromPathDataset, DatasetName};
use mlinrust::model::decision_tree::{DecisionTree, InfoGains};
use mlinrust::utils::evaluate;

fn main() {
    println!("Hello, world!");

    let path = ".data/MobilePhonePricePredict/train.csv";

    let dataset = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
    let mut temp =  dataset.split_dataset(vec![0.8, 0.2], 0);
    let (train_dataset, test_dataset) = (temp.remove(0), temp.remove(0));

    println!("train {} : test {}", train_dataset.len(), test_dataset.len());

    let mut model = DecisionTree::<usize>::new(1, 7, InfoGains::Entropy);
    model.train(train_dataset);
    println!("model training done!");
    model.print_self(&model.root, 0);

    let (correct, acc) = evaluate(&test_dataset, &model);
    println!("evaluate results\ncorrect {} / total {}, acc = {:.5}", correct, test_dataset.len(), acc);
}
