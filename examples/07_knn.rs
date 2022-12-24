use mlinrust::dataset::{Dataset, FromPathDataset, DatasetName};
use mlinrust::model::knn::{KNNAlg, KNNModel, KNNWeighting};
use mlinrust::utils::evaluate;

fn main() {

    let path = ".data/MobilePhonePricePredict/train.csv";

    let dataset = Dataset::<usize>::from_name(path, DatasetName::MobilePhonePricePredictDataset, None);
    let mut temp =  dataset.split_dataset(vec![0.8, 0.2], 0);
    let (train_dataset, test_dataset) = (temp.remove(0), temp.remove(0));

    println!("train {} : test {}", train_dataset.len(), test_dataset.len());

    let model = KNNModel::new(KNNAlg::KdTree, 16, Some(KNNWeighting::Distance), train_dataset, Some(2));
    println!("model training done!");

    let (correct, acc) = evaluate(&test_dataset, &model);
    println!("evaluate results\ncorrect {correct} / total {}, acc = {acc:.5}", test_dataset.len());
}