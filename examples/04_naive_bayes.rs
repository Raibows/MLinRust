use mlinrust::model::naive_bayes::NaiveBayes;
use mlinrust::dataset::{Dataset, DatasetName, FromPathDataset};
use mlinrust::utils::evaluate;

fn main() {
    // from oracle pandas unique
    let is_discrete_feature = vec![false,true,false,true,true,true,false,true,false,true,true,false,false,false,true,true,true,true,true,true];

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

    // note that the NaiveBayes classifier needs to know which feature is discrete and which feature is continuous
    let mut model = NaiveBayes::new(train_dataset.class_num(), is_discrete_feature, Some(1.0));
    model.train(&train_dataset);

    let (correct, acc) = evaluate(&train_dataset, &model);
    println!("train set correct {correct} / {}, acc = {acc:.3}", train_dataset.len());

    let (correct, acc) = evaluate(&test_dataset, &model);
    println!("test set correct {correct} / {}, acc = {acc:.3}", test_dataset.len());
    assert!(acc > 0.75);
}