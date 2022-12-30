<div align="center">
    <h1>
        Machine Learning in Rust
	</h1>
</div>


Learn the Rust programming language through implementing classic machine learning algorithms. This project is self-completed without relying on any third-party libraries, serving as a **bootstrap machine learning library**.

❗❗❗：Actively seeking code reviews and welcome suggestions on fixing bugs or code refactoring. Please feel free to share your ideas. Happy to accept advice!

## Basics

1. **NdArray Module**, just as the name. It has implemented ``broadcast``, ``matrix operations``, ``permute`` and etc. in arbitrary dimension. SIMD is used in matrix multiplication thanks to auto vectorizing by Rust.
2. **Dataset Module**, supporting customized loading data, re-format, ``normalize``, ``shuffle`` and ``Dataloader``. Several popular dataset pre-processing recipes are available.

## Algorithms

1. **Decision Tree**, supporting both classification and regression tasks. Info gains like ``gini`` or ``entropy`` are provided.
2. **Logistic Regression**, supporting regularization (``Lasso``, ``Ridge`` and ``L-inf``)
3. **Linear Regression**, same as logistic regression, but for regression tasks.
4. **Naive Bayes**, free to handle discrete or continuous feature values.
5. **SVM**, with linear kernel using SGD and Hinge Loss to optimize.
6. **nn Module**, containing ``linear(MLP)`` and some ``activation`` functions which could be freely stacked and optimized by gradient back propagations.
6. **KNN**, supporting both ``KdTree`` and vanilla ``BruteForceSearch``.

## Start

Let's use KNN algorithm to solve a classification task. More examples can be found in ``examples`` directory.

1. create some synthetic data for tests

   ```rust
   use std::collections::HashMap;
   
   let features = vec![
       vec![0.6, 0.7, 0.8],
       vec![0.7, 0.8, 0.9],
       vec![0.1, 0.2, 0.3],
   ];
   let labels = vec![0, 0, 1];
   // so it is a binary classifiction task, 0 is for the large label, 1 is for the small label
   let mut label_map = HashMap::new();
   label_map.insert(0, "large".to_string());
   label_map.insert(1, "small".to_string());
   ```

2. convert the data to the ``dataset``

   ```rust
   use mlinrust::dataset::Dataset;
   
   let dataset = Dataset::new(features, labels, Some(label_map));
   ```

3. split the dataset into ``train`` and ``valid`` sets and normalize them by Standard normalization

   ```rust
   let mut temp =  dataset.split_dataset(vec![2.0, 1.0], 0); // [2.0, 1.0] is the split fraction, 0 is the seed
   let (mut train_dataset, mut valid_dataset) = (temp.remove(0), temp.remove(0));
   
   use mlinrust::dataset::utils::{normalize_dataset, ScalerType};
   
   normalize_dataset(&mut train_dataset, ScalerType::Standard);
   normalize_dataset(&mut valid_dataset, ScalerType::Standard);
   ```

4. build and train our KNN model using ``KdTree``

   ```rust
   use mlinrust::model::knn::{KNNAlg, KNNModel, KNNWeighting};
   
   // KdTree is one implementation of KNN; 1 defines the k of neighbours; Weighting decides the way of ensemble prediction; train_dataset is for training KNN; Some(2) is the param of minkowski distance
   let model = KNNModel::new(KNNAlg::KdTree, 1, Some(KNNWeighting::Distance), train_dataset, Some(2));
   ```

5. evaluate the model

   ```rust
   use mlinrust::utils::evaluate;
   
   let (correct, acc) = evaluate(&valid_dataset, &model);
   println!("evaluate results\ncorrect {correct} / total {}, acc = {acc:.5}", test_dataset.len());
   ```
   

## Todo

1. K-means clustering
1. model weights serialization for saving and loading
1. Boosting/bagging
3. refactor codes, sincerely request for comments from senior developers

## Thanks

The rust community. I received many help from [rust-lang Discord](https://discord.gg/rust-lang).

## License

Under GPL-v3 license. And commercial use is strictly prohibited.