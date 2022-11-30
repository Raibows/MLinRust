<div align="center">
    <h2>
        Machine Learning in Rust
	</h2>
</div>

This is an exercise project for beginners of Rust programming language. This project implements several classical machine learning algorithms in Rust from scratch, with almost no dependencies on any third-party libraries, which indicates that it is a **bootstrap machine learning library**.

### Basics

1. NdArray Module, just as the name. It has implemented ``matrix operations``, ``broadcast``, ``permute`` and etc. in arbitrary dimension. **Highlight**:  SIMD is used in matrix multiplication thanks to auto vectorizing by Rust.
2. Dataset Module, supporting customized loading data and re-format and ``Dataloader``. Several popular dataset pre-processing recipes are available.

### Algorithms

1. Decision Tree, supporting both classification and regression tasks. Different info gains like ``gini`` or ``entropy`` are provided.
2. Logistic Regression, supports regularization (Lasso, Ridge and L-inf)
3. Linear Regression, same as logistic regression, but for regression tasks.
4. Naive Bayes, free to handle discrete or continuous feature values.

### Todo

1. a simple random number generator
2. SVM
3. Boosting/bagging
4. docs for codes and guides for users (hackertobefest)
5. refactor codes (hackertobefest), sincerely request for comments from senior developers

### Dependency

1. rand, we need random number to implement like ``shuffle`` or ``split dataset`` functions. It is in the todo list of removing dependencies.

### Thanks

1. The rust community. I received many help from [rust-lang Discord](https://discord.gg/rust-lang).

### License

Under GPL-v3 license.