<div align="center">
    <h1>
        Machine Learning in Rust
	</h1>
</div>


This is a self-practice project for learning the Rust programming language. This project implements several classical machine learning algorithms in Rust from scratch without relying on any third-party libraries. It serves as a **bootstrap machine learning library**.

Actively seeking code reviews and welcome suggestions on fixing bugs or code refactoring. Please feel free to share your ideas. Happy to accept advice!

## Basics

1. NdArray Module, just as the name. It has implemented ``broadcast``, ``matrix operations``, ``permute`` and etc. in **arbitrary dimension**. SIMD is used in matrix multiplication thanks to auto vectorizing by Rust.
2. Dataset Module, supporting customized loading data, re-format, ``normalize``, ``shuffle`` and ``Dataloader``. Several popular dataset pre-processing recipes are available.

## Algorithms

1. **Decision Tree**, supporting both classification and regression tasks. Info gains like ``gini`` or ``entropy`` are provided.
2. **Logistic Regression**, supporting regularization (``Lasso``, ``Ridge`` and ``L-inf``)
3. **Linear Regression**, same as logistic regression, but for regression tasks.
4. **Naive Bayes**, free to handle discrete or continuous feature values.
5. **SVM**, with linear kernel using SGD and Hinge Loss to optimize.
6. **nn Module**, containing ``linear(MLP)`` and some ``activation`` functions which could be freely stacked and optimized by gradient back propagations.
6. **KNN**, supporting both ``KdTree`` and vanilla ``BruteForceSearch``.

## Todo

1. K-means clustering
1. Boosting/bagging
2. docs for codes and guides for users
3. refactor codes, sincerely request for comments from senior developers

## Thanks

The rust community. I received many help from [rust-lang Discord](https://discord.gg/rust-lang).

## License

Under GPL-v3 license.

## Trivial Details...

### Matrix Multiplication

The initial implementation of matrix multiplication is in a simple triple-loop way, even not considering cache-friendly array indexing. The code looks like

```rust
for j in 0..n { // every column of B
    for i in 0..m { // every row of A
        for k in 0..k {
            C[i][j] += A[i][k] * B[k][j] // indexing B by every row makes cache miss a lot
        }
    }
}
```

I modified the indexing order a little when I was going to optimize its outrageously slow speed...

```rust
for i in 0..m {
    for k in 0..k {
    	for j in 0..n {
            C[i][j] += A[i][k] * B[k][j] // no cache miss now
        }
    }
}
```

The reordering makes it about 2x faster (test size is 512^3).

Then, I found the rust compiler could usually apply auto-vectorizing optimization to iterations easily by SIMD, so I converted the indexing to iterating and got about **40 x faster**.

### NdArray Permute

The permute operation involves the reordering of the actual data, so at least you have to traverse all elements in the ndarray, which indicates a $O(N)$ time complexity. Initially I implement it by a DFS recursive calling. It is very shocking that I find the permutation even consumes more resources than multiplication operation (while this is another story, should not blame for permute). A simple optimization is replacing recursion with loop, and it gets **20x faster** for high dimension ndarray (like 512^3 with more than 4 dims).

