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
