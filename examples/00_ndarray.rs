use std::time::{Instant};
use mlinrust::ndarray::NdArray;

// numpy with 4 threads, 7.7 seconds
// rust version 30 seconds~
fn main() {
    const PSIZE: usize = 512 * 512 * 512;
    let mut a = NdArray::new((0..PSIZE).map(|i| i as f32).collect::<Vec<f32>>());
    a.reshape(vec![8, 128, 256, 512]);
    a += 1.0;
    a /= 0.5;

    let mut b = NdArray::new((0..PSIZE).map(|i| i as f32).collect::<Vec<f32>>());
    b.reshape(vec![128, 512, 2048]);
    b -= 1.5;
    b = &b + &b;


    let start = Instant::now();
    let _ = a * b; // large matrix multiplication
    let dur = start.elapsed();
    println!("execute {dur:?}");
}