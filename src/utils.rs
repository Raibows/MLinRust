use crate::dataset::{Dataset, TaskLabelType};
use crate::model::Model;


pub fn evaluate<T: TaskLabelType + Copy + std::cmp::PartialEq>(dataset: &Dataset<T>, model: &impl Model<T>) -> (usize, f32) {
    let mut correct = 0;
    for i in 0..dataset.len() {
        let (feature, label) = dataset.get(i);
        if model.predict(feature) == *label {
            correct += 1;
        }
    }
    (correct, correct as f32 / dataset.len() as f32)
}

pub fn evaluate_regression(dataset: &Dataset<f32>, model: &impl Model<f32>) -> f32 {
    let mut error = 0.0;
    for i in 0..dataset.len() {
        let (feature, label) = dataset.get(i);
        error += (model.predict(feature) - label).abs();
    }
    error / dataset.len() as f32
}



/// pseudo random number generator
/// 
/// [Wiki linear congruential generator](https://en.wikipedia.org/wiki/Linear_congruential_generator)
/// 
/// Note that the max rand number is up to 2^31 - 1
#[derive(Debug, Clone, Copy)]
pub struct RandGenerator {
    seed: usize,
    a: usize,
    c: usize,
    m: usize,
}

pub trait RandRangeTrait {
    fn to_f32(self) -> f32;

    fn f32_to_self(n: f32) -> Self;
}


macro_rules! rand_range_trait_for {
    ($name:tt) => {
        impl RandRangeTrait for $name {
            fn to_f32(self) -> f32 {
                self as f32
            }

            fn f32_to_self(n: f32) -> Self {
                n as Self
            }
        }
    }
}

rand_range_trait_for!(i32);
rand_range_trait_for!(f32);
rand_range_trait_for!(usize);

impl RandGenerator {
    pub fn new(seed: usize) -> Self {
        // to avoid overflow, so init seed is up to 2^31
        Self { seed: seed & 0x7f_ff_ff_ff, a: 1103515245, c: 12345, m: 0x7f_ff_ff_ff }
    }

    /// Watch out the range, [0, 2^31 - 1] instead of uszie::max 2^64 - 1
    pub fn gen_u32(&mut self) -> usize {
        self.seed = (self.a * self.seed + self.c) % self.m;
        self.seed
    }

    /// generate rand f32 from [0.0, 1.0)
    pub fn gen_f32(&mut self) -> f32 {
        self.gen_u32() as f32 / (self.m - 1) as f32
    }

    /// generate rand usize(u32)/f32/i32 from [lower, upper)
    pub fn gen_range<T: RandRangeTrait>(&mut self, low: T, upper: T) -> T {
        T::f32_to_self(self.gen_f32() * upper.to_f32() + low.to_f32())
    }

    /// provable evenly shuffle
    /// 
    /// each element is swapped with **equal probability** to any positions
    /// 
    /// ## proof:
    /// 
    /// for the last element, there is no probability for other elements swap with it before, each position is 1/n;
    /// for the second last element, the probability for position [0, n-1] is 1/(n-1), and the probability for finally swapping at the last position is depending on the last element, i.e. 1/n; then for the first (n-1) positions, the probability is 1/(n-1) * (1 - 1/n) = 1/n; so as the (n-2), (n-3)...0th element
    pub fn shuffle<T>(&mut self, arr: &mut Vec<T>) {
        for i in 0..arr.len() {
            if self.gen_f32() > 1.0 / (i+1) as f32 {
                arr.swap(i, self.gen_range(0, i));
            } // otherwise keep
        }
    }
}

#[cfg(test)]
mod test {
    use super::RandGenerator;

    #[test]
    fn test_rand() {
        let mut rng = RandGenerator::new(0);
        // let mut p = 0.0;
        let mut high = 0;
        let mut low = usize::MAX;
        for _ in 0..10000 {
            println!("{} {} {}", rng.gen_f32(), rng.gen_range(-10.0, 11.0), rng.gen_range(0, 100));
            high = high.max(rng.gen_range(0, 100));
            low = low.min(rng.gen_range(0, 100));
        }
        assert!(low == 0);
        assert!(high == 99);

        let mut a: Vec<usize> = (0..10).collect();
        rng.shuffle(&mut a);
        println!("{:?}", a);
    }
}