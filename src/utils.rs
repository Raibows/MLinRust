use crate::dataset::dataloader::{Dataloader, BatchIterator};
use crate::dataset::{Dataset, TaskLabelType};
use crate::model::Model;

/// Trait for 
/// * &Dataset<T>
/// * &mut Dataloader<T, &Dataset<T>>
/// 
/// pass a dataloader will reduce the construction and enable batch prediction
pub trait EvaluateArgTrait<'a, T: TaskLabelType + Copy> {
    fn dataloader_iter(self, batch: usize) -> BatchIterator<T>;
}

impl<'a, T: TaskLabelType + Copy> EvaluateArgTrait<'a, T> for &Dataset<T> {
    fn dataloader_iter(self, batch: usize) -> BatchIterator<T> {
        let mut loader = Dataloader::new(self, batch, false, None);
        loader.iter_mut()
    }
}

impl<'a, T: TaskLabelType + Copy> EvaluateArgTrait<'a, T> for &mut Dataloader<T, &Dataset<T>> {
    fn dataloader_iter(self, batch: usize) -> BatchIterator<T> {
        if self.batch_size != batch {
            self.batch_size = batch;
        }
        self.iter_mut()
    }
}

/// evaluate classification dataset<usize>
/// * data: &Dataset<usize> or &mut Dataloader<usize, &Dataset<usize>>
/// * return: (correct_num, accuracy)
pub fn evaluate<'a, T: EvaluateArgTrait<'a, usize>>(data: T, model: &impl Model<usize>) -> (usize, f32)
{
    let mut correct = 0;
    let mut total = 0;
    for (feature, label) in data.dataloader_iter(128) {
        model.predict_with_batch(&feature).iter().zip(label.iter()).for_each(|(p, l)| {
            if p == l {
                correct += 1;
            }
            total += 1;
        })
    }
    (correct, correct as f32 / total as f32)
}

/// evaluate regression dataset<f32>
/// * data: &Dataset<f32> or &mut Dataloader<f32, &Dataset<f32>>
/// * return: mean absolute error
pub fn evaluate_regression<'a, T: EvaluateArgTrait<'a, f32>>(dataset: T, model: &impl Model<f32>) -> f32 {
    let mut error = 0.0;
    let mut total = 0;

    for (feature, label) in dataset.dataloader_iter(128) {
        error += model.predict_with_batch(&feature).iter().zip(label.iter())
        .fold(0.0, |s, (p, l)| {
            total += 1;
            s + (p - l).abs()
        });
    }
    error / total as f32
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