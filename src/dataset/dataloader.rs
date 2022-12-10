use std::marker::PhantomData;

use crate::{ndarray::NdArray, utils::RandGenerator};

use super::{Dataset, TaskLabelType};


pub struct Dataloader<T: TaskLabelType + Copy, E: DatasetBorrowTrait<T>> {
    pub batch_size: usize,
    pub shuffle: bool,
    raw_dataset: E,
    rng: RandGenerator,
    phantom: PhantomData<T>,
}

pub struct BatchIterator<T: TaskLabelType + Copy> {
    batch_features: Vec<NdArray>,
    batch_labels: Vec<Vec<T>>,
}

impl<'a, T: TaskLabelType + Copy> Iterator for BatchIterator<T> {
    type Item = (NdArray, Vec<T>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.batch_features.is_empty() {
            None
        } else {
            Some((self.batch_features.pop().unwrap(), self.batch_labels.pop().unwrap()))
        }
    }
}

pub trait DatasetBorrowTrait<T: TaskLabelType + Copy> {
    fn total_len(&self) -> usize;

    fn get_feature(&self, idx: usize) -> Vec<f32>;

    fn get_label(&self, idx: usize) -> T;
}

macro_rules! write_dataset_to_dataloader_trait {
    () => {
        fn total_len(&self) -> usize {
            self.len()
        }
    
        fn get_feature(&self, idx: usize) -> Vec<f32> {
            self.features[idx].clone()
        }
    
        fn get_label(&self, idx: usize) -> T {
            self.labels[idx]
        }
    }
}

impl<T: TaskLabelType + Copy> DatasetBorrowTrait<T> for Dataset<T> {
    write_dataset_to_dataloader_trait!();
}

/// note that you should not expect to shuffle a evaluation dataset since evaluate often accepts a non mut reference, it cannot be shuffled
impl<T: TaskLabelType + Copy> DatasetBorrowTrait<T> for &Dataset<T> {
    write_dataset_to_dataloader_trait!();
}


impl<T: TaskLabelType + Copy, E: DatasetBorrowTrait<T>> Dataloader<T, E> {

    pub fn new(dataset: E, batch_size: usize, shuffle: bool, seed: Option<usize>) -> Self {
        Self { batch_size: batch_size, shuffle: shuffle, raw_dataset: dataset, rng: RandGenerator::new(seed.unwrap_or(0)), phantom: PhantomData}
    }

    fn init_batches(&mut self) -> (Vec<NdArray>, Vec<Vec<T>>) {
        let mut sampler: Vec<usize> = (0..self.raw_dataset.total_len()).collect();
        if self.shuffle {
            self.rng.shuffle(&mut sampler);
        }
        
        sampler.chunks(self.batch_size)
        .fold((Vec::with_capacity(self.raw_dataset.total_len() / self.batch_size + 1), Vec::with_capacity(self.raw_dataset.total_len() / self.batch_size + 1)), |mut s, batch| {
            let f: Vec<Vec<f32>> = batch.iter().map(|i| self.raw_dataset.get_feature(*i)).collect();
            let f = NdArray::new(f);
            let l: Vec<T> = batch.iter().map(|i| self.raw_dataset.get_label(*i)).collect();
            s.0.push(f);
            s.1.push(l);
            s
        })
    }

    pub fn iter_mut(&mut self) -> BatchIterator<T> {
        let (features, labels) = self.init_batches();
        BatchIterator {
            batch_features: features, 
            batch_labels: labels,
        }
    }
}


#[cfg(test)]
mod test {
    use super::{Dataloader, Dataset};

    #[test]
    fn test_dataloader_iterator() {
        let features = vec![vec![1.0, 2.0, 3.0]; 15];
        let labels = (0..15).map(|i| i as f32).collect();
        let dataset = Dataset::new(features, labels, None);
        let mut dataloader = Dataloader::new(dataset, 4, true, None);
        println!("epoch 1 ----------------------------");
        for batch in dataloader.iter_mut() {
            let (_, label) = batch;
            println!("{:?}", label);
        }
        println!("epoch 2 ----------------------------");
        for batch in dataloader.iter_mut() {
            let (_, label) = batch;
            println!("{:?}", label);
        }
    }
}