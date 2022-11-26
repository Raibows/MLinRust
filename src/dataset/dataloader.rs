use crate::ndarray::NdArray;

use super::{Dataset, TaskLabelType};


pub struct Dataloader<T: TaskLabelType> {
    pub batch_size: usize,
    pub shuffle: bool,
    batch_features: Option<Vec<NdArray>>, // for now, ndarray is not supporting &T, so we have to take clone
    batch_labels: Option<Vec<Vec<T>>>,
    raw_dataset: Dataset<T>,
}

pub struct BatchIterator<T: TaskLabelType> {
    batch_features: Vec<Option<NdArray>>, 
    batch_labels: Vec<Option<Vec<T>>>,
    iter_idx: usize,
}

impl<T: TaskLabelType> Iterator for BatchIterator<T> {
    type Item = (NdArray, Vec<T>);
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.iter_idx;
        if idx >= self.batch_labels.len() {
            None
        } else {
            self.iter_idx += 1;
            let feature = std::mem::replace(&mut self.batch_features[idx], None).unwrap();
            let label = std::mem::replace(&mut self.batch_labels[idx], None).unwrap();
            Some((feature, label))
        }
    }
}

impl<T: TaskLabelType + Copy> IntoIterator for &mut Dataloader<T> {
    type Item = <BatchIterator<T> as Iterator>::Item;
    type IntoIter = BatchIterator<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.init_batches();

        let batch_features = std::mem::replace(&mut self.batch_features, None).unwrap();
        let batch_labels = std::mem::replace(&mut self.batch_labels, None).unwrap();

        BatchIterator {
            batch_features: batch_features.into_iter().map(|i| Some(i)).collect(), 
            batch_labels: batch_labels.into_iter().map(|i| Some(i)).collect(),
            iter_idx: 0,
        }        
    }
}

impl<T: TaskLabelType + Copy> Dataloader<T> {

    pub fn new(dataset: Dataset<T>, batch_size: usize, shuffle: bool) -> Self {
        Self { batch_size: batch_size, shuffle: shuffle, batch_features: None, batch_labels: None, raw_dataset: dataset }
    }

    fn init_batches(&mut self) {
        if self.shuffle {
            self.raw_dataset.shuffle(0)
        }
        let sampler: Vec<usize> = (0..self.raw_dataset.len()).collect();
        let iter = sampler.chunks(self.batch_size);
        let mut batch_features = vec![];
        let mut batch_labels = vec![];
        for batch in iter {
            let f: Vec<Vec<f32>> = batch.iter().map(|i| self.raw_dataset.features[*i].clone()).collect();
            let f = NdArray::new(f);
            let l: Vec<T> = batch.iter().map(|i| self.raw_dataset.labels[*i]).collect();
            batch_features.push(f);
            batch_labels.push(l);
        }
        self.batch_features = Some(batch_features);
        self.batch_labels = Some(batch_labels);
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
        let mut dataloader = Dataloader::new(dataset, 4, true);
        println!("epoch 1 ----------------------------");
        for batch in &mut dataloader {
            let (_, label) = batch;
            println!("{:?}", label);
        }
        println!("epoch 2 ----------------------------");
        for batch in dataloader.into_iter() {
            let (_, label) = batch;
            println!("{:?}", label);
        }
    }
}