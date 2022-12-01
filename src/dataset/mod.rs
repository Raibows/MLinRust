use std::{fs::OpenOptions, io::Read, collections::HashMap};
use rand::{thread_rng, seq::SliceRandom};

mod iris_dataset;
mod mobile_phone_price_predict;
mod car_price_dataset;
mod utils;
pub mod dataloader;


#[derive(Debug)]
pub enum DatasetName {
    IrisDataset,
    MobilePhonePricePredictDataset,
    CarPriceRegressionDataset,
}

#[derive(Clone)]
pub struct Dataset<T: TaskLabelType> {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<T>,
    pub label_map: Option<HashMap<usize, String>>, // for regression type dataset, there is no need for label_map
}

pub struct DatasetIterator<'a, T: TaskLabelType + Copy> {
    iter_idx: usize,
    dataset: &'a Dataset<T>,
}

impl<'a, T: TaskLabelType + Copy> Iterator for DatasetIterator<'a, T> {
    type Item = (&'a Vec<f32>, T);
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.iter_idx;
        if idx >= self.dataset.len() {
            None
        } else {
            self.iter_idx += 1;
            let item = self.dataset.get(idx);
            Some((item.0, *item.1))
        }
    }
}

impl<'a, T: TaskLabelType + Copy> IntoIterator for &'a Dataset<T> {
    type Item = <DatasetIterator<'a, T> as Iterator>::Item;
    type IntoIter = DatasetIterator<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        DatasetIterator {
            dataset: self,
            iter_idx: 0
        }
    }
}


pub trait TaskLabelType {}
impl TaskLabelType for f32 {}
impl TaskLabelType for usize {}

pub trait FromPathDataset {
    type Output;
    fn from_name(path: &str, name: DatasetName, fill_missing_value: Option<utils::ImputeType>) -> Self::Output;

    fn read_data_from_file(path: &str) -> std::io::Result<String> {
        let mut buf = String::with_capacity(1024 * 8);
        OpenOptions::new().read(true).open(path)?.read_to_string(&mut buf)?;
        Ok(buf)
    }
}

impl FromPathDataset for Dataset<usize> {
    type Output = Dataset<usize>;
    fn from_name(path: &str, name: DatasetName, _fill_missing_value: Option<utils::ImputeType>) -> Self::Output {
        if let Ok(data) = Self::read_data_from_file(path) {
            match name {
                DatasetName::IrisDataset => {
                    let (features, labels, label_map) = iris_dataset::process_iris_dataset(data);
                    Dataset::new(features, labels, label_map)
                },
                DatasetName::MobilePhonePricePredictDataset => {
                    let (features, labels, label_map) = mobile_phone_price_predict::process_mobile_phone_price_dataset(data);
                    Dataset::new(features, labels, label_map)
                },
                _ => unimplemented!("dataset type {:?} is not implemented for classification<usize>", name),
            }
        } else {
            panic!("Err when reading data from {}", path)
        }
    }
}

impl FromPathDataset for Dataset<f32> {
    type Output = Dataset<f32>;

    fn from_name(path: &str, name: DatasetName, fill_missing_value: Option<utils::ImputeType>) -> Self::Output {
        if let Ok(data) = Self::read_data_from_file(path) {
            match name {
                DatasetName::CarPriceRegressionDataset => {
                    let (features, labels, label_map) = car_price_dataset::process_tianchi_car_price_regression_dataset(data, fill_missing_value.unwrap_or(utils::ImputeType::Mean));
                    Dataset::new(features, labels, label_map)
                },
                _ => unimplemented!("dataset type {:?} is not implemented for regression<f32>", name),
            }
        } else {
            panic!("Err when reading data from {}", path)
        }
    }
}

impl<T: TaskLabelType> Default for Dataset<T> {
    fn default() -> Self {
        Dataset { features: vec![vec![]], labels: vec![], label_map: None }
    }
}

impl<T: TaskLabelType + Copy> From<Vec<(&Vec<f32>, &T)>> for Dataset<T> {
    fn from(data: Vec<(&Vec<f32>, &T)>) -> Self {

        let mut features = vec![];
        let mut labels = vec![];

        data.into_iter().for_each(|item| {
            features.push((*item.0).clone());
            labels.push(*item.1);
        });

        Dataset::<T>::new(features, labels, None)
    }
}

impl<T:TaskLabelType + Copy> Dataset<T> {
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<T>, label_map: Option<HashMap<usize, String>>) -> Self {
        assert_eq!(features.len(), labels.len());
        Self { features: features, labels: labels, label_map: label_map}
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn feature_len(&self) -> usize {
        if let Some(item) = self.features.get(0) {
            item.len()
        } else {
            0
        }
    }

    pub fn class_num(&self) -> usize {
        assert!(self.label_map.is_some(), "dataset is not for classification!");
        self.label_map.as_ref().unwrap().len()
    }

    pub fn get_feature_by_idx(&self, feature_idx: usize) -> Vec<&f32> {
        assert!(feature_idx < self.feature_len());
        self.features.iter().map(|item|{
            item.get(feature_idx).unwrap()
        }).collect()
    }

    pub fn get_unique_feature_values(&self, feature_idx: usize) -> Vec<&f32> {
        let mut features = self.get_feature_by_idx(feature_idx);
        features.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut temp: Vec<&f32> = vec![];
        for item in features {
            if item != *temp.last().unwrap_or(&&f32::MIN) {
                temp.push(item);
            }
        }
        temp
    }

    pub fn get(&self, idx: usize) -> (&Vec<f32>, &T) {
        assert!(idx < self.len());
        (&self.features[idx], &self.labels[idx])
    }

    pub fn shuffle(&mut self, _seed: usize) {
        let mut rng = thread_rng();
        let mut idxs: Vec<usize> = (0..self.len()).collect();
        idxs.shuffle(&mut rng);
        let mut features = vec![vec![]; self.len()];
        let mut labels = vec![*self.labels.first().unwrap(); self.len()];
        for i in idxs {
            features[i] = self.features.remove(0);
            labels[i] = self.labels.remove(0);
        }
        self.features = features;
        self.labels = labels;
    }

    pub fn split_dataset(mut self, mut ratio: Vec<f32>) -> Vec<Dataset<T>> {
        let total = self.len();
        // normalize the passed in ratio to ensure the sum is 1.0
        let t = ratio.iter().sum::<f32>();
        ratio.iter_mut().for_each(|item| *item /= t);
        
        let mut res = vec![];
        let mut start = 0;
        for r in ratio {
            let r = (total as f32 * r) as usize;
            let mut features = Vec::with_capacity(r);
            let mut labels = Vec::with_capacity(r);
            for i in 0..r {
                if start + i == total {
                    break;
                }
                features.push(self.features.remove(0));
                labels.push(self.labels.remove(0));
            }
            start += r;
            res.push(Self::new(features, labels, self.label_map.clone()));
        }
        for (feature, label) in self.features.into_iter().zip(self.labels.into_iter()) {
            res.last_mut().unwrap().features.push(feature);
            res.last_mut().unwrap().labels.push(label);
        }
        res
    }
}

#[cfg(test)]
mod test {
    use super::{Dataset, DatasetName, FromPathDataset};

    #[test]
    fn test_dataset_iter() {
        let path = ".data/IRIS.csv";
        let dataset = Dataset::<usize>::from_name(path, DatasetName::IrisDataset, None);
        for (x, y) in &dataset {
            println!("{:?} {}", x, y);
        }
        for (x, y) in &dataset {
            println!("{:?} {}", x, y);
        }
    }
}