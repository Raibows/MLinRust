use std::{fs::OpenOptions, io::Read};
use rand::{thread_rng, seq::SliceRandom};

mod iris_dataset;
mod mobile_phone_price_predict;


pub enum DatasetName {
    IrisDataset,
    MobilePhonePricePredictDataset,
}

pub struct Dataset<T: TaskLabelType> {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<T>,
    pub label_map: Option<Vec<String>>,
}


#[derive(PartialEq, Debug)]
pub enum Task {
    Classification, // argmax
    Regression, // value
}

pub trait TaskLabelType {}
impl TaskLabelType for f32 {}
impl TaskLabelType for usize {}

pub trait FromPathDataset {
    type Output;
    fn from_name(path: &str, name: DatasetName) -> Self::Output;

    fn read_data_from_file(path: &str) -> std::io::Result<String> {
        let mut buf = String::with_capacity(1024 * 8);
        OpenOptions::new().read(true).open(path)?.read_to_string(&mut buf)?;
        Ok(buf)
    }
}

impl FromPathDataset for Dataset<usize> {
    type Output = Dataset<usize>;
    fn from_name(path: &str, name: DatasetName) -> Self::Output {
        if let Ok(data) = Self::read_data_from_file(path) {
            match name {
                DatasetName::IrisDataset => {
                    let (features, labels, label_map) = iris_dataset::process_iris_dataset(data);
                    Dataset::new(features, labels, label_map)
                },
                DatasetName::MobilePhonePricePredictDataset => {
                    let (features, labels, label_map) = mobile_phone_price_predict::process_mobile_phone_price_dataset(data);
                    Dataset::new(features, labels, label_map)
                }
            }
        } else {
            println!("Err when reading data");
            return Dataset::<usize>::default()
        }
    }
}

impl FromPathDataset for Dataset<f32> {
    type Output = Dataset<f32>;

    fn from_name(path: &str, name: DatasetName) -> Self::Output {
        let res = match name {
            // DatasetName::MobilePhonePricePredictDataset(p) => Self::read_data_from_file(p),
            _ => todo!(),
        };
        // if let Ok(data) = res {
        //     todo!()
        // } else {
        //     println!("Err when reading data {}");
        //     Dataset::<f32>::default()
        // }
        todo!()
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
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<T>, label_map: Option<Vec<String>>) -> Self {
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