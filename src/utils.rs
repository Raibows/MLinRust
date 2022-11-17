use std::{fs::OpenOptions, io::Read, collections::HashMap};



pub enum DatasetName {
    IrisDataset(&'static str),
}


pub trait TaskLabelType {}
impl TaskLabelType for f32 {}
impl TaskLabelType for usize {}

pub struct Dataset<T: TaskLabelType> {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<T>,
    pub label_map: Option<Vec<String>>,
}


fn process_iris_dataset(data: String) -> (Vec<Vec<f32>>, Vec<usize>, Vec<String>) {
    let lines: Vec<&str> = data.split("\n").collect();
    let mut lines = lines.iter();
    let mut features: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
    let mut labels: Vec<usize> = Vec::with_capacity(lines.len());
    let mut label_map: HashMap<String, usize> = HashMap::new();
    lines.next();
    for (i, l) in lines.enumerate() {
        // println!("{i}\t\t🚗{:?}🚗", l);

        if l.len() == 0 {
            continue;
        }
        
        let temp: Vec<&str> = l.split(",").collect();
        let label = temp.last().unwrap().to_string();
        let size = label_map.len();
        labels.push(
            *label_map.entry(label).or_insert(size)
        );
        features.push(
            temp.iter().take(4).map(|item| item.parse::<f32>().unwrap()).collect::<Vec<f32>>()
        )
    }
    let label_map: Vec<String> = label_map.into_iter().map(|(k, _)| k).collect();
    println!("{:?}", labels);
    (features, labels, label_map)
}



pub trait FromPathDataset {
    type Output;
    fn from_name(name: DatasetName) -> Self::Output;

    fn read_data_from_file(path: &str) -> std::io::Result<String> {
        let mut buf = String::with_capacity(4096);
        OpenOptions::new().read(true).open(path)?.read_to_string(&mut buf)?;
        Ok(buf)
    }
}

impl<T:TaskLabelType> FromPathDataset for Dataset<T> {
    type Output = Dataset<usize>;
    fn from_name(name: DatasetName) -> Self::Output {
        let DatasetName::IrisDataset(p) = name;
        if let Ok(data) = Self::read_data_from_file(p) {
            let res = process_iris_dataset(data);
            Dataset::new(res.0, res.1, Some(res.2))
        } else {
            println!("Err when reading data");
            Dataset::<usize>::default()
        }
    }
}

impl<T: TaskLabelType> Default for Dataset<T> {
    fn default() -> Self {
        Dataset { features: vec![vec![]], labels: vec![], label_map: None }
    }
}

impl<T:TaskLabelType> Dataset<T> {
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<T>, label_map: Option<Vec<String>>) -> Self {
        Self { features: features, labels: labels, label_map: label_map}
    }

    // pub fn from_file(name: DatasetName) -> std::io::Result<Dataset<T>> {
    //     let mut buf = String::with_capacity(4096);
    //     match name {
    //         DatasetName::IrisDataset(path) => {
    //             OpenOptions::new().read(true).open(path)?.read_to_string(&mut buf)?;
    //             let res = process_iris_dataset(buf);
    //             // Ok(Self::new(res.0, res.1, Some(res.2)))
    //             Ok(Self {features: res.0, labels: res.1, label_map: Some(res.2)})
    //         },
    //     }

        // Ok(Self::new(vec![vec![]], vec![], None))
    // }

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



#[derive(PartialEq, Debug)]
pub enum Task {
    Classification, // argmax
    Regression, // value
}


// pub fn extract_features_labels(dataset: &Vec<HashMap<String, f32>>) -> (Vec<HashMap<String, f32>>, Vec<HashMap<String, f32>>) {
//     let features: Vec<HashMap<String, f32>> = dataset.iter().map(|x| {
//         let t: HashMap<String, f32> = x.iter().filter_map(|item| {
//             if item.0 != "label" {
//                 Some((item.0.clone(), *item.1))
//             } else {
//                 None
//             }
//         }).collect();
//         t
//     }).collect();
//     let labels: Vec<HashMap<String, f32>> = dataset.iter().map(|x| {
//         let t: HashMap<String, f32> = x.iter().filter_map(|item| {
//             if item.0 == "label" {
//                 Some((item.0.clone(), *item.1))
//             } else {
//                 None
//             }
//         }).collect();
//         t
//     }).collect();
//     (features, labels)
// }