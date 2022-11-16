
pub trait TaskLabelType {}
impl TaskLabelType for f32 {}
impl TaskLabelType for usize {}

pub struct Dataset<T: TaskLabelType> {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<T>,
}

impl<T:TaskLabelType> Dataset<T> {
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<T>) -> Self {
        Self { features: features, labels: labels }
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
}

impl<T: TaskLabelType + Copy> From<Vec<(&Vec<f32>, &T)>> for Dataset<T> {
    fn from(data: Vec<(&Vec<f32>, &T)>) -> Self {

        let mut features = vec![];
        let mut labels = vec![];

        data.into_iter().for_each(|item| {
            features.push((*item.0).clone());
            labels.push(*item.1);
        });

        Dataset::<T> { features: features, labels: labels }
    }
}

#[derive(PartialEq)]
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