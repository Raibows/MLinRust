use std::collections::HashMap;

pub struct Dataset {
    features: Vec<Vec<f32>>,
    labels: Vec<usize>,
}

impl Dataset {
    pub fn new(path: String) -> Self {
        todo!()
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
}

pub fn extract_features_labels(dataset: &Vec<HashMap<String, f32>>) -> (Vec<HashMap<String, f32>>, Vec<HashMap<String, f32>>) {
    let features: Vec<HashMap<String, f32>> = dataset.iter().map(|x| {
        let t: HashMap<String, f32> = x.iter().filter_map(|item| {
            if item.0 != "label" {
                Some((item.0.clone(), *item.1))
            } else {
                None
            }
        }).collect();
        t
    }).collect();
    let labels: Vec<HashMap<String, f32>> = dataset.iter().map(|x| {
        let t: HashMap<String, f32> = x.iter().filter_map(|item| {
            if item.0 == "label" {
                Some((item.0.clone(), *item.1))
            } else {
                None
            }
        }).collect();
        t
    }).collect();
    (features, labels)
}