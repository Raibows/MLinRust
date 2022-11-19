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
