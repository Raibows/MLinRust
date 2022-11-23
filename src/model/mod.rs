use crate::dataset::TaskLabelType;
pub mod decision_tree;
pub mod logistic_regression;
pub mod utils;

pub trait Model<T: TaskLabelType> {
    fn predict(&self, feature: &Vec<f32>) -> T;
}