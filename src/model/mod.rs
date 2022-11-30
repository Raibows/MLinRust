use crate::dataset::TaskLabelType;
pub mod decision_tree;
pub mod logistic_regression;
pub mod utils;
pub mod linear_regression;
pub mod naive_bayes;

pub trait Model<T: TaskLabelType> {
    // todo implement generics for feature to accept batch or ndarray etc.
    fn predict(&self, feature: &Vec<f32>) -> T;
}