use crate::dataset::TaskLabelType;
pub mod decision_tree;

pub trait Model<T: TaskLabelType> {
    fn predict(&self, feature: &Vec<f32>) -> T;
}