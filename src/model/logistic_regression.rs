use super::Model;


struct LogitRegression {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
    class_num: usize,
}

impl LogitRegression {
    fn new<F>(feature_dim: usize, class_num: usize, weight_init_fn: F) -> Self
    where F: Fn(Vec<Vec<f32>>) -> Vec<Vec<f32>>
    {
        let mut weight = vec![vec![0.0f32; feature_dim]; class_num];
        let bias = vec![0.0f32; class_num];
        weight = weight_init_fn(weight);
        Self { weight: weight, bias: bias, feature_dim: feature_dim, class_num: class_num }
    }
}

impl Model<usize> for LogitRegression {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        todo!()
    }
}

