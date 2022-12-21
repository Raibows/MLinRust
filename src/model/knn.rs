use crate::{dataset::{Dataset, TaskLabelType}, ndarray::NdArray};

use super::{Model, utils::minkowski_distance};


#[derive(Debug, Clone, Copy)]
pub enum KNNAlg {
    BruteForce(usize),
    KdTree,
}

struct KdNode<T: TaskLabelType + Copy> {
    feature_idx: usize,
    sample: Vec<f32>,
    label: T,
    left: Option<Box<KdNode<T>>>,
    right: Option<Box<KdNode<T>>>,
}

struct KDTree<T: TaskLabelType + Copy> {
    root: Option<Box<KdNode<T>>>,
    minkowski_distance_p: f32,
}

impl<T: TaskLabelType + Copy>  KDTree<T> {
    /// * features: \[batch, feature\]
    /// * labels: \[batch\]
    /// * total_dim: total dim of the feature
    /// * p: the parameter p of [minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)
    fn new(features: Vec<Vec<f32>>, labels: Vec<T>, total_dim: usize, p: usize) -> Self {
        assert!(features.len() > 0 && features.len() == labels.len());
        let feature_label_zip: Vec<(Vec<f32>, T)> = features.into_iter().zip(labels.into_iter()).map(|(f,l)| (f, l)).collect();


        Self { root:  Self::build(feature_label_zip, total_dim, 0), minkowski_distance_p: p as f32}
    }

    /// features: [batch, (feature, label)]
    fn build(mut feature_label_zip: Vec<(Vec<f32>, T)>, total_dim: usize, depth: usize) -> Option<Box<KdNode<T>>> {
        if feature_label_zip.len() == 0 {
            None
        } else if feature_label_zip.len() == 1 {
            let axis = depth % total_dim;
            let (feature, label) = feature_label_zip.pop().unwrap();
            Some(Box::new(KdNode {feature_idx: axis, label: label, sample: feature, left: None, right: None}))
        } else {
            let axis = depth % total_dim;
            feature_label_zip.sort_by(|a, b| {
                a.0[axis].partial_cmp(&b.0[axis]).unwrap()
            });


            let median = feature_label_zip.len() / 2;

            let right_feature_label_zip = feature_label_zip.split_off(median + 1);
            let (median_f, median_l) = feature_label_zip.pop().unwrap();

            
            let left = Self::build(feature_label_zip, total_dim, depth + 1);
            let right = Self::build(right_feature_label_zip, total_dim, depth + 1);
            
            Some(Box::new(KdNode {feature_idx: axis, label: median_l, sample: median_f, left: left, right: right}))
        }
    }
    
    /// find the nearest node around the query
    /// * return: (node_sample_feature, node_label, distance)
    fn nearest(&self, query: &Vec<f32>) -> (Vec<f32>, T, f32) {
        // the initial best records is trivial, so borrow query
        let records = self.recursive_nearest(&self.root, query, (query, None, f32::MAX));
        (records.0.clone(), records.1.unwrap(), records.2)
    }

    /// * return: (node_sample_feature, node_label, distance)
    fn recursive_nearest<'a>(&'a self, node: &'a Option<Box<KdNode<T>>>, query: &Vec<f32>, mut best_records: (&'a Vec<f32>, Option<T>, f32)) -> (&Vec<f32>, Option<T>, f32) {
        if node.is_none() {
            best_records
        } else {
            // calculate distance from query and current node
            let d = minkowski_distance(query, &node.as_ref().unwrap().sample, self.minkowski_distance_p);

            let node = node.as_ref().unwrap();

            // update best records
            if d < best_records.2 {
                best_records.0 = &node.sample;
                best_records.1 = Some(node.label);
                best_records.2 = d;
            }

            // find the best from subtrees
            // good is the one that follows the median value (less goes left, more goes right)
            // then, bad is the opposite choice
            let (good, bad) = if query[node.feature_idx] < node.sample[node.feature_idx] {
                (&node.left, &node.right)
            } else {
                (&node.right, &node.left)
            };

            // explore the good side
            best_records = self.recursive_nearest(good, query, best_records);

            // explore the bad side
            // only if it has probability for less than the best distance, i.e., other features except feature[axis] are equal to query (has that probability)
            // otherwise, take pruning
            if (query[node.feature_idx] - node.sample[node.feature_idx]).abs() < best_records.2 {
                best_records = self.recursive_nearest(bad, query, best_records);
            }
            
            best_records
        }
    }
}



/// KNN classifier implemented by [KDTree](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
pub struct KNearestNeighbor {
    alg: KNNAlg,
}


impl KNearestNeighbor {
    pub fn new(alg: KNNAlg) -> Self {
        Self { alg: alg }
    }

    pub fn train<T: TaskLabelType + Copy>(&mut self, mut dataset: Dataset<T>) {
        let total_dim = dataset.feature_len();
        let mut features = NdArray::new(dataset.features);
        features = features.permute(vec![1, 0]); // [dim, batch]
        match self.alg {
            KNNAlg::KdTree => {
                
            },
            _ => {},
        }
    }
}

#[cfg(test)]
mod test {
    use super::KNearestNeighbor;
    use super::KDTree;

    #[test]
    fn test_kdtree() {
        let features = vec![
            vec![2.0, 3.0],
            vec![5.0, 4.0],
            vec![9.0, 6.0],
            vec![4.0, 7.0],
            vec![8.0, 1.0],
            vec![7.0, 2.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = KDTree::new(features, labels, 2, 2);
        let query = vec![6.0, 7.0];
        
        println!("nearest {:?}", tree.nearest(&query));
    }

    #[test]
    fn test_knn() {
        let mut v = vec![1, 2, 3];
        println!("{:?}", v.split_off(3));
    }
}