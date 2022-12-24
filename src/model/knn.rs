use std::collections::{BinaryHeap, HashMap};

use crate::{dataset::{Dataset, TaskLabelType}, ndarray::{NdArray, utils::softmax}};

use super::{Model, utils::minkowski_distance};


#[derive(Debug, Clone, Copy)]
pub enum KNNAlg {
    BruteForce,
    KdTree,
}

/// Uniform means marjorty voting for classification task
/// Distance means weighting ensemble based on distance
#[derive(Debug, Clone, Copy)]
pub enum KNNWeighting {
    Uniform,
    Distance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryRecord<'a, T: TaskLabelType + Copy + std::cmp::PartialEq> {
    feature: &'a Vec<f32>,
    label: T,
    distance: f32,
}

impl<'a, T: TaskLabelType + Copy + std::cmp::PartialEq> Eq for QueryRecord<'a, T> {
    
}

impl<'a, T: TaskLabelType + Copy + std::cmp::PartialEq> Ord for QueryRecord<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<'a, T: TaskLabelType + Copy + std::cmp::PartialEq> PartialOrd for QueryRecord<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
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
    k: usize,
    weighting: KNNWeighting,
}

pub trait KNNInterface<T: TaskLabelType + Copy + std::cmp::PartialEq> {
    /// find the nearest k nodes around the query
    /// * return: ordering Vector<(node_sample_feature, node_label, distance)>, size = k
    fn nearest<'a>(&'a self, query: &Vec<f32>) -> Vec<QueryRecord<'a, T>>;

    fn get_weighting(&self) -> KNNWeighting;

}

impl<T: TaskLabelType + Copy + std::cmp::PartialEq + 'static> KNNInterface<T> for KDTree<T> {

    fn nearest<'a>(&'a self, query: &Vec<f32>) -> Vec<QueryRecord<'a, T>> {
        // the initial best records is trivial, so borrow query
        assert!(self.root.is_some());

        let records_heap = BinaryHeap::new();
        let mut records_heap = self.recursive_nearest(&self.root, query, records_heap);
        let mut nearest: Vec<QueryRecord<'a, T>>  = vec![];
        while let Some(item) = records_heap.pop() {
            nearest.push(item);
        }
        nearest.reverse();
        nearest
    }

    fn get_weighting(&self) -> KNNWeighting {
        self.weighting
    }
}

impl<T: TaskLabelType + Copy + std::cmp::PartialEq + 'static>  KDTree<T> {
    
    /// * k: k nearest neighbours
    /// * weighting: weighting the neibours, default is these neighbours are equal
    /// * features: \[batch, feature\]
    /// * labels: \[batch\]
    /// * total_dim: total dim of the feature
    /// * p: the parameter p of [minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)
    ///     * default is p = 2
    fn new(k: usize, weighting: Option<KNNWeighting>, features: Vec<Vec<f32>>, labels: Vec<T>, total_dim: usize, p: Option<usize>) -> Box<dyn KNNInterface<T>> {
        assert!(features.len() > 0 && features.len() == labels.len());
        assert!(k > 0);
        let feature_label_zip: Vec<(Vec<f32>, T)> = features.into_iter().zip(labels.into_iter()).map(|(f,l)| (f, l)).collect();

        // Box::new(Self { root: None, minkowski_distance_p: p.unwrap_or(2) as f32, k: k, weighting: weighting.unwrap_or(KNNWeighting::Uniform)})

        Box::new(Self { root:  Self::build(feature_label_zip, total_dim, 0), minkowski_distance_p: p.unwrap_or(2) as f32, k: k, weighting: weighting.unwrap_or(KNNWeighting::Uniform)})
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
    
    /// * return: MaxHeap<queryrecord>
    fn recursive_nearest<'a>(&'a self, node: &'a Option<Box<KdNode<T>>>, query: &Vec<f32>, mut records_heap: BinaryHeap<QueryRecord<'a, T>>) -> BinaryHeap<QueryRecord<'a, T>> {
        if node.is_none() {
            records_heap
        } else {
            // calculate distance from query and current node
            let d = minkowski_distance(query, &node.as_ref().unwrap().sample, self.minkowski_distance_p);

            let node = node.as_ref().unwrap();

            // update best records
            if records_heap.len() == self.k {
                let worst_record = records_heap.peek().unwrap();
                if worst_record.distance > d {
                    records_heap.pop();
                    records_heap.push(QueryRecord { feature: &node.sample, label: node.label, distance: d });
                }
            } else {
                records_heap.push(QueryRecord { feature: &node.sample, label: node.label, distance: d });
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
            records_heap = self.recursive_nearest(good, query, records_heap);

            // explore the bad side
            // only if it has probability for less than the best distance, i.e., other features except feature[axis] are equal to query (has that probability)
            // otherwise, take pruning
            let worst_record = records_heap.peek().unwrap();
            if records_heap.len() < self.k ||
            (query[node.feature_idx] - node.sample[node.feature_idx]).abs() < worst_record.distance {
                records_heap = self.recursive_nearest(bad, query, records_heap);
            }
            
            records_heap
        }
    }
}


struct BruteForceSearch<T: TaskLabelType + Copy> {
    k: usize,
    minkowski_distance_p: f32,
    weighting: KNNWeighting,   
    features: Vec<Vec<f32>>,
    labels: Vec<T>,
}

impl<T: TaskLabelType + Copy + PartialEq> KNNInterface<T> for BruteForceSearch<T> {
    fn nearest<'a>(&'a self, query: &Vec<f32>) -> Vec<QueryRecord<'a, T>> {
        let mut records_heap: BinaryHeap<QueryRecord<'a, T>> = BinaryHeap::new();
        for (feature, label) in self.features.iter().zip(self.labels.iter()) {
            let d = minkowski_distance(query, feature, self.minkowski_distance_p);
            if records_heap.len() == self.k {
                let worst_record = records_heap.peek().unwrap();
                if d < worst_record.distance {
                    records_heap.pop();
                    records_heap.push(
                        QueryRecord { feature: feature, label: *label, distance: d }
                    );
                }
            } else {
                records_heap.push(
                    QueryRecord { feature: feature, label: *label, distance: d }
                );
            }
        }
        let mut res = vec![];
        while let Some(item) = records_heap.pop() {
            res.push(item);
        }
        res.reverse();
        res
    }

    fn get_weighting(&self) -> KNNWeighting {
        self.weighting
    }
}

impl<T: TaskLabelType + Copy + std::cmp::PartialEq + 'static> BruteForceSearch<T> {
    /// * k: k nearest neighbours
    /// * weighting: weighting the neibours, default is these neighbours are equal
    /// * features: \[batch, feature\]
    /// * labels: \[batch\]
    /// * p: the parameter p of [minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance)
    ///     * default is p = 2
    fn new(k: usize, weighting: Option<KNNWeighting>, features: Vec<Vec<f32>>, labels: Vec<T>, p: Option<usize>) -> Box<dyn KNNInterface<T>> {
        assert!(features.len() > 0 && features.len() == labels.len());
        assert!(k > 0);
        Box::new(Self { k: k, minkowski_distance_p: p.unwrap_or(2) as f32, weighting: weighting.unwrap_or(KNNWeighting::Uniform), features: features, labels: labels })
    }
}


impl Model<usize> for dyn KNNInterface<usize> {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let res = self.nearest(feature);
        let mut predicts: HashMap<usize, f32> = HashMap::new();
        for item in res {
            *predicts.entry(item.label).or_insert(0.0) += match self.get_weighting() {
                KNNWeighting::Distance => 1.0 / f32::max(item.distance, 1e-6),
                KNNWeighting::Uniform => 1.0,
            }
        }
        predicts.iter().fold((0, f32::MAX), |s, i| {
            if *i.1 > s.1 {
                (*i.0, *i.1)
            } else {
                s
            }
        }).0
    }
}

impl Model<f32> for dyn KNNInterface<f32> {
    fn predict(&self, feature: &Vec<f32>) -> f32 {
        let res = self.nearest(feature);
        let weights = match self.get_weighting() {
            KNNWeighting::Distance => {
                let mut a = NdArray::new(res.iter().map(|i| i.distance).collect::<Vec<f32>>());
                softmax(&mut a, 0);
                a.destroy().1
            },
            KNNWeighting::Uniform => {
                vec![1.0 / res.len() as f32; res.len()]
            }
        };
        res.iter().zip(weights.iter()).fold(0.0, |s, (i, w)| {
            s + i.label * w
        })
    }
}



/// KNN implemented by [KDTree](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) or BruteForceSearch
pub struct KNNModel<T: TaskLabelType + Copy + PartialEq> {
    pub alg: KNNAlg,
    interface: Box<dyn KNNInterface<T>>,
}


impl<T: TaskLabelType + Copy + PartialEq + 'static> KNNModel<T> {
    /// initialization is training of KdTree; for BruteForce, it is lazy training (i.e., no training)
    /// * alg: algorithm of knn
    /// * k: k nearest neighbours
    /// * weighting: for the ensemble results
    ///     * default is uniform
    /// * p: parameter of minkowski distance
    ///     * default is 2, i.e., Euclidean distance
    pub fn new(alg: KNNAlg, k: usize, weighting: Option<KNNWeighting>, dataset: Dataset<T>, p: Option<usize>) -> Self {
        let interface= match alg {
            KNNAlg::BruteForce => {
                BruteForceSearch::new(k, weighting, dataset.features, dataset.labels, p)
            },
            KNNAlg::KdTree => {
                let total_dim = dataset.feature_len();
                KDTree::new(k, weighting, dataset.features, dataset.labels, total_dim, p)
            }
        };
        Self { alg: alg, interface: interface }
    }

    pub fn nearest(&self, query: &Vec<f32>) -> Vec<QueryRecord<T>> {
        self.interface.nearest(query)
    }
}

impl Model<usize> for KNNModel<usize> {
    fn predict(&self, feature: &Vec<f32>) -> usize {
        let res = self.interface.nearest(feature);
        let mut predicts: HashMap<usize, f32> = HashMap::new();
        for item in res {
            *predicts.entry(item.label).or_insert(0.0) += match self.interface.get_weighting() {
                KNNWeighting::Distance => 1.0 / f32::max(item.distance, 1e-6),
                KNNWeighting::Uniform => 1.0,
            }
        }
        predicts.iter().fold((0, f32::MIN), |s, i| {
            if *i.1 > s.1 {
                (*i.0, *i.1)
            } else {
                s
            }
        }).0
    }
}

impl Model<f32> for KNNModel<f32> {
    fn predict(&self, feature: &Vec<f32>) -> f32 {
        let res = self.interface.nearest(feature);
        let weights = match self.interface.get_weighting() {
            KNNWeighting::Distance => {
                let mut a = NdArray::new(res.iter().map(|i| i.distance).collect::<Vec<f32>>());
                softmax(&mut a, 0);
                a.destroy().1
            },
            KNNWeighting::Uniform => {
                vec![1.0 / res.len() as f32; res.len()]
            }
        };
        res.iter().zip(weights.iter()).fold(0.0, |s, (i, w)| {
            s + i.label * w
        })
    }
}

#[cfg(test)]
mod test {
    use crate::dataset::{Dataset};
    use crate::model::Model;
    use crate::model::knn::{KNNWeighting, BruteForceSearch};

    use super::{KNNModel, KNNAlg};
    use super::{KDTree};

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
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let tree = KDTree::new(20, Some(KNNWeighting::Distance), features, labels, 2, Some(2));
        let query = vec![6.0, 7.0];
        let results =  tree.nearest(&query);
        println!("size {} predict {}\nnearest {:?}", results.len(), tree.predict(&query), results);
    }

    #[test]
    fn test_brute_force_search() {
        let features = vec![
            vec![2.0, 3.0],
            vec![5.0, 4.0],
            vec![9.0, 6.0],
            vec![4.0, 7.0],
            vec![8.0, 1.0],
            vec![7.0, 2.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let tree = BruteForceSearch::new(20, Some(KNNWeighting::Distance), features, labels, Some(2));
        let query = vec![6.0, 7.0];
        let results =  tree.nearest(&query);
        println!("size {} predict {}\nnearest {:?}", results.len(), tree.predict(&query), results);
    }

    #[test]
    fn test_knn() {
        let features = vec![
            vec![2.0, 3.0],
            vec![5.0, 4.0],
            vec![9.0, 6.0],
            vec![4.0, 7.0],
            vec![8.0, 1.0],
            vec![7.0, 2.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let dataset = Dataset::new(features, labels, None);
        let knn = KNNModel::new(KNNAlg::KdTree, 1, None, dataset, None);
        let query = vec![7.0, 1.9];
        println!("nearest {:?}\npredict = {}", knn.nearest(&query), knn.predict(&query));
    }
}