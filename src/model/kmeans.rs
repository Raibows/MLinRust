use crate::{utils::RandGenerator, model::utils::minkowski_distance};
use super::Model;

pub struct KMeansClustering {
    pub k: usize,
    pub clusters: Vec<Vec<f32>>,
    p: f32,
}

impl KMeansClustering {

    /// build K-Means model with unsupervised learning
    /// * k: number of clusters
    /// * max_iter: max iterations
    /// * p: decide the distance, parameter of minkowski distance
    ///     * default: 2, i.e., Euclidean distance
    /// * init_clusters: you can give a init clusters
    ///     * default: it will random choose k clusters from the samples
    /// * early_stop: decide the tolerance error of the distance between the last two iterations
    ///     * default: 1e-3
    /// * seed: seed for randomly choosing the init clusters
    ///     * default: 0
    /// * features: the samples set
    pub fn new(k: usize, max_iter: usize, p: Option<usize>, init_clusters: Option<Vec<Vec<f32>>>, early_stop: Option<f32>, seed: Option<usize>, features: &Vec<Vec<f32>>) -> Self {
        assert!(k > 0 && k <= features.len());
        let mut rng = RandGenerator::new(seed.unwrap_or(0));
        let mut clusters = init_clusters.unwrap_or(
            rng.choice(&features, k, false)
        );
        // check whether the dim of init clusters is matching the feature dim
        assert!(clusters.len() == k && clusters[0].len() == features[0].len());
        let p = p.unwrap_or(2) as f32;
        let early_stop = early_stop.unwrap_or(1e-3);
        
        // starting iteration
        let feature_dim = features[0].len();
        for _ in 0..max_iter {
            let mut new_clusters = vec![vec![0.0; feature_dim]; k];
            let mut cnts = vec![0; k];
            features.iter().for_each(|item| {

                // find the center that is closest to the sample(item)
                let idx = clusters.iter().enumerate().fold((0, f32::MAX), |s, (i, center)| {
                    let d = minkowski_distance(center, item, p);
                    if s.1 < d {
                        s
                    } else {
                        (i, d)
                    }
                }).0;
                
                // accumulate to the cluster center
                cnts[idx] += 1;
                new_clusters[idx].iter_mut().zip(item.iter()).for_each(|(nc, i)| {                   
                     *nc += i;
                });

            });

            // take average
            new_clusters.iter_mut().zip(cnts.into_iter()).for_each(|(cl, c)| {
                cl.iter_mut().for_each(|i| *i /= f32::max(c as f32, 1e-6));
            });

            // see if the new clusters are as same as the old_clusters, then we can decide early stop
            let err = new_clusters.iter().zip(clusters.iter()).fold(0.0, |err, (ni, oi)| {
                err + minkowski_distance(ni, oi, p)
            });

            clusters = new_clusters;

            if err < early_stop {
                if cfg!(test) {
                    println!("early stop with err {err}");
                }
                break;
            }

        }

        Self { k: k, p: p, clusters: clusters }
    }
}

impl Model<usize> for KMeansClustering {
    /// return the nearest cluster idx, note that it is NOT the classification prediction
    fn predict(&self, feature: &Vec<f32>) -> usize {
        self.clusters.iter().enumerate().fold((0, f32::MAX), |s, (i, center)| {
            let d = minkowski_distance(center, feature, self.p);
            if s.1 < d {
                s
            } else {
                (i, d)
            }
        }).0
    }
}

#[cfg(test)]
mod test {
    use crate::model::Model;

    use super::KMeansClustering;

    #[test]
    fn test_kmeans() {
        let datas = vec![
            vec![1.0, 3.0],
            vec![2.0, 3.0],
            vec![1.0, 2.0],
            vec![4.0, 0.0],
            vec![3.0, 0.0],
            vec![3.0, -1.0],
            vec![3.0, 0.5],
        ];
        let model = KMeansClustering::new(3, 100, Some(2), None, None, None, &datas);
        for item in datas.iter() {
            println!("label: {}", model.predict(item));
        }
    }
}