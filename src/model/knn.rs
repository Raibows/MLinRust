
/// KNN classifier implemented by [KDTree](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
pub struct KNearestNeighbor {
    k: usize,
}


impl KNearestNeighbor {
    pub fn new(k: usize) -> Self {
        Self { k: k }
    }
}

#[cfg(test)]
mod test {
    use super::KNearestNeighbor;

    #[test]
    fn test_knn() {
        let model = KNearestNeighbor::new(10);
    }
}