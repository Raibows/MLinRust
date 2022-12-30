use mlinrust::model::kmeans::KMeansClustering;
use mlinrust::model::Model;

fn main() {
    // clustering is useful! See this case:
    // we can often use unsupervised k-means algorithm to cluster data first
    // then we can build multiple classifiers or models to fit different clusters
    // that should be better than the single model

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
        println!("belong to {}th cluster", model.predict(item));
    }
}