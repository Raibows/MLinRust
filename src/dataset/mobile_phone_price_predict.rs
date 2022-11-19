use std::collections::HashMap;


pub fn process_mobile_phone_price_dataset(data: String) -> (Vec<Vec<f32>>, Vec<usize>, Option<Vec<String>>) {
    // x = 20 features
    // y = usize price range
    // in test dataset, the first column is id and there is no golden label
    let lines: Vec<&str> = data.split("\n").collect();
    let mut lines = lines.iter();
    let mut features: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
    let mut labels: Vec<usize> = Vec::with_capacity(lines.len());
    let mut label_map: HashMap<String, usize> = HashMap::new();
    lines.next(); // skip the first row
    for line in lines {
        if line.len() == 0 {
            continue;
        }
        let mut temp: Vec<&str> = line.split(",").collect();
        let label = temp.pop().unwrap().to_string();
        let size = label_map.len();
        labels.push(
            *label_map.entry(label).or_insert(size)
        );
        features.push(
            temp.into_iter().map(|item| item.parse::<f32>().unwrap()).collect()
        );
    }

    let label_map: Vec<String> = label_map.into_iter().map(|(k, _)| k).collect();

    (features, labels, Some(label_map))
}