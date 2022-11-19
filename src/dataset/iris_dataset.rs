use std::collections::HashMap;


pub fn process_iris_dataset(data: String) -> (Vec<Vec<f32>>, Vec<usize>, Option<Vec<String>>) {
    let lines: Vec<&str> = data.split("\n").collect();
    let mut lines = lines.iter();
    let mut features: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
    let mut labels: Vec<usize> = Vec::with_capacity(lines.len());
    let mut label_map: HashMap<String, usize> = HashMap::new();
    lines.next();
    for line in lines {
        // println!("{i}\t\t🚗{:?}🚗", l);

        if line.len() == 0 {
            continue;
        }
        
        let temp: Vec<&str> = line.split(",").collect();
        let label = temp.last().unwrap().to_string();
        let size = label_map.len();
        labels.push(
            *label_map.entry(label).or_insert(size)
        );
        features.push(
            temp.iter().take(4).map(|item| item.parse::<f32>().unwrap()).collect::<Vec<f32>>()
        )
    }
    let label_map: Vec<String> = label_map.into_iter().map(|(k, _)| k).collect();
    if cfg!(test) {
        println!("{:?}", labels);
    }
    (features, labels, Some(label_map))
}