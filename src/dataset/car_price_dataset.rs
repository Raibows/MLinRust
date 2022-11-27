use std::collections::HashMap;

use super::utils::{ImputeType, impute_lost_values};

pub fn process_tianchi_car_price_regression_dataset(data: String, fill_missing_value_by: ImputeType) -> (Vec<Vec<f32>>, Vec<f32>, Option<HashMap<usize, String>>) {
    let lines: Vec<&str> = data.split("\n").collect();
    let mut lines = lines.into_iter();
    let header = lines.next().unwrap().split(" ").enumerate().find(|(_, item)| *item == "price").unwrap().0;

    let mut features = Vec::with_capacity(lines.len());
    let mut labels = Vec::with_capacity(lines.len());

    for line in lines {
        if line.len() == 0 {
            continue;
        }
        let mut feature: Vec<Result<f32, Box<dyn std::error::Error>>> = line.split(" ").map(|item| item.parse::<f32>().map_err(|e| e.into())).collect();
        let label = feature.remove(header).unwrap();
        features.push(feature);
        labels.push(label);
    }
    
    (impute_lost_values(features, fill_missing_value_by), labels, None)
}


#[cfg(test)]
mod test {
    use crate::dataset::{Dataset, FromPathDataset};
    use crate::dataset::utils::{impute_lost_values, ImputeType};
    use super::process_tianchi_car_price_regression_dataset;

    #[test]
    fn test_impute_value() {
        let line = "134890 734 20160002 13.0 9  0.0 1.0 0 15.0  725 1 0 20160316 520 60.559197618107895 16.66714133921907 1.4173299483986161 1.6520004122882843 -11.864145362401656 -0.45671072151414377 9.083770100847348 0.5666899569182301 0.0 1.12438620388016 0.0 0.05428270535363877 0.05349221415727359 0.1243141400920556 0.09213871303802613 0.0 18.76383196596702 -1.512063144207823 -1.0087179111456093 -12.100623394882023 -0.9470519461996554 9.077297192449008 0.5812143061207315 3.9459233801110463";

        let line1 = "134890 734 20160002 13.0 9 9999.9 0.0 1.0 0 15.0  725 1 0 20160316 520 60.559197618107895 16.66714133921907 1.4173299483986161 1.6520004122882843 -11.864145362401656 -0.45671072151414377 9.083770100847348 0.5666899569182301 0.0 1.12438620388016 0.0 0.05428270535363877 0.05349221415727359 0.1243141400920556 0.09213871303802613 0.0 18.76383196596702 -1.512063144207823 -1.0087179111456093 -12.100623394882023 -0.9470519461996554 9.077297192449008 0.5812143061207315 3.9459233801110463";
        // let line = "1.0  2.0";

        // let t: Vec<(usize, Result<f32, _>)> = line.split(" ").enumerate().map(|(i, item)| (i, item.parse())).collect();
        let t: Vec<Result<f32, Box<dyn std::error::Error>>> = line.split(" ").enumerate().map(|(_, item)| item.parse::<f32>().map_err(|e| e.into()))
        .collect();

        let t1: Vec<Result<f32, Box<dyn std::error::Error>>> = line1.split(" ").enumerate().map(|(_, item)| item.parse::<f32>().map_err(|e| e.into()))
        .collect();

        println!("{:?} {}", t, t.len());

        let t = impute_lost_values(vec![t, t1], ImputeType::Value(f32::MAX));
        println!("{:?} {}", t, t.len());
        
    }

    #[test]
    fn test_process() {
        let data = Dataset::<f32>::read_data_from_file(".data/TianchiCarPriceRegression/train_5w.csv").unwrap();
        let (feature, _label, label_map) = process_tianchi_car_price_regression_dataset(data, ImputeType::Mean);
        assert_eq!(feature.len(), 50_000);
        assert_eq!(feature[0].len(), 39);
        assert_eq!(label_map, None);
    }
}