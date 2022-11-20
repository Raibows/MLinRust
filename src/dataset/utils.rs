use std::vec;



pub enum ImputeType {
    Mean,
    Zero,
    Value(f32),
}

pub fn impute_lost_values(res: Vec<Vec<Result<f32, Box<dyn std::error::Error>>>>, filled: ImputeType) -> Vec<Vec<f32>> {
    match filled {
        ImputeType::Mean => {
            let counter = res.iter().fold(vec![(0.0f32, 0.0f32); res[0].len()], |mut fold, item| {
                item.iter().enumerate().for_each(|(i, item)| {
                    if let Ok(v) = item {
                        fold[i].0 += 1.0;
                        fold[i].1 += v;
                    }
                });
                fold
            });
            let counter: Vec<f32> = counter.into_iter().map(|(num, sum)| sum / num).collect();
            res.into_iter().map(|item| {
                item.into_iter().enumerate().map(|(i, e)| {
                    if let Ok(v) = e {
                        v
                    } else {
                        counter[i]
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
        ImputeType::Zero => {
            res.into_iter().map(|item| {
                item.into_iter().map(|e| {
                    if let Ok(v) = e {
                        v
                    } else {
                        0.0
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
        ImputeType::Value(value) => {
            res.into_iter().map(|item| {
                item.into_iter().map(|e| {
                    if let Ok(v) = e {
                        v
                    } else {
                        value
                    }
                }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
        },
    }
}
