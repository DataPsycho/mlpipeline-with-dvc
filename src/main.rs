use polars::{prelude::*, datatypes::DataType::Float64};
use std::path::Path;
use std::fs;


fn read_csv_into_df() -> Result<DataFrame, PolarsError>{
    let file_path = Path::new("data/raw/autompg.csv");
    let df = CsvReader::from_path(file_path).unwrap().infer_schema(None).has_header(true).finish();
    df
}

fn bucketize(val: &Series) -> Series{
    let result = val.i64().unwrap().into_iter().map(|yr| {
        yr.map(|value| {
            let value = if value < 73 {
                0
            } else if value >= 73 && value < 76 {
                1} else if value >=76 && value < 79 {
                    2} else if value >= 79 {
                        3} else {
                            999
                        };
            value
        })
    });
    result.collect::<Int64Chunked>().into_series()
}

fn standardize(val : &Series) -> Series {
    let val_mean = val.mean().unwrap();
    let casted_series = val.cast(&Float64).unwrap();
    let val_std = val.std_as_series(1).iter().nth(0).unwrap().try_extract::<f64>().unwrap();
    let result = casted_series.f64().unwrap().into_iter().map(|atom|{
        atom.map(|proton|{(proton - val_mean)/val_std as f64})
    });
    result.collect::<Float64Chunked>().into_series()
}

fn apply_standardization(df: DataFrame, col_list: Vec<&str>) -> DataFrame {
    let mut _df = df.clone();
    for item in col_list.iter(){
        // let std_result = standardize(df.column(*item).unwrap());
        let _df = _df.apply(item, standardize).unwrap();
    }
    _df
}

fn write_to_csv(path: String, mut df: DataFrame) {
    let mut file = std::fs::File::create(path).unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();
}

fn main() -> PolarsResult<()>{
    fs::create_dir_all("data/processed")?;
    let mut df = read_csv_into_df().unwrap();
    let df = df.apply("model year", bucketize)?;
    let df = df.drop("car name").unwrap();
    let df = df.drop("origin").unwrap();
    let df = df.drop_nulls(None)?;
    let mean_df = df.mean();
    let std_df = df.std(1);
    let df = apply_standardization(
        df, 
        vec!["displacement", "horsepower", "weight", "acceleration", "cylinders"]
    );
    println!("{:?}", df.tail(Some(3)));
    write_to_csv("data/processed/autompg.csv".to_string(), df);
    write_to_csv("data/processed/autompg_mean.csv".to_string(), mean_df);
    write_to_csv("data/processed/autompg_std.csv".to_string(), std_df);
    println!("Pipeline Executed Successfully!");
    Ok(())
}
