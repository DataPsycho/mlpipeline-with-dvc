# ML Pipeline Decoupled: Write Framework Agonstic ML Pipeline with DVC, RUST and Python

The feature creating and data cleaning part is written in rust and code can be found in src/main.rs file. This part is using Polars dataframe to clean, filter and featurize the data.

The training code can be found in pysrc/training.py file. Here featured data from the first step is read into pandas dataframe and then trained using a RandomForest regression model from scikitlearn.

The preprocessing and training is orchrastrated using DVC. The main ML pipeline can be found in `dvc.yaml` file as follows:
```yaml
stages:
  build_bin:
    cmd: cargo build --release
    deps:
      - src/main.rs
      - Cargo.toml
  preprocess:
    cmd: ./target/release/torch-mlops
    deps:
      - data/raw/autompg.csv
      - ./target/release/torch-mlops
    outs:
      - data/processed
  train:
    cmd: python pysrc/train.py data/processed model.pkl
    deps:
      - data/processed
      - pysrc/train.py
    params:
      - train.min_split
      - train.n_est
      - train.seed
    outs:
      - models
```

Lets explain the steps:
- Build Bin:
  - any change in `main.rs` or `cargo.toml` will trigger the build of the binary file other wise it will skip that step

- Preprocess:
  - Data processing will execute the rust binary file and save the outputs into to the `data/processed` directory
  - Any change in the csv or binary file will reproduce that step

- Train:
  - Data generated from previous step will be used as dependency in the train step
  - But it also depends on the parameters yaml file and the `train.py`, so any changes in those files will reproduce this step
  - The model will be saved in the models directory

You can add more steps like validation, testing etc. See more details on DVC documentation. It is just a poc.

Useful commands:
- `dvc repro` to reproduce any step
- `dvc repro --force` to reproduce all step by force, in that case all steps will be run no matter if there is any change or not detected


*For the project I have installed DVC as binary globally in my machine not with PIP, so that it can be used in other projects. For production machine I would install DVC as binary or choose any Prebuild image provided by IterativeIO.*