import pandas as pd
import os
import pickle
import sys
import yaml
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from collections import namedtuple

DataRepo = namedtuple("DataRepo", "features, targets")
PARAMS = yaml.safe_load(open("params.yaml"))["train"]
MODEL_REPO = Path("models")


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py features model\n")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]
seed = PARAMS["seed"]
n_est = PARAMS["n_est"]
min_split = PARAMS["min_split"]


def read_data() -> DataRepo:
    feature_cols = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
    ]

    df = pd.read_csv(Path(input).joinpath("autompg.csv"))
    features = df[feature_cols].to_numpy()
    targets = df["mpg"].to_numpy()
    return DataRepo(features=features, targets=targets)


def train_clf(data: DataRepo) -> RandomForestRegressor:
    clf = RandomForestRegressor(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
    )

    clf.fit(data.features, data.targets)
    return clf



def save_model(model: RandomForestRegressor):
    write_path = MODEL_REPO.joinpath(output)
    with open(write_path, "wb") as fd:
        pickle.dump(model, fd)
        sys.stderr.write(f"Model written successfully at {write_path}")


def main():
    data_repo = read_data()
    sys.stderr.write("X matrix size {}\n".format(data_repo.features.shape))
    sys.stderr.write("Y matrix size {}\n".format(data_repo.targets.shape))
    model = train_clf(data=data_repo)
    MODEL_REPO.mkdir(parents=True, exist_ok=True)
    save_model(model=model)

if __name__ == "__main__":
    main()

