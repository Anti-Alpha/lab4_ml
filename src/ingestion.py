import os, pickle, logging, yaml
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

def train_test_split(data: pd.DataFrame, test_size: Union[float, int] = 0.25, random_state: Union[int, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if random_state is not None: np.random.seed(random_state)
    n = len(data)
    t = int(n * test_size) if isinstance(test_size, float) else test_size
    idx = np.random.permutation(n)
    return data.iloc[idx[t:]], data.iloc[idx[:t]]

def unpickle(file): return pickle.load(open(file, "rb"), encoding="bytes")

def assign_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy(); df["batch_name"] = "not_set"
    n = cfg["prepare"]["n_batches"]; b = len(df) // n
    i = 0
    for j in range(n):
        end = None if j == n - 1 else i + b
        df.iloc[i:end, df.columns.get_loc("batch_name")] = str(j)
        i += b
    return df

def select_batches(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    return df[df["batch_name"].isin(cfg["prepare"]["batch_names_select"])].copy()

def process_data(data_dir: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b_paths = [f"{data_dir}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_path = f"{data_dir}/cifar-10-batches-py/test_batch"
    data, labels = [], []

    for p in b_paths:
        d = unpickle(p)
        data.append(d[b"data"])
        labels += d[b"labels"]

    x_train = np.vstack(data).reshape(-1, 3, 32, 32).astype("float32") / 255
    y_train = np.array(labels)
    x_test = unpickle(test_path)
    x_test_data = x_test[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = np.array(x_test[b"labels"])

    df_train = pd.DataFrame({"image": list(x_train), "label": y_train})
    df_test = pd.DataFrame({"image": list(x_test_data), "label": y_test})

    df_train = assign_batches(df_train, cfg)
    logging.info(f"Batches: {cfg['prepare']['n_batches']}")
    df_train = select_batches(df_train, cfg)
    logging.info(f"Selected: {cfg['prepare']['batch_names_select']}")

    df_train, df_val = train_test_split(df_train, test_size=cfg["prepare"].get("val_size", 0.2), random_state=cfg["prepare"].get("random_state", 42))
    logging.info(f"Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test

def main():
    with open("params.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(cfg["prepare"]["train_split_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["prepare"]["val_split_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["prepare"]["test_split_path"]), exist_ok=True)

    df_train, df_val, df_test = process_data(cfg["download"]["cifar_dir"], cfg)
    for path, df in [
        (cfg["prepare"]["train_split_path"], df_train),
        (cfg["prepare"]["val_split_path"], df_val),
        (cfg["prepare"]["test_split_path"], df_test),
    ]:
        logging.info(f"Saving to: {path}")
        df.to_pickle(path, compression="zip")

if __name__ == "__main__":
    main()