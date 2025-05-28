import os, yaml, logging
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import mlflow
from mlflow.pytorch import load_model
from loader import create_data_loader
from metrics import evaluate_metrics

logging.basicConfig(level=logging.INFO, force=True)

def test_model(model, test_loader, loss_function, device):
    model.eval()
    loss, acc, prec, rec, f1 = evaluate_metrics(model, test_loader, loss_function, device)
    mlflow.log_metrics({
        "test_loss": loss,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_F1": f1
    })
    logging.info(f"Test: loss={loss:.4f}, acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    return loss, acc, prec, rec, f1

def get_mlflow_run_id(config):
    path = config["mlflow"]["run_id_path"]
    if os.path.exists(path):
        with open(path) as f:
            run_id = f.read().strip()
        logging.info(f"Run ID loaded: {run_id}")
        return run_id
    logging.error(f"No run_id file at {path}")
    return None

def get_mlflow_best_model(run_id, config):
    uri = f"runs:/{run_id}/{config['mlflow']['best_model_artifact_name']}"
    return load_model(uri)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open("params.yaml"))

    run_id = get_mlflow_run_id(config)
    if not run_id:
        return

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"].get("experiment_name"))

    with mlflow.start_run(run_id=run_id):
        logging.info(f"Testing on run: {run_id}")
        model = get_mlflow_best_model(run_id, config)
        if model is None:
            logging.error("Model not loaded from MLflow")
            return

        test_df = pd.read_pickle(config["prepare"]["val_split_path"], compression="zip")
        loader = create_data_loader(test_df, config)
        loss_fn = nn.CrossEntropyLoss()

        test_model(model, loader, loss_fn, device)

if __name__ == "__main__":
    main()