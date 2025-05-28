import os, yaml, logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.pytorch import log_model
from loader import create_data_loader
from metrics import evaluate_metrics
from model import EfficientNetV2

logging.basicConfig(level=logging.INFO, force=True)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, save_path=Path("best_model.pth")):
    model.to(device)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}")

        val_loss, acc, prec, rec, f1 = evaluate_metrics(model, val_loader, loss_fn, device)
        mlflow.log_metrics({
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "F1": f1
        }, step=epoch + 1)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logging.info(f"New best model saved (val_loss={best_loss:.4f})")

    return save_path

def save_run_id(run_id, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(run_id)

def main():
    config = yaml.safe_load(open("params.yaml"))
    os.makedirs(os.path.dirname(config["train"]["model_save_path"]), exist_ok=True)
    torch.manual_seed(config["prepare"]["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"].get("experiment_name"))

    run_name = f"LR={config['train']['lr']}_Epochs={config['train']['num_epochs']}"
    with mlflow.start_run(run_name=run_name) as run:
        save_run_id(run.info.run_id, config["mlflow"]["run_id_path"])
        mlflow.log_params({**config["train"], **config["prepare"], **config["loader"]})

        train_df = pd.read_pickle(config["prepare"]["train_split_path"], compression="zip")
        val_df = pd.read_pickle(config["prepare"]["val_split_path"], compression="zip")
        train_loader = create_data_loader(train_df, config)
        val_loader = create_data_loader(val_df, config)

        model = EfficientNetV2(n_classes=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"])

        logging.info("Training started")
        best_model_path = train_model(model, train_loader, val_loader, loss_fn, optimizer, config["train"]["num_epochs"], device, Path(config["train"]["model_save_path"]))

        best_model = EfficientNetV2(n_classes=10).to(device)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.eval()

        try:
            log_model(
                pytorch_model=best_model,
                artifact_path=config["mlflow"]["best_model_artifact_name"],
                signature=best_model.mlflow_signature(),
            )
            logging.info("Model saved to MLflow")
        except Exception as e:
            logging.error(f"MLflow model log error: {e}")

if __name__ == "__main__":
    main()