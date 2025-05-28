# lab4_ml

Image classification using a neural network (CIFAR-10). Includes download, preprocessing, training, evaluation, and MLflow tracking.

## Setup

```bash
cd lab4_ml
curl -sSL https://install.python-poetry.org | python -
poetry install
```

## Run

```bash
mlflow ui  # optional
dvc repro  # full pipeline
```

Or manually:

```bash
python src/download.py
python src/ingest.py
python src/train_model.py
python src/test_model.py
```

## Quick example

```python
import torch, yaml
import src.download as download, src.ingestion as ingest
import src.loader as loader, src.model as mdl
import src.train_model as train, src.test_model as test

with open("params.yaml") as f:
    cfg = yaml.safe_load(f)

data_dir = download.download_and_extract(cfg["download"]["cifar_url"], cfg["download"]["cifar_dir"])
train_df, val_df, test_df = ingest.process_data(data_dir, cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mdl.EfficientNetV2(n_classes=10).to(device)

train_loader = loader.create_data_loader(train_df, cfg)
val_loader = loader.create_data_loader(val_df, cfg)
test_loader = loader.create_data_loader(test_df, cfg)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"])

train.train_model(model, train_loader, val_loader, loss, opt, cfg["train"]["num_epochs"], device)
test.test_model(model, test_loader, loss, device)
```
