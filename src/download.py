import os
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
import yaml
import argparse

logging.basicConfig(level=logging.INFO)

def download_and_extract(url: str, save_dir: str, filename: Optional[str] = None) -> str:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    name = filename or Path(url).name
    file_path = save_path / name

    if not file_path.exists():
        logging.info(f"Downloading: {url}")
        r = requests.get(url)
        r.raise_for_status()
        file_path.write_bytes(r.content)

        if name.endswith(".zip"):
            with zipfile.ZipFile(file_path) as zf:
                zf.extractall(save_path)
            file_path.unlink()
        elif name.endswith((".tar.gz", ".tgz", ".gz")):
            with tarfile.open(file_path, "r:gz") as tf:
                tf.extractall(save_path)
            file_path.unlink()
        elif name.endswith(".tar"):
            with tarfile.open(file_path, "r") as tf:
                tf.extractall(save_path)
            file_path.unlink()
    else:
        logging.info(f"File already exists: {file_path}")

    return str(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["download"]["cifar_dir"], exist_ok=True)
    download_and_extract(cfg["download"]["cifar_url"], cfg["download"]["cifar_dir"])

if __name__ == "__main__":
    main()