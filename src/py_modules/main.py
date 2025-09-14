# Copyright 2025 MODEL_X

import argparse
import logging
from pathlib import Path
from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def download_and_save_dataset(
    name: str, folder: Path = Path("./datasets")
) -> DatasetDict | None:
    """
    Download a Hugging Face dataset and save locally, avoiding re-download.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name.replace("/", "_")

    if path.exists() and any(path.iterdir()):
        logging.info(f"Loading existing dataset '{name}' from {path}")
        return DatasetDict.load_from_disk(str(path))

    try:
        logging.info(f"Downloading dataset '{name}'...")
        ds = load_dataset(name)
        ds.save_to_disk(str(path))
        logging.info(f"Dataset saved to {path}")
        return ds
    except DatasetNotFoundError:
        logging.error(f"Dataset '{name}' not found.")
    except Exception as e:
        logging.error(f"Error: {e}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save a Hugging Face dataset locally."
    )
    parser.add_argument("--dataset_name", type=str, default="Trelis/tiny-shakespeare")
    parser.add_argument("--folder_name", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    download_and_save_dataset(args.dataset_name, args.folder_name)
