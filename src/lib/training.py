"""Utilidades de entrenamiento de la Etapa 2: hiperparámetros por defecto y dataset de PyTorch."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib import preprocessing as prep

# Hiperparámetros por defecto de cada modelo. Se pueden cambiar desde la notebook sin tocar el código:
#   classifier.hparams = {"epochs": 15}
DEFAULT_HPARAMS: dict[str, dict] = {
    "resnet18_finetuned": {
        "strategy": "fine_tuning",   # "fine_tuning" (toda la red) o "feature_extraction" (solo la cabeza)
        "epochs": 10,
        "batch_size": 64,
        "lr_head": 1e-3,             # learning rate de la capa nueva
        "lr_backbone": 1e-4,         # learning rate de las capas preentrenadas (más bajo: ya saben)
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "patience": 3,               # early stopping: épocas sin mejorar en valid antes de cortar
        "class_weights": False,
        "seed": 42,
        "num_workers": 2,
    },
    "cnn_custom": {
        "epochs": 40,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout": 0.3,
        "embedding_dim": 512,
        "patience": 6,
        "class_weights": False,
        "seed": 42,
        "num_workers": 2,
    },
}


def leer_splits(dataset_path: Path) -> pd.DataFrame:
    """Splits guardados por 1-pre-processing.ipynb (data/splits.csv), con la ruta absoluta de cada imagen."""
    splits = pd.read_csv(Path(dataset_path).parent / "splits.csv")
    splits["abs"] = splits["path"].map(lambda p: Path(dataset_path) / p)
    return splits


class BreedDataset(Dataset):
    """Imágenes + índice de raza, con el preprocesamiento compartido y augmentation opcional (solo train)."""

    def __init__(self, df: pd.DataFrame, classes: list[str], image_size: int, augment: bool = False) -> None:
        self.paths = df["abs"].tolist()
        indice = {c: i for i, c in enumerate(classes)}
        self.labels = [indice[b] for b in df["breed"]]
        self.image_size = image_size
        self.aug = prep.train_augmentation(image_size) if augment else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rgb = prep.to_rgb_resized(cv2.imread(str(self.paths[i])), self.image_size)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]
        return prep.to_tensor(rgb), self.labels[i]