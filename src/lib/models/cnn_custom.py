"""Modelo B: CNN propia, entrenada desde cero (sin preentrenamiento).

Está en un módulo (y no en la notebook) porque el backend carga el checkpoint con torch.load:
la clase tiene que poder importarse con el mismo módulo y nombre que tenía al guardarse.
"""
import torch
from torch import nn


def _bloque(c_in: int, c_out: int) -> nn.Sequential:
    """Dos convoluciones 3x3 con BatchNorm y ReLU, y un max pooling que reduce la imagen a la mitad."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNCustom(nn.Module):
    """5 bloques convolucionales (224 -> 7), global average pooling, capa de embedding y clasificador."""

    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _bloque(3, 32), _bloque(32, 64), _bloque(64, 128), _bloque(128, 256), _bloque(256, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Capa de embedding de dimensión EMBEDDING_DIM, para usarla en la búsqueda por similitud
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Vector de la capa previa al clasificador."""
        return self.embedding(self.pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.embed(x)))