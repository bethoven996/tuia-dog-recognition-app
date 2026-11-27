"""Preprocesamiento compartido entre las notebooks y los servicios de la app."""
import albumentations as A
import cv2
import numpy as np
import torch

# Estadísticas de ImageNet (RGB, escala 0-1), las que esperan los modelos preentrenados
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_rgb_resized(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 (como lo entrega OpenCV) -> RGB uint8 de size x size."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 -> tensor float32 3 x H x W normalizado con ImageNet."""
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def train_augmentation(size: int) -> A.Compose:
    """Augmentation SOLO para train. Se aplica sobre RGB uint8 de size x size, antes de to_tensor."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),                                              # un perro espejado sigue siendo de la misma raza
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),        # fotos levemente inclinadas
        A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0),
                            ratio=(0.75, 1.33), p=0.5),                       # simula recortes de YOLO
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),                             # blur leve: no borrar la textura del pelo
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),                          # ruido de cámara
        A.ToGray(p=0.05),                                                     # pocas fotos en grises: no depender solo del color
    ])