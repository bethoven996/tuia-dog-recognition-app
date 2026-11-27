from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from lib import preprocessing as prep
from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.models.cnn_custom import CNNCustom
from lib.training import DEFAULT_HPARAMS, BreedDataset, leer_splits

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 (o ResNet50) pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en 1-pre-processing.ipynb.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        name = self.active_model_name
        # Hiperparámetros: los de DEFAULT_HPARAMS, pisados por los que se definan en self.hparams
        hp = {**DEFAULT_HPARAMS[name], **getattr(self, "hparams", {})}
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
        torch.manual_seed(hp["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Splits de 1-pre-processing.ipynb. El orden de las clases (alfabético) se guarda con el modelo.
        splits = leer_splits(self.dataset_path)
        train_df = splits[splits["split"] == "train"]
        valid_df = splits[splits["split"] == "valid"]
        classes = sorted(train_df["breed"].unique())

        workers = hp["num_workers"] if device.type == "cuda" else 0
        train_dl = DataLoader(
            BreedDataset(train_df, classes, self.image_size, augment=True),  # augmentation solo en train
            batch_size=hp["batch_size"], shuffle=True, num_workers=workers,
            generator=torch.Generator().manual_seed(hp["seed"]),
        )
        valid_dl = DataLoader(BreedDataset(valid_df, classes, self.image_size),
                              batch_size=hp["batch_size"], num_workers=workers)

        model = self._build_model(name, len(classes), hp).to(device)
        model.classes = classes  # mapeo índice -> raza, guardado dentro del checkpoint

        pesos = None
        if hp["class_weights"]:
            conteos = train_df["breed"].value_counts().reindex(classes).to_numpy()
            pesos = torch.tensor(len(train_df) / (len(classes) * conteos), dtype=torch.float32, device=device)
        criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=hp["label_smoothing"])
        optimizer = self._build_optimizer(name, model, hp)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

        historial: list[dict[str, float]] = []
        mejor_acc, sin_mejora = -1.0, 0
        for epoca in range(1, hp["epochs"] + 1):
            model.train()
            suma_loss, aciertos, n = 0.0, 0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterio(logits, y)
                loss.backward()
                optimizer.step()
                suma_loss += loss.item() * len(y)
                aciertos += (logits.argmax(1) == y).sum().item()
                n += len(y)
            val_loss, val_acc = self._validar(model, valid_dl, criterio, device)
            scheduler.step()
            historial.append({"epoca": epoca, "train_loss": suma_loss / n, "train_acc": aciertos / n,
                              "val_loss": val_loss, "val_acc": val_acc})
            logger.info("[%s] época %d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f",
                        name, epoca, hp["epochs"], suma_loss / n, aciertos / n, val_loss, val_acc)

            # Se guarda el mejor checkpoint según valid (no el de la última época)
            if val_acc > mejor_acc:
                mejor_acc, sin_mejora = val_acc, 0
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(copy.deepcopy(model).cpu(), self.active_checkpoint)
            else:
                sin_mejora += 1
                if sin_mejora >= hp["patience"]:
                    logger.info("[%s] early stopping en la época %d", name, epoca)
                    break

        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / f"history_{name}.json").write_text(
            json.dumps({"hparams": hp, "historial": historial, "mejor_val_acc": mejor_acc}, indent=2),
            encoding="utf-8",
        )
        self._loaded.pop(name, None)  # la próxima carga lee el mejor checkpoint recién guardado

    def evaluate_classifier(self, split: str = "test") -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en 2-train.ipynb.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}

        `split` permite evaluar tambien sobre el conjunto independiente (por defecto: test).
        Las predicciones quedan en self.last_predictions para la matriz de confusion y el
        analisis de errores de la notebook.
        """
        model = self.load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        classes = list(model.classes)

        df = leer_splits(self.dataset_path)
        df = df[df["split"] == split].reset_index(drop=True)
        dl = DataLoader(BreedDataset(df, classes, self.image_size), batch_size=64)
        probs = []
        with torch.no_grad():
            for x, _ in dl:
                probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        y_true = np.array([classes.index(b) for b in df["breed"]])
        y_pred = probs.argmax(1)

        # Métricas por clase (one-vs-rest) y promedio macro
        prec, rec, f1s, espec = [], [], [], []
        for c in range(len(classes)):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            tn = len(y_true) - tp - fp - fn
            p, r, f = precision_recall_f1(tp, fp, fn)
            prec.append(p)
            rec.append(r)
            f1s.append(f)
            espec.append(specificity(tn, fp))

        self.last_predictions = df.assign(
            pred=[classes[i] for i in y_pred], prob=probs.max(1), acierto=y_pred == y_true
        )
        return {
            "accuracy": float((y_pred == y_true).mean()),
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
            "specificity": float(np.mean(espec)),
            "f1": float(np.mean(f1s)),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        model = self.load_model()
        model.eval()
        device = next(model.parameters()).device
        x = prep.to_tensor(prep.to_rgb_resized(image, self.image_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "embed"):  # CNN propia: capa de embedding antes del clasificador
                emb = model.embed(x)
            else:  # ResNet: todo menos la capa final (fc)
                emb = torch.flatten(nn.Sequential(*list(model.children())[:-1])(x), 1)
        emb = emb[0].cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb.astype(float).tolist()

    # ------------------------------------------------------------------
    # Auxiliares privados de la Etapa 2
    # ------------------------------------------------------------------

    def _build_model(self, name: str, num_classes: int, hp: dict) -> nn.Module:
        if name == "cnn_custom":
            return CNNCustom(num_classes, embedding_dim=hp["embedding_dim"], dropout=hp["dropout"])
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # una salida por raza
        if hp["strategy"] == "feature_extraction":
            for nombre, p in model.named_parameters():
                p.requires_grad = nombre.startswith("fc.")  # backbone congelado: solo se entrena la cabeza
        return model

    def _build_optimizer(self, name: str, model: nn.Module, hp: dict) -> torch.optim.Optimizer:
        if name == "cnn_custom":
            return torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        cabeza = list(model.fc.parameters())
        if hp["strategy"] == "feature_extraction":
            return torch.optim.AdamW(cabeza, lr=hp["lr_head"], weight_decay=hp["weight_decay"])
        backbone = [p for nombre, p in model.named_parameters() if not nombre.startswith("fc.")]
        # Learning rates diferenciados: la cabeza nueva aprende rápido, el backbone preentrenado despacio
        return torch.optim.AdamW(
            [{"params": backbone, "lr": hp["lr_backbone"]}, {"params": cabeza, "lr": hp["lr_head"]}],
            weight_decay=hp["weight_decay"],
        )

    @staticmethod
    def _validar(model: nn.Module, dl: DataLoader, criterio: nn.Module, device: torch.device) -> tuple[float, float]:
        model.eval()
        suma_loss, aciertos, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                suma_loss += criterio(logits, y).item() * len(y)
                aciertos += (logits.argmax(1) == y).sum().item()
                n += len(y)
        return suma_loss / n, aciertos / n