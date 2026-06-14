from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


class ActivityPredictor:
    def __init__(self, artifacts_dir: str = "./artifacts"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artifacts = Path(artifacts_dir)

        self.mapping = joblib.load(artifacts / "label_mapping.pkl")
        self.scaler = joblib.load(artifacts / "feature_scaler.pkl")

        self.window = int(self.mapping["window"])
        self.step = int(self.mapping["step"])
        self.n_features = int(self.mapping["n_features"])
        self.feature_cols = self.mapping["feature_cols"]
        self.idx_to_label = self.mapping["idx_to_label"]
        self.idx_to_name_ru = self.mapping["idx_to_name_ru"]

        n_classes = len(self.mapping["classes"])
        self.model = CNNLSTM(n_features=self.n_features, n_classes=n_classes).to(self.device)
        self.model.load_state_dict(
            torch.load(artifacts / "activity_cnn_lstm_state_dict.pt", map_location=self.device)
        )
        self.model.eval()

    def _predict_proba_window(self, window_data: np.ndarray) -> np.ndarray:
        if window_data.shape != (self.window, self.n_features):
            raise ValueError(
                f"Expected shape {(self.window, self.n_features)}, got {window_data.shape}"
            )

        w_scaled = self.scaler.transform(window_data).reshape(1, self.window, self.n_features)
        x = torch.tensor(np.transpose(w_scaled, (0, 2, 1)), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).detach().cpu().numpy()[0]
        return probs

    def predict_window(self, window_data: np.ndarray, threshold: float = 0.60, topk: int = 3) -> dict[str, Any]:
        probs = self._predict_proba_window(window_data)
        order = np.argsort(probs)[::-1]
        best_idx = int(order[0])
        best_prob = float(probs[best_idx])

        top = [
            {
                "rank": rank + 1,
                "class_idx": int(i),
                "label": int(self.idx_to_label[int(i)]),
                "name_ru": self.idx_to_name_ru[int(i)],
                "probability": float(probs[int(i)]),
            }
            for rank, i in enumerate(order[:topk])
        ]

        return {
            "class_idx": best_idx,
            "label": int(self.idx_to_label[best_idx]),
            "name_ru": self.idx_to_name_ru[best_idx],
            "confidence": best_prob,
            "status": "CONFIDENT" if best_prob >= threshold else "LOW_CONFIDENCE",
            "topk": top,
        }

    def predict_dataframe(self, df: pd.DataFrame, threshold: float = 0.60, topk: int = 3) -> list[dict[str, Any]]:
        if any(c not in df.columns for c in self.feature_cols):
            missing = [c for c in self.feature_cols if c not in df.columns]
            raise ValueError(f"Missing required columns: {missing[:10]}")

        arr = df[self.feature_cols].values
        if len(arr) < self.window:
            raise ValueError(f"Need at least {self.window} rows, got {len(arr)}")

        out = []
        for start in range(0, len(arr) - self.window + 1, self.step):
            window_data = arr[start:start + self.window]
            pred = self.predict_window(window_data, threshold=threshold, topk=topk)
            pred["start_idx"] = int(start)
            pred["end_idx"] = int(start + self.window - 1)
            out.append(pred)
        return out
