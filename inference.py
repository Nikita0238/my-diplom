import argparse
import json
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(description="Window-based activity inference from CSV")
    parser.add_argument("--csv", required=True, help="Path to input CSV with feature columns")
    parser.add_argument("--artifacts", default="./artifacts", help="Path to artifacts directory")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    parser.add_argument("--topk", type=int, default=3, help="Top-K classes to print")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts = Path(args.artifacts)
    mapping = joblib.load(artifacts / "label_mapping.pkl")
    scaler = joblib.load(artifacts / "feature_scaler.pkl")

    n_features = int(mapping["n_features"])
    n_classes = len(mapping["classes"])
    window = int(mapping["window"])
    step = int(mapping["step"])
    feature_cols = mapping["feature_cols"]
    idx_to_name_ru = mapping["idx_to_name_ru"]
    idx_to_label = mapping["idx_to_label"]

    model = CNNLSTM(n_features=n_features, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(artifacts / "activity_cnn_lstm_state_dict.pt", map_location=device))
    model.eval()

    df = pd.read_csv(args.csv)
    if any(c not in df.columns for c in feature_cols):
        missing = [c for c in feature_cols if c not in df.columns]
        raise ValueError(f"Missing required columns in CSV: {missing[:10]}")

    arr = df[feature_cols].values
    if len(arr) < window:
        raise ValueError(f"Need at least {window} rows, got {len(arr)}")

    results = []
    for start in range(0, len(arr) - window + 1, step):
        w = arr[start:start + window]
        w_scaled = scaler.transform(w).reshape(1, window, n_features)
        x = torch.tensor(np.transpose(w_scaled, (0, 2, 1)), dtype=torch.float32, device=device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()[0]

        order = np.argsort(probs)[::-1][: args.topk]
        best_idx = int(order[0])
        best_prob = float(probs[best_idx])
        status = "CONFIDENT" if best_prob >= args.threshold else "LOW_CONFIDENCE"

        topk = [
            {
                "rank": rank + 1,
                "class_idx": int(i),
                "label": int(idx_to_label[int(i)]),
                "name_ru": idx_to_name_ru[int(i)],
                "probability": float(probs[int(i)]),
            }
            for rank, i in enumerate(order)
        ]

        results.append(
            {
                "start_idx": int(start),
                "end_idx": int(start + window - 1),
                "pred_class_idx": best_idx,
                "pred_label": int(idx_to_label[best_idx]),
                "pred_name_ru": idx_to_name_ru[best_idx],
                "confidence": best_prob,
                "status": status,
                "topk_json": json.dumps(topk, ensure_ascii=False),
            }
        )

    out_path = artifacts / "inference_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
