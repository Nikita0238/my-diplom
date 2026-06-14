from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor_core import ActivityPredictor


app = FastAPI(title="Health Monitoring Activity API", version="1.0.0")
predictor = ActivityPredictor(artifacts_dir="./artifacts")


class PredictWindowRequest(BaseModel):
    window: list[list[float]] = Field(..., description="Window data with shape [WINDOW, N_FEATURES]")
    threshold: float = 0.60
    topk: int = 3


class PredictRecordsRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., description="Raw rows with feature keys f_00..f_22")
    threshold: float = 0.60
    topk: int = 3


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "window": predictor.window,
        "step": predictor.step,
        "n_features": predictor.n_features,
    }


@app.post("/predict/window")
def predict_window(payload: PredictWindowRequest) -> dict[str, Any]:
    try:
        window_data = pd.DataFrame(payload.window).values
        return predictor.predict_window(window_data, threshold=payload.threshold, topk=payload.topk)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/records")
def predict_records(payload: PredictRecordsRequest) -> dict[str, Any]:
    try:
        df = pd.DataFrame(payload.records)
        preds = predictor.predict_dataframe(df, threshold=payload.threshold, topk=payload.topk)
        return {"count": len(preds), "predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
