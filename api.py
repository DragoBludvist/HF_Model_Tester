"""
api.py — SOC Alert Classification API.

Full pipeline in one endpoint:
    Raw alert → Standardize → SecureBERT classify → LangChain analyze → Response

Endpoints:
    POST /classify     — classify a single alert
    POST /classify/batch — classify multiple alerts
    POST /analyze      — classify + LangChain analysis
    POST /ask          — analyst follow-up question on a classified alert
    GET  /health       — service health check

Usage:
    pip install fastapi uvicorn
    python api.py                     # starts on port 8000
    python api.py --port 9000         # custom port

Requirements:
    pip install fastapi uvicorn torch transformers peft
    pip install langchain langchain-anthropic  (optional, for LangChain path)
"""

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from standardize import route_row, handle_wazuh, SKIP_SOURCETYPES
from langchain_analyst import analyze_alert

# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_ID = "cisco-ai/SecureBERT2.0-base"
ADAPTER_PATH = "models/secureBERT_botsv3_lora"
MAX_LEN = 256

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SOC Alert Classifier",
    description="SecureBERT + LangChain alert classification and analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

model = None
tokenizer = None
device = None


def load_model():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, problem_type="single_label_classification"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(device)
    model.eval()
    print(f"Model loaded from {ADAPTER_PATH}")


def classify_text(text):
    """Classify a single standardized text. Returns (prediction, confidence, threat_prob)."""
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt"
    ).to(device, non_blocking=True)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    prediction = "threat" if pred_idx == 1 else "benign"
    confidence = float(probs[pred_idx])
    threat_prob = float(probs[1])

    return prediction, confidence, threat_prob


# ── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class AlertInput(BaseModel):
    """A single alert — can be raw Wazuh JSON, Splunk CSV row, or pre-standardized text."""
    # Pre-standardized text (if already processed)
    text: Optional[str] = None
    # Raw Wazuh JSON alert
    wazuh_alert: Optional[dict] = None
    # Raw Splunk CSV row fields
    splunk_row: Optional[dict] = None

class BatchInput(BaseModel):
    alerts: list[AlertInput]

class AnalyzeInput(BaseModel):
    text: Optional[str] = None
    wazuh_alert: Optional[dict] = None
    splunk_row: Optional[dict] = None
    api_key: Optional[str] = None

class AskInput(BaseModel):
    alert_text: str
    prediction: str
    confidence: float
    question: str
    api_key: Optional[str] = None

class ClassifyResponse(BaseModel):
    standardized_text: str
    prediction: str
    confidence: float
    threat_probability: float
    category: str
    inference_ms: float

class AnalyzeResponse(BaseModel):
    standardized_text: str
    prediction: str
    confidence: float
    threat_probability: float
    category: str
    analysis: str
    inference_ms: float


# ── HELPER ────────────────────────────────────────────────────────────────────

def standardize_input(alert: AlertInput):
    """Convert any input format to standardized text."""
    # Already standardized
    if alert.text:
        category = "unknown"
        if alert.text.startswith("["):
            category = alert.text.split("]")[0].strip("[")
        return alert.text, category

    # Wazuh JSON
    if alert.wazuh_alert:
        text, category, level = handle_wazuh(alert.wazuh_alert)
        if text is None:
            raise HTTPException(status_code=422, detail=f"Alert skipped: level or category filtered")
        return text, category

    # Splunk row
    if alert.splunk_row:
        stype = alert.splunk_row.get("sourcetype", "")
        if stype in SKIP_SOURCETYPES:
            raise HTTPException(status_code=422, detail=f"Non-security sourcetype: {stype}")
        text = route_row(alert.splunk_row)
        if text is None:
            raise HTTPException(status_code=422, detail="Could not standardize alert")
        category = text.split("]")[0].strip("[") if "]" in text else "unknown"
        return text, category

    raise HTTPException(status_code=400, detail="Provide text, wazuh_alert, or splunk_row")


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "adapter": ADAPTER_PATH,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device and device.type == "cuda" else "cpu",
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(alert: AlertInput):
    """Classify a single alert. Accepts raw or pre-standardized input."""
    text, category = standardize_input(alert)

    t0 = time.time()
    prediction, confidence, threat_prob = classify_text(text)
    ms = (time.time() - t0) * 1000

    return ClassifyResponse(
        standardized_text=text,
        prediction=prediction,
        confidence=confidence,
        threat_probability=threat_prob,
        category=category,
        inference_ms=round(ms, 2),
    )


@app.post("/classify/batch")
async def classify_batch(batch: BatchInput):
    """Classify multiple alerts."""
    results = []
    t0 = time.time()

    for alert in batch.alerts:
        try:
            text, category = standardize_input(alert)
            prediction, confidence, threat_prob = classify_text(text)
            results.append({
                "standardized_text": text,
                "prediction": prediction,
                "confidence": confidence,
                "threat_probability": threat_prob,
                "category": category,
            })
        except HTTPException as e:
            results.append({"skipped": True, "reason": e.detail})

    total_ms = (time.time() - t0) * 1000
    threats = sum(1 for r in results if r.get("prediction") == "threat")

    return {
        "total": len(results),
        "threats": threats,
        "benign": len(results) - threats,
        "total_ms": round(total_ms, 2),
        "avg_ms": round(total_ms / len(results), 2) if results else 0,
        "results": results,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(alert: AnalyzeInput):
    """Classify + LangChain analysis. Returns full analyst report."""
    input_alert = AlertInput(
        text=alert.text,
        wazuh_alert=alert.wazuh_alert,
        splunk_row=alert.splunk_row,
    )
    text, category = standardize_input(input_alert)

    t0 = time.time()
    prediction, confidence, threat_prob = classify_text(text)

    analysis = analyze_alert(
        alert_text=text,
        prediction=prediction,
        confidence=confidence,
        api_key=alert.api_key,
    )
    ms = (time.time() - t0) * 1000

    return AnalyzeResponse(
        standardized_text=text,
        prediction=prediction,
        confidence=confidence,
        threat_probability=threat_prob,
        category=category,
        analysis=analysis,
        inference_ms=round(ms, 2),
    )


@app.post("/ask")
async def ask(req: AskInput):
    """Analyst asks a follow-up question about an alert."""
    analysis = analyze_alert(
        alert_text=req.alert_text,
        prediction=req.prediction,
        confidence=req.confidence,
        analyst_question=req.question,
        api_key=req.api_key,
    )
    return {"question": req.question, "answer": analysis}


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
