"""Offline experiment results: GaTech -> exo transform verification and transform ablations.

Reads what the ML_model scripts already write:
  ML_model/runs/transform_verification/<id>/   (Sensor_transformation_script/verify_transform.py)
  ML_model/runs/ablation/<id>/summary.json      (ML_model/scripts/transform_ablation.py)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

ML_RUNS = Path(__file__).resolve().parents[2] / "ML_model" / "runs"
VERIFICATION_ROOT = ML_RUNS / "transform_verification"
ABLATION_ROOT = ML_RUNS / "ablation"

router = APIRouter(prefix="/api/experiments")


def _child(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return path


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return json.loads(path.read_text())


def _dirs(root: Path, marker: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted((d for d in root.iterdir() if (d / marker).exists()), reverse=True)


@router.get("/transform")
def list_verifications() -> list[dict[str, Any]]:
    return [{"id": d.name, **_read_json(d / "meta.json")} for d in _dirs(VERIFICATION_ROOT, "meta.json")]


@router.get("/transform/{verification_id}")
def get_verification(verification_id: str) -> dict[str, Any]:
    d = _child(VERIFICATION_ROOT, verification_id)
    summary = pd.read_csv(d / "summary.csv")
    return {
        "id": verification_id,
        "meta": _read_json(d / "meta.json"),
        "summary": json.loads(summary.to_json(orient="records")),
    }


@router.get("/transform/{verification_id}/strides/{subject}")
def get_subject_strides(verification_id: str, subject: str) -> dict[str, Any]:
    templates = _read_json(_child(VERIFICATION_ROOT, verification_id) / "templates.json")
    if subject not in templates:
        raise HTTPException(status_code=404, detail=f"{subject} not found")
    return templates[subject]


@router.get("/ablations")
def list_ablations() -> list[dict[str, Any]]:
    out = []
    for d in _dirs(ABLATION_ROOT, "summary.json"):
        results = _read_json(d / "summary.json")
        out.append({"id": d.name, "variants": [r["variant"] for r in results]})
    return out


@router.get("/ablations/{ablation_id}")
def get_ablation(ablation_id: str) -> list[dict[str, Any]]:
    return _read_json(_child(ABLATION_ROOT, ablation_id) / "summary.json")
