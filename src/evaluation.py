"""Accuracy evaluation against a manually labelled ground-truth CSV.

Ground truth CSV columns: ``roll_no, present`` where ``present`` is one of
1/0, yes/no, true/false, present/absent/late (late counts as present).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

POSITIVE = {"1", "yes", "y", "true", "present", "p", "late", "l"}
NEGATIVE = {"0", "no", "n", "false", "absent", "a", "excused"}


def parse_label(value: Any) -> bool:
    v = str(value).strip().lower()
    if v in POSITIVE:
        return True
    if v in NEGATIVE:
        return False
    raise ValueError(f"Unrecognised label {value!r}")


def load_labels(path: str | Path, roll_col: str = "roll_no", label_cols: tuple[str, ...] = (
        "present", "status", "label")) -> dict[str, bool]:
    """Load roll_no -> present from a CSV (ground truth or SmartAttend CSV export)."""
    out: dict[str, bool] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        rc = cols.get(roll_col) or cols.get("roll no") or cols.get("roll")
        lc = next((cols[c] for c in label_cols if c in cols), None)
        if rc is None or lc is None:
            raise ValueError(f"{path}: needs columns '{roll_col}' and one of {label_cols}")
        for row in reader:
            roll = str(row[rc]).strip()
            if roll:
                out[roll] = parse_label(row[lc])
    return out


def load_predictions_json(path: str | Path) -> dict[str, bool]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["attendance"] if isinstance(data, dict) else data
    return {str(r["roll_no"]): parse_label(r["status"]) for r in rows}


@dataclass
class EvalResult:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    accuracy: float
    f1: float
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    missing_in_predictions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(truth: dict[str, bool], pred: dict[str, bool]) -> EvalResult:
    """Compare predictions to ground truth. Rolls missing from predictions count as Absent."""
    tp = fp = fn = tn = 0
    fps, fns, missing = [], [], []
    for roll, actual in sorted(truth.items()):
        if roll not in pred:
            missing.append(roll)
        p = pred.get(roll, False)
        if p and actual:
            tp += 1
        elif p and not actual:
            fp += 1
            fps.append(roll)
        elif not p and actual:
            fn += 1
            fns.append(roll)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvalResult(tp, fp, fn, tn, round(precision, 4), round(recall, 4), round(accuracy, 4), round(f1, 4),
                      fps, fns, missing)
