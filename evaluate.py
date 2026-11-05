"""Evaluate ClassSightAI against a manually labelled ground-truth CSV.

Usage:
    python evaluate.py --truth ground_truth.csv --session 3
    python evaluate.py --truth ground_truth.csv --pred outputs/<session>/Attendance_....csv
    python evaluate.py --truth ground_truth.csv --pred outputs/<session>/Attendance_....json --out eval.json

Ground-truth CSV format (header required):
    roll_no,present
    1,1
    2,0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation import evaluate, load_labels, load_predictions_json, parse_label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="Ground-truth CSV (roll_no,present)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", help="ClassSightAI CSV or JSON export")
    g.add_argument("--session", type=int, help="Session id in the ClassSightAI database")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", help="Write metrics JSON here")
    args = ap.parse_args(argv)

    truth = load_labels(args.truth)
    if args.pred:
        pred = load_predictions_json(args.pred) if Path(args.pred).suffix.lower() == ".json" \
            else load_labels(args.pred)
    else:
        from src.config import load_config
        from src.db import Database
        cfg = load_config(args.config)
        db = Database.from_config(cfg)
        pred = {a["roll_no"]: parse_label(a["status"]) for a in db.get_attendance(args.session)}
        db.close()
        if not pred:
            print(f"No attendance rows for session {args.session}", file=sys.stderr)
            return 2

    r = evaluate(truth, pred)
    print(f"Students evaluated : {r.tp + r.fp + r.fn + r.tn}")
    print(f"TP={r.tp}  FP={r.fp}  FN={r.fn}  TN={r.tn}")
    print(f"Precision : {r.precision:.4f}")
    print(f"Recall    : {r.recall:.4f}")
    print(f"Accuracy  : {r.accuracy:.4f}")
    print(f"F1 score  : {r.f1:.4f}")
    if r.false_positives:
        print(f"False positives (marked present, actually absent): {', '.join(r.false_positives)}")
    if r.false_negatives:
        print(f"False negatives (missed students): {', '.join(r.false_negatives)}")
    if r.missing_in_predictions:
        print(f"Not in predictions (treated as absent): {', '.join(r.missing_in_predictions)}")
    if args.out:
        Path(args.out).write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
        print(f"Metrics written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
