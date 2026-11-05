"""Evaluate predicted behaviours against a manually labelled CSV.

Usage:
    python evaluate_behaviour.py --truth labels.csv --session 3
    python evaluate_behaviour.py --truth labels.csv --pred outputs/<session>/Behaviour_..._events.csv --out eval.json

Ground truth CSV: roll_no,start,end,behaviour   (seconds or HH:MM:SS; behaviour keys such as
looking_board, writing, looking_away, sleeping, talking, hand_raised, phone, standing, other)

Reports per-behaviour precision / recall / F1, overall accuracy, macro-F1 and the confusion matrix,
computed on 1-second steps of the labelled time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.behaviour.evaluation import evaluate_behaviours, load_intervals
from src.behaviour.signals import DISPLAY


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", help="Behaviour events CSV (SmartAttend export or same format as truth)")
    g.add_argument("--session", type=int, help="Session id in the SmartAttend database")
    ap.add_argument("--config", default=None)
    ap.add_argument("--step", type=float, default=1.0, help="Rasterisation step in seconds")
    ap.add_argument("--out", help="Write metrics JSON here")
    args = ap.parse_args(argv)

    truth = load_intervals(args.truth)
    if args.pred:
        pred = load_intervals(args.pred)
    else:
        from src.config import load_config
        from src.db import Database
        db = Database.from_config(load_config(args.config))
        res = db.get_behaviour_result(args.session)
        db.close()
        if not res or not res.get("events"):
            print(f"No per-student behaviour events for session {args.session}", file=sys.stderr)
            return 2
        pred = {}
        for e in res["events"]:
            pred.setdefault(e["key"], []).append((e["start"], e["end"], e["behaviour"]))

    r = evaluate_behaviours(truth, pred, args.step)
    print(f"Labelled seconds: {r.seconds}   accuracy: {r.accuracy:.4f}   macro-F1: {r.macro_f1:.4f}\n")
    print(f"{'Behaviour':28s} {'Precision':>9s} {'Recall':>7s} {'F1':>7s} {'Support':>8s}")
    for b, m in r.per_class.items():
        print(f"{DISPLAY[b]:28s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:7.3f} {int(m['support']):8d}")
    print("\nConfusion matrix (rows = truth, columns = predicted):")
    short = [b[:10] for b in r.labels]
    print(" " * 12 + "".join(f"{s:>11s}" for s in short))
    for b, row in zip(short, r.confusion):
        print(f"{b:>12s}" + "".join(f"{v:11d}" for v in row))
    if args.out:
        Path(args.out).write_text(json.dumps(r.__dict__, indent=2), encoding="utf-8")
        print(f"\nMetrics written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
