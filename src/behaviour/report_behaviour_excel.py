"""Behaviour Excel report (+ CSV / JSON exports)."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.chart import BarChart, DoughnutChart, LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from ..report_excel import HEADER, HEADER_FONT, THIN, _autosize, _style_header, fmt_ts, safe_name
from .signals import BEHAVIOURS, DISPLAY

log = logging.getLogger(__name__)
WARN_FILL = PatternFill("solid", start_color="FFF2CC", end_color="FFF2CC")
SCORE_SCALE = dict(start_type="num", start_value=0, start_color="F8696B", mid_type="num", mid_value=50,
                   mid_color="FFEB84", end_type="num", end_value=100, end_color="63BE7B")


def behaviour_filename(class_name: str, subject: str, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return f"Behaviour_{safe_name(class_name)}_{safe_name(subject)}_{when.strftime('%Y-%m-%d_%H-%M')}.xlsx"


def _bin_label(start: float) -> str:
    s = int(start)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _table(ws, top: int, left: int, headers: list[str], rows: list[list[Any]]) -> int:
    for j, h in enumerate(headers):
        ws.cell(row=top, column=left + j, value=h)
    for j in range(len(headers)):
        c = ws.cell(row=top, column=left + j)
        c.fill, c.font, c.border = HEADER, HEADER_FONT, THIN
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            ws.cell(row=top + i, column=left + j, value=v).border = THIN
    return top + len(rows) + 1


def write_behaviour_workbook(path: str | Path, result: dict[str, Any], include_movement: bool = True,
                             max_movement_rows: int = 200_000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    class_only = bool(result.get("class_only"))
    scored = sorted([s for s in result.get("subjects", []) if s["score"] is not None], key=lambda s: -s["score"])

    # ----------------------------------------------------------- Class Summary
    ws = wb.active
    ws.title = "Class Summary"
    ws.merge_cells("A1:H2")
    ws["A1"] = "NOTE: " + result.get("disclaimer", "")
    ws["A1"].font = Font(bold=True, color="9C0006")
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="center")
    ws["A1"].fill = WARN_FILL
    ws.row_dimensions[1].height = 30
    info = [("Class", result.get("class_name", "")), ("Subject", result.get("subject", "")),
            ("Date", str(result.get("started_at", ""))[:16].replace("T", " ")),
            ("Camera", result.get("camera_id", "")), ("Video duration", fmt_ts(result.get("duration", 0))),
            ("Students analysed", result.get("students_analysed", 0)),
            ("Class attentiveness score", result.get("class_score")),
            ("Class label", result.get("class_label", "")),
            ("Attentive students (>= Attentive)", result.get("attentive_count", 0)),
            ("Inattentive students", result.get("inattentive_count", 0)),
            ("Engagement drop points", len(result.get("drop_points", []))),
            ("Teacher corrections applied", result.get("corrections", 0)),
            ("Privacy mode", "Class-level only" if class_only else "Per-student")]
    if result.get("withheld_no_consent"):
        info.append(("Students without analytics consent (aggregate only)", result["withheld_no_consent"]))
    row = _table(ws, 4, 1, ["Field", "Value"], [[k, v] for k, v in info]) + 1

    # label distribution (chart data)
    lc = result.get("label_counts", {})
    lab_top = row
    row = _table(ws, row, 1, ["Attentiveness label", "Students"], [[k, v] for k, v in lc.items()]) + 1
    bar = BarChart()
    bar.title, bar.y_axis.title = "Students per attentiveness label", "Students"
    bar.add_data(Reference(ws, min_col=2, min_row=lab_top, max_row=lab_top + len(lc)), titles_from_data=True)
    bar.set_categories(Reference(ws, min_col=1, min_row=lab_top + 1, max_row=lab_top + len(lc)))
    bar.height, bar.width, bar.legend = 7, 12, None
    ws.add_chart(bar, "J4")

    # behaviour time distribution (doughnut)
    totals = result.get("behaviour_totals", {})
    tot_rows = [[DISPLAY[b], round(totals.get(b, 0) / 60, 2)] for b in BEHAVIOURS if totals.get(b, 0) > 0]
    tot_top = row
    row = _table(ws, row, 1, ["Behaviour (class total)", "Minutes"], tot_rows) + 1
    if tot_rows:
        d = DoughnutChart()
        d.title = "Class time per behaviour"
        d.add_data(Reference(ws, min_col=2, min_row=tot_top, max_row=tot_top + len(tot_rows)), titles_from_data=True)
        d.set_categories(Reference(ws, min_col=1, min_row=tot_top + 1, max_row=tot_top + len(tot_rows)))
        d.dataLabels = DataLabelList()
        d.dataLabels.showPercent = True
        d.height, d.width = 8, 12
        ws.add_chart(d, "J20")

    dist = result.get("distractions", [])
    ws.cell(row=row, column=1, value="Most common distractions").font = Font(bold=True)
    row = _table(ws, row + 1, 1, ["Behaviour", "Minutes", "Students affected"],
                 [[d["display"], d["minutes"], d["students"]] for d in dist]) + 1
    drops = result.get("drop_points", [])
    ws.cell(row=row, column=1, value="Class engagement drop points").font = Font(bold=True)
    row = _table(ws, row + 1, 1, ["From", "To", "Class score", "Previous mean", "Drop"],
                 [[_bin_label(d["start"]), _bin_label(d["end"]), d["score"], d["previous_mean"], d["drop"]]
                  for d in drops]) + 1
    if not class_only and scored:
        ws.cell(row=row, column=1, value="Top 5 students").font = Font(bold=True)
        row = _table(ws, row + 1, 1, ["Roll No / ID", "Name", "Score", "Label"],
                     [[s["roll_no"] or s["key"], s["name"], s["score"], s["label"]] for s in scored[:5]]) + 1
        ws.cell(row=row, column=1, value="Bottom 5 students").font = Font(bold=True)
        row = _table(ws, row + 1, 1, ["Roll No / ID", "Name", "Score", "Label"],
                     [[s["roll_no"] or s["key"], s["name"], s["score"], s["label"]]
                      for s in scored[::-1][:5]]) + 1
    _autosize(ws, max_width=45)
    ws.column_dimensions["A"].width = 42

    # ----------------------------------------------------------- Student Scores
    if not class_only:
        ss = wb.create_sheet("Student Scores")
        beh = [b for b in BEHAVIOURS]
        headers = ["Roll No / ID", "Name", "Attendance", "Attentiveness Score", "Label", "Confidence",
                   "Observed (min)"]
        for b in beh:
            headers += [f"{DISPLAY[b]} (min)", f"{DISPLAY[b]} (%)"]
        headers += ["Hand raises", "Phone uses", "Talking episodes", "Sleep episodes", "Stand-ups", "Head turns",
                    "Posture changes", "Displacement (face heights)"]
        ss.append(headers)
        _style_header(ss, 1, len(headers))
        subj = sorted(result.get("subjects", []), key=lambda s: (s["kind"] != "student", str(s["key"])))
        for s in subj:
            r = [s["roll_no"] or s["key"], s["name"], s["attendance"], s["score"], s["label"], s["confidence"],
                 round(s["observed_sec"] / 60, 2)]
            for b in beh:
                r += [round(s["behaviour_seconds"][b] / 60, 2), s["behaviour_pct"][b]]
            c, m = s["counts"], s["movement"]
            r += [c["hand_raises"], c["phone_uses"], c["talking_episodes"], c["sleep_episodes"], c["stand_ups"],
                  m["head_turns"], m["posture_changes"], m["displacement_face_heights"]]
            ss.append(r)
        last = max(2, ss.max_row)
        ss.conditional_formatting.add(f"D2:D{last}", ColorScaleRule(**SCORE_SCALE))
        ss.freeze_panes = "C2"
        ss.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last}"
        _autosize(ss, max_width=22)

        # ------------------------------------------------------- Behaviour Log
        lg = wb.create_sheet("Behaviour Log")
        lh = ["Roll No / ID", "Name", "Behaviour", "Start", "End", "Duration (s)", "Confidence", "Frame start",
              "Frame end", "Teacher-corrected"]
        lg.append(lh)
        _style_header(lg, 1, len(lh))
        for e in result.get("events", []):
            lg.append([e["roll_no"] or e["key"], e["name"], e["display"], fmt_ts(e["start"]), fmt_ts(e["end"]),
                       e["duration"], e["confidence"], e["frame_start"], e["frame_end"],
                       "yes" if e.get("corrected") else ""])
        lg.freeze_panes = "A2"
        lg.auto_filter.ref = f"A1:{get_column_letter(len(lh))}{max(2, lg.max_row)}"
        _autosize(lg)

    # ----------------------------------------------------------- Class Timeline
    ct = wb.create_sheet("Class Timeline")
    ct.append(["Time bin start", "Time bin end", "Class attention score", "Students", "Drop point"])
    _style_header(ct, 1, 5)
    drop_bins = {d["bin"] for d in drops}
    tl = result.get("class_timeline", [])
    for t in tl:
        ct.append([_bin_label(t["start"]), _bin_label(t["end"]), t["score"], t["students"],
                   "DROP" if t["bin"] in drop_bins else ""])
    ct.freeze_panes = "A2"
    if tl:
        lc_ = LineChart()
        lc_.title, lc_.y_axis.title, lc_.x_axis.title = "Class attention curve", "Score (0-100)", "Time"
        lc_.y_axis.scaling.min, lc_.y_axis.scaling.max = 0, 100
        lc_.add_data(Reference(ct, min_col=3, min_row=1, max_row=len(tl) + 1), titles_from_data=True)
        lc_.set_categories(Reference(ct, min_col=1, min_row=2, max_row=len(tl) + 1))
        lc_.height, lc_.width = 9, 18
        ct.add_chart(lc_, "G2")
        ct.conditional_formatting.add(f"C2:C{len(tl) + 1}", ColorScaleRule(**SCORE_SCALE))
    _autosize(ct)

    # ----------------------------------------------------------- Student Timelines
    if not class_only:
        st = wb.create_sheet("Student Timelines")
        nb = max([len(s["bins"]) for s in result.get("subjects", [])] + [len(tl)])
        st.append(["Roll No / ID", "Name"] + [_bin_label(i * result.get("bin_sec", 60)) for i in range(nb)])
        _style_header(st, 1, nb + 2)
        for s in subj:
            if not s["bins"]:
                continue
            st.append([s["roll_no"] or s["key"], s["name"]] + list(s["bins"]) + [None] * (nb - len(s["bins"])))
        if st.max_row >= 2 and nb:
            st.conditional_formatting.add(f"C2:{get_column_letter(nb + 2)}{st.max_row}",
                                          ColorScaleRule(**SCORE_SCALE))
        st.freeze_panes = "C2"
        _autosize(st, max_width=12)

        # ------------------------------------------------------- Movement Data
        if include_movement and result.get("per_second"):
            mv = wb.create_sheet("Movement Data")
            mh = ["Roll No / ID", "Name", "Second", "Head yaw", "Head pitch", "X (0-1)", "Y (0-1)", "State", "Score"]
            mv.append(mh)
            _style_header(mv, 1, len(mh))
            for p in result["per_second"][:max_movement_rows]:
                mv.append([p["roll_no"] or p["key"], p["name"], p["t"], p["yaw"], p["pitch"], p["x"], p["y"],
                           DISPLAY.get(p["state"], p["state"]), p["score"]])
            mv.freeze_panes = "A2"
            mv.auto_filter.ref = f"A1:I{max(2, mv.max_row)}"
            _autosize(mv)

    wb.save(path)
    log.info("Behaviour report written: %s", path)
    return path


def export_behaviour(result: dict[str, Any], out_dir: Path, cfg: Any) -> dict[str, str]:
    """Write the Excel workbook plus CSV / JSON exports; returns their paths."""
    rcfg = cfg.behaviour.report
    try:
        when = datetime.fromisoformat(str(result.get("started_at")))
    except ValueError:
        when = datetime.now()
    xlsx = out_dir / behaviour_filename(result.get("class_name", ""), result.get("subject", ""), when)
    write_behaviour_workbook(xlsx, result, bool(rcfg.include_movement_sheet), int(rcfg.max_movement_rows))
    base = xlsx.with_suffix("")
    files = {"excel": str(xlsx)}
    if rcfg.write_csv:
        stud = base.with_name(base.name + "_students.csv")
        with open(stud, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "roll_no", "name", "kind", "attendance", "score", "label", "confidence"]
                       + [f"{b}_sec" for b in BEHAVIOURS] + ["hand_raises", "phone_uses", "talking_episodes",
                                                              "sleep_episodes", "stand_ups", "head_turns",
                                                              "posture_changes", "displacement_face_heights"])
            for s in result.get("subjects", []):
                w.writerow([s["key"], s["roll_no"], s["name"], s["kind"], s["attendance"], s["score"], s["label"],
                            s["confidence"]] + [s["behaviour_seconds"][b] for b in BEHAVIOURS]
                           + [s["counts"]["hand_raises"], s["counts"]["phone_uses"],
                              s["counts"]["talking_episodes"], s["counts"]["sleep_episodes"],
                              s["counts"]["stand_ups"], s["movement"]["head_turns"],
                              s["movement"]["posture_changes"], s["movement"]["displacement_face_heights"]])
        ev = base.with_name(base.name + "_events.csv")
        with open(ev, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "roll_no", "name", "behaviour", "start", "end", "duration",
                                               "confidence", "frame_start", "frame_end", "corrected"])
            w.writeheader()
            for e in result.get("events", []):
                w.writerow({k: e.get(k) for k in w.fieldnames})
        tl = base.with_name(base.name + "_class_timeline.csv")
        with open(tl, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["bin", "start", "end", "score", "students"])
            w.writeheader()
            w.writerows(result.get("class_timeline", []))
        files.update({"csv_students": str(stud), "csv_events": str(ev), "csv_timeline": str(tl)})
    if rcfg.write_json:
        js = base.with_suffix(".json")
        js.write_text(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=1, default=str),
                      encoding="utf-8")
        files["json"] = str(js)
    return files
