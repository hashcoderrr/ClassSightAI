"""Formatted Excel (plus CSV / JSON) attendance reports built with openpyxl."""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.chart import PieChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

log = logging.getLogger(__name__)

GREEN = PatternFill("solid", start_color="C6EFCE", end_color="C6EFCE")
RED = PatternFill("solid", start_color="FFC7CE", end_color="FFC7CE")
AMBER = PatternFill("solid", start_color="FFEB9C", end_color="FFEB9C")
GREY = PatternFill("solid", start_color="E7E6E6", end_color="E7E6E6")
HEADER = PatternFill("solid", start_color="1F4E78", end_color="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
THIN = Border(*(Side(style="thin", color="BFBFBF"),) * 4)
STATUS_FILL = {"Present": GREEN, "Late": AMBER, "Absent": RED, "Excused": GREY}

THUMB_PX = 64
ATTENDANCE_HEADERS = ["Roll No", "Name", "Class", "Status", "First Seen", "Last Seen", "Total Time Visible",
                      "Confidence Score", "Remarks", "Face Snapshot"]


def fmt_ts(seconds: Optional[float]) -> str:
    """Seconds into the video -> HH:MM:SS."""
    if seconds is None:
        return "-"
    s = int(round(float(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", str(text)).strip("-") or "NA"


def report_filename(class_name: str, subject: str, when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return f"Attendance_{safe_name(class_name)}_{safe_name(subject)}_{when.strftime('%Y-%m-%d_%H-%M')}.xlsx"


def _thumb(path: Optional[str]) -> Optional[XLImage]:
    if not path or not Path(path).exists():
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_PX, THUMB_PX))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        buf.seek(0)
        img = XLImage(buf)
        img.width, img.height = im.size
        return img
    except Exception as exc:  # corrupted thumbnail must not break the report
        log.warning("Could not embed thumbnail %s: %s", path, exc)
        return None


def _style_header(ws: Worksheet, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill, cell.font, cell.border = HEADER, HEADER_FONT, THIN
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _autosize(ws: Worksheet, fixed: Optional[dict[int, float]] = None, max_width: float = 50) -> None:
    fixed = fixed or {}
    widths: dict[int, float] = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            widths[cell.column] = max(widths.get(cell.column, 0), len(str(cell.value)))
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = fixed.get(col, min(max_width, w + 3))
    for col, w in fixed.items():
        ws.column_dimensions[get_column_letter(col)].width = w


def write_workbook(path: str | Path, meta: dict[str, Any], rows: list[dict[str, Any]],
                   unknowns: list[dict[str, Any]], history: Optional[list[dict[str, Any]]] = None,
                   embed_images: bool = True, history_threshold: float = 75.0) -> Path:
    """Write the 4-sheet attendance workbook.

    Args:
        path: Destination .xlsx path.
        meta: Session info: class_name, subject, date, source, duration_sec, conflicts, headcounts.
        rows: One dict per enrolled student (see :func:`src.attendance.build_rows` / DB rows).
        unknowns: Unknown-face rows (cluster_id, track_id, first_seen, last_seen, thumb_path).
        history: Cumulative per-student history (optional sheet 4).
        embed_images: Embed face thumbnails (disable for privacy-preserving exports).
        history_threshold: Highlight students below this attendance percentage.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()

    # ---------------------------------------------------------- Sheet 1
    ws = wb.active
    ws.title = "Attendance"
    ws.append(ATTENDANCE_HEADERS)
    _style_header(ws, 1, len(ATTENDANCE_HEADERS))
    snap_col = len(ATTENDANCE_HEADERS)
    for i, r in enumerate(rows, start=2):
        status = r.get("status", "Absent")
        ws.append([r.get("roll_no"), r.get("name"), r.get("class_name"), status,
                   fmt_ts(r.get("first_seen")), fmt_ts(r.get("last_seen")), fmt_ts(r.get("time_visible") or 0),
                   round(float(r.get("confidence") or 0.0), 3), r.get("remarks") or "", ""])
        fill = STATUS_FILL.get(status, RED)
        for c in range(1, len(ATTENDANCE_HEADERS) + 1):
            cell = ws.cell(row=i, column=c)
            cell.fill, cell.border = fill, THIN
            cell.alignment = Alignment(vertical="center")
        if embed_images:
            img = _thumb(r.get("snapshot_path"))
            if img is not None:
                ws.add_image(img, f"{get_column_letter(snap_col)}{i}")
                ws.row_dimensions[i].height = THUMB_PX * 0.78
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(ATTENDANCE_HEADERS))}{max(1, len(rows) + 1)}"
    _autosize(ws, {snap_col: 11} if embed_images else None)

    # ---------------------------------------------------------- Sheet 2
    present = sum(1 for r in rows if r.get("status") == "Present")
    late = sum(1 for r in rows if r.get("status") == "Late")
    absent = sum(1 for r in rows if r.get("status") == "Absent")
    excused = sum(1 for r in rows if r.get("status") == "Excused")
    total = len(rows)
    denom = total - excused
    pct = round(100.0 * (present + late) / denom, 2) if denom > 0 else 0.0
    n_unknown_clusters = len({u.get("cluster_id") for u in unknowns})
    sm = wb.create_sheet("Summary")
    sm.append(["Field", "Value"])
    _style_header(sm, 1, 2)
    summary = [
        ("Date", meta.get("date", datetime.now().strftime("%Y-%m-%d %H:%M"))),
        ("Class", meta.get("class_name", "")),
        ("Session / Subject", meta.get("subject", "")),
        ("Source", meta.get("source", "")),
        ("Video Duration", fmt_ts(meta.get("duration_sec", 0))),
        ("Total Enrolled", total),
        ("Present (on time)", present),
        ("Late", late),
        ("Absent", absent),
        ("Excused", excused),
        ("Attendance %", pct),
        ("Unknown Faces (tracks)", len(unknowns)),
        ("Unknown Faces (clusters)", n_unknown_clusters),
        ("Duplicate/Spoof Flags", len(meta.get("conflicts", []))),
    ]
    for k, v in summary:
        sm.append([k, v])
    for row in sm.iter_rows(min_row=2, max_row=sm.max_row):
        for cell in row:
            cell.border = THIN
        row[0].font = Font(bold=True)
    sm.freeze_panes = "A2"

    # chart data block
    sm["D1"], sm["E1"] = "Status", "Count"
    _style_header(sm, 1, 5)
    sm["C1"].fill, sm["C1"].border = PatternFill(), Border()
    for j, (label, val) in enumerate([("Present", present), ("Late", late), ("Absent", absent),
                                      ("Excused", excused)], start=2):
        sm.cell(row=j, column=4, value=label).fill = STATUS_FILL[label]
        sm.cell(row=j, column=5, value=val)
    pie = PieChart()
    pie.title = f"Attendance - {meta.get('class_name', '')} {meta.get('subject', '')}".strip()
    pie.add_data(Reference(sm, min_col=5, min_row=1, max_row=5), titles_from_data=True)
    pie.set_categories(Reference(sm, min_col=4, min_row=2, max_row=5))
    pie.dataLabels = DataLabelList()
    pie.dataLabels.showPercent = True
    pie.height, pie.width = 8, 12
    sm.add_chart(pie, "G2")

    headcounts = meta.get("headcounts") or []
    if headcounts:
        start = sm.max_row + 3
        sm.cell(row=start, column=1, value="Periodic Headcount")
        sm.cell(row=start, column=1).font = Font(bold=True)
        sm.cell(row=start + 1, column=1, value="Time")
        sm.cell(row=start + 1, column=2, value="Students Visible")
        _style_header(sm, start + 1, 2)
        for k, hc in enumerate(headcounts, start=start + 2):
            sm.cell(row=k, column=1, value=fmt_ts(hc.get("ts")))
            sm.cell(row=k, column=2, value=hc.get("count"))
    conflicts = meta.get("conflicts") or []
    if conflicts:
        start = sm.max_row + 3
        sm.cell(row=start, column=1, value="Duplicate / Spoof Flags").font = Font(bold=True)
        hdr = ["Roll No", "Time", "Kept Track", "Dropped Track", "Kept Score", "Dropped Score"]
        for c, h in enumerate(hdr, start=1):
            sm.cell(row=start + 1, column=c, value=h)
        _style_header(sm, start + 1, len(hdr))
        for k, cf in enumerate(conflicts, start=start + 2):
            vals = [cf.get("roll_no"), fmt_ts(cf.get("ts")), cf.get("kept_track"), cf.get("dropped_track"),
                    round(float(cf.get("kept_score", 0)), 3), round(float(cf.get("dropped_score", 0)), 3)]
            for c, v in enumerate(vals, start=1):
                sm.cell(row=k, column=c, value=v)
    _autosize(sm)

    # ---------------------------------------------------------- Sheet 3
    uk = wb.create_sheet("Unknown Faces")
    uk_headers = ["Cluster ID", "Track ID", "First Seen", "Last Seen", "Assigned To", "Thumbnail"]
    uk.append(uk_headers)
    _style_header(uk, 1, len(uk_headers))
    for i, u in enumerate(unknowns, start=2):
        uk.append([u.get("cluster_id"), u.get("track_id"), fmt_ts(u.get("first_seen")), fmt_ts(u.get("last_seen")),
                   u.get("assigned_roll") or "", ""])
        for c in range(1, len(uk_headers) + 1):
            uk.cell(row=i, column=c).border = THIN
        if embed_images:
            img = _thumb(u.get("thumb_path"))
            if img is not None:
                uk.add_image(img, f"F{i}")
                uk.row_dimensions[i].height = THUMB_PX * 0.78
    uk.freeze_panes = "A2"
    _autosize(uk, {6: 11} if embed_images else None)

    # ---------------------------------------------------------- Sheet 4
    if history is not None:
        hs = wb.create_sheet("History")
        h_headers = ["Roll No", "Name", "Class", "Sessions", "Attended", "Late", "Absent", "Excused", "Attendance %"]
        hs.append(h_headers)
        _style_header(hs, 1, len(h_headers))
        for i, h in enumerate(history, start=2):
            hs.append([h.get("roll_no"), h.get("name"), h.get("class_name"), h.get("sessions", 0),
                       h.get("attended", 0), h.get("late", 0), h.get("absent", 0), h.get("excused", 0),
                       h.get("percentage", 0.0)])
            low = h.get("sessions", 0) > 0 and float(h.get("percentage", 0.0)) < history_threshold
            for c in range(1, len(h_headers) + 1):
                cell = hs.cell(row=i, column=c)
                cell.border = THIN
                if low:
                    cell.fill = RED
                    cell.font = Font(bold=True, color="9C0006")
        hs.freeze_panes = "A2"
        _autosize(hs)

    wb.save(path)
    log.info("Excel report written: %s", path)
    return path


def export_csv_json(base_path: str | Path, meta: dict[str, Any], rows: list[dict[str, Any]],
                    write_csv: bool = True, write_json: bool = True) -> list[Path]:
    """Write ``<base>.csv`` / ``<base>.json`` copies of the attendance rows (no images)."""
    base = Path(base_path).with_suffix("")
    fields = ["roll_no", "name", "class_name", "status", "first_seen", "last_seen", "time_visible",
              "confidence", "remarks"]
    out: list[Path] = []
    clean = [{k: r.get(k) for k in fields} for r in rows]
    if write_csv:
        p = base.with_suffix(".csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(clean)
        out.append(p)
    if write_json:
        p = base.with_suffix(".json")
        payload = {"session": {k: v for k, v in meta.items() if k not in ("headcounts",)},
                   "headcounts": meta.get("headcounts", []), "attendance": clean}
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        out.append(p)
    return out


def generate_report(db: Any, session_id: int, cfg: Any, embed_images: Optional[bool] = None,
                    out_path: Optional[Path] = None, record: bool = True) -> Path:
    """(Re)build the report for a session from the database (includes overrides).

    With ``record=False`` only the workbook is written (no CSV/JSON, and the
    session's stored report path is left unchanged) - used for ad-hoc exports.
    """
    sess = db.get_session(session_id)
    if sess is None:
        raise ValueError(f"Session {session_id} not found")
    students = {s.roll_no: s for s in db.list_students(sess["class_name"])}
    att = {a["roll_no"]: a for a in db.get_attendance(session_id)}
    rows: list[dict[str, Any]] = []
    for roll in sorted(set(students) | set(att)):
        a = att.get(roll, {"status": "Absent"})
        s = students.get(roll)
        rows.append({"roll_no": roll, "name": s.name if s else "(deleted)",
                     "class_name": s.class_name if s else sess["class_name"], **{k: a.get(k) for k in (
                         "status", "first_seen", "last_seen", "time_visible", "confidence", "snapshot_path",
                         "remarks")}})
        if a.get("overridden"):
            rows[-1]["remarks"] = "; ".join(x for x in [a.get("remarks") or "", "Manual override"] if x)
    unknowns = db.get_unknowns(session_id)
    started = datetime.fromisoformat(sess["started_at"])
    meta = {"session_id": session_id, "class_name": sess["class_name"], "subject": sess["subject"],
            "date": started.strftime("%Y-%m-%d %H:%M"), "source": sess.get("source", ""),
            "duration_sec": sess.get("duration_sec", 0), **sess.get("meta", {})}
    rcfg = cfg.report
    embed = bool(rcfg.embed_snapshots) if embed_images is None else embed_images
    out_dir = Path(sess.get("output_dir") or cfg.path("outputs_dir"))
    out_path = out_path or out_dir / report_filename(sess["class_name"], sess["subject"], started)
    history = db.history(sess["class_name"])
    write_workbook(out_path, meta, rows, unknowns, history, embed, float(rcfg.history_threshold))
    if not record:
        return out_path
    export_csv_json(out_path, meta, rows, bool(rcfg.write_csv), bool(rcfg.write_json))
    db.update_session(session_id, report_path=str(out_path))
    return out_path
