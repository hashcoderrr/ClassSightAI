"""Tests for Excel / CSV / JSON report generation."""
from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np
from openpyxl import load_workbook

from src.report_excel import (GREEN, RED, export_csv_json, fmt_ts, generate_report, report_filename,
                              write_workbook)


def _rows(tmp_path):
    snap = tmp_path / "snap.jpg"
    cv2.imwrite(str(snap), np.full((80, 80, 3), 128, np.uint8))
    return [
        {"roll_no": "1", "name": "Asha", "class_name": "10A", "status": "Present", "first_seen": 5.0,
         "last_seen": 3605.0, "time_visible": 3000.0, "confidence": 0.71234, "snapshot_path": str(snap),
         "remarks": ""},
        {"roll_no": "2", "name": "Bilal", "class_name": "10A", "status": "Absent", "first_seen": None,
         "last_seen": None, "time_visible": 0.0, "confidence": 0.0, "snapshot_path": None, "remarks": ""},
        {"roll_no": "3", "name": "Chitra", "class_name": "10A", "status": "Late", "first_seen": 400.0,
         "last_seen": 900.0, "time_visible": 500.0, "confidence": 0.6, "snapshot_path": "missing.jpg",
         "remarks": "Left early"},
    ], snap


def test_workbook_structure_and_formatting(tmp_path):
    rows, snap = _rows(tmp_path)
    unknowns = [{"cluster_id": 1, "track_id": 9, "first_seen": 10.0, "last_seen": 20.0, "thumb_path": str(snap)}]
    history = [{"roll_no": "1", "name": "Asha", "class_name": "10A", "sessions": 4, "attended": 4, "late": 0,
                "absent": 0, "excused": 0, "percentage": 100.0},
               {"roll_no": "2", "name": "Bilal", "class_name": "10A", "sessions": 4, "attended": 2, "late": 0,
                "absent": 2, "excused": 0, "percentage": 50.0}]
    meta = {"class_name": "10A", "subject": "Maths", "date": "2026-11-05 09:00", "duration_sec": 3700,
            "conflicts": [{"roll_no": "1", "ts": 12.0, "kept_track": 1, "dropped_track": 4, "kept_score": 0.8,
                           "dropped_score": 0.5}],
            "headcounts": [{"ts": 300, "count": 2, "rolls": ["1", "3"]}]}
    out = write_workbook(tmp_path / "r.xlsx", meta, rows, unknowns, history)
    wb = load_workbook(out)
    assert wb.sheetnames == ["Attendance", "Summary", "Unknown Faces", "History"]

    ws = wb["Attendance"]
    assert [c.value for c in ws[1]][:9] == ["Roll No", "Name", "Class", "Status", "First Seen", "Last Seen",
                                            "Total Time Visible", "Confidence Score", "Remarks"]
    assert ws["J1"].value == "Face Snapshot"
    assert ws.freeze_panes == "A2"
    assert ws["D2"].value == "Present" and ws["D2"].fill.start_color.rgb.endswith(GREEN.start_color.rgb[-6:])
    assert ws["D3"].value == "Absent" and ws["D3"].fill.start_color.rgb.endswith(RED.start_color.rgb[-6:])
    assert ws["E2"].value == "00:00:05" and ws["F2"].value == "01:00:05"
    assert ws["E3"].value == "-"
    assert len(ws._images) == 1                       # missing snapshot skipped gracefully
    assert ws.column_dimensions["B"].width >= len("Chitra")

    sm = wb["Summary"]
    vals = {sm.cell(row=r, column=1).value: sm.cell(row=r, column=2).value for r in range(2, 16)}
    assert vals["Total Enrolled"] == 3 and vals["Present (on time)"] == 1 and vals["Late"] == 1
    assert vals["Absent"] == 1 and vals["Attendance %"] == round(200 / 3, 2)
    assert vals["Unknown Faces (tracks)"] == 1 and vals["Duplicate/Spoof Flags"] == 1
    assert len(sm._charts) == 1

    uk = wb["Unknown Faces"]
    assert uk["A2"].value == 1 and uk["B2"].value == 9 and len(uk._images) == 1

    hs = wb["History"]
    assert hs["I3"].value == 50.0 and hs["A3"].fill.start_color.rgb.endswith(RED.start_color.rgb[-6:])
    assert not hs["A2"].fill.start_color.rgb.endswith(RED.start_color.rgb[-6:])


def test_no_images_mode(tmp_path):
    rows, _ = _rows(tmp_path)
    wb = load_workbook(write_workbook(tmp_path / "n.xlsx", {"class_name": "10A"}, rows, [], None,
                                      embed_images=False))
    assert wb.sheetnames == ["Attendance", "Summary", "Unknown Faces"]
    assert len(wb["Attendance"]._images) == 0


def test_empty_session(tmp_path):
    wb = load_workbook(write_workbook(tmp_path / "e.xlsx", {}, [], []))
    assert wb["Attendance"].max_row == 1


def test_filename_format():
    name = report_filename("10 A", "Maths/Algebra", datetime(2026, 11, 5, 9, 30))
    assert name == "Attendance_10-A_Maths-Algebra_2026-11-05_09-30.xlsx"


def test_fmt_ts():
    assert fmt_ts(None) == "-" and fmt_ts(0) == "00:00:00" and fmt_ts(3661.4) == "01:01:01"


def test_csv_json_export(tmp_path):
    rows, _ = _rows(tmp_path)
    paths = export_csv_json(tmp_path / "Attendance_x.xlsx", {"class_name": "10A"}, rows)
    assert [p.suffix for p in paths] == [".csv", ".json"]
    data = json.loads(paths[1].read_text())
    assert data["attendance"][0]["roll_no"] == "1" and "snapshot_path" not in data["attendance"][0]
    assert paths[0].read_text().splitlines()[0].startswith("roll_no,name")


def test_generate_report_from_db_reflects_override(cfg, enrolled):
    db = enrolled
    sid = db.create_session("10A", "Maths", "test.mp4", output_dir=str(cfg.path("outputs_dir")))
    db.save_attendance(sid, [{"roll_no": "1", "status": "Present", "first_seen": 1, "last_seen": 9,
                              "time_visible": 8, "confidence": 0.7}])
    db.update_session(sid, status="completed")
    db.set_status(sid, "2", "Present", "teacher", "came late, camera missed")
    p = generate_report(db, sid, cfg)
    assert p.name.startswith("Attendance_10A_Maths_") and p.exists()
    assert p.with_suffix(".csv").exists() and p.with_suffix(".json").exists()
    ws = load_workbook(p)["Attendance"]
    status = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=4).value for r in range(2, ws.max_row + 1)}
    assert status == {"1": "Present", "2": "Present", "3": "Absent"}
    remarks = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=9).value for r in range(2, ws.max_row + 1)}
    assert "Manual override" in remarks["2"]
    log = db.get_audit_log(sid)
    assert log[0]["roll_no"] == "2" and log[0]["old_value"] is None and log[0]["new_value"] == "Present"
