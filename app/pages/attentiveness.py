"""'Attentiveness & Behaviour' tab of the SmartAttend web app.

Loaded by ``app/streamlit_app.py`` (which handles login and role checks) via
``render(...)``. Opening this file directly as a Streamlit page only shows a
pointer to the main app, so the analytics are never reachable without login.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.behaviour import DISCLAIMER
from src.behaviour.signals import BEHAVIOURS, DISPLAY
from src.report_excel import fmt_ts

BASE = datetime(2000, 1, 1)
COLORS = {"looking_board": "#2ca02c", "writing": "#98df8a", "hand_raised": "#1f77b4", "looking_away": "#ff7f0e",
          "talking": "#ffbb78", "phone": "#d62728", "sleeping": "#9467bd", "standing": "#8c564b",
          "other": "#c7c7c7", "out_of_frame": "#7f7f7f"}


def _first_frame(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not Path(path).exists():
        return None
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def _sessions_with_results(db: Any) -> list[dict[str, Any]]:
    ids = set(db.behaviour_session_ids())
    return [s for s in db.list_sessions() if s["id"] in ids][::-1]


def _filters(result: dict[str, Any], key: str) -> tuple[tuple[float, float], list[str], float]:
    dur = max(1.0, float(result.get("duration") or 1.0))
    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 1])
        tr = c1.slider("Time range (s)", 0.0, dur, (0.0, dur), step=1.0, key=f"tr_{key}")
        beh = c2.multiselect("Behaviour types", BEHAVIOURS, default=BEHAVIOURS, format_func=DISPLAY.get,
                             key=f"bh_{key}")
        ms = c3.slider("Minimum score", 0, 100, 0, key=f"ms_{key}")
    return tr, beh, float(ms)


# --------------------------------------------------------------------------- views
def class_view(result: dict[str, Any], subjects: list[dict[str, Any]], tr: tuple[float, float],
               beh: list[str], class_only: bool) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class attentiveness", "--" if result["class_score"] is None else f"{result['class_score']:.0f}",
              result.get("class_label"))
    c2.metric("Students analysed", result.get("students_analysed", 0))
    c3.metric("Attentive / inattentive", f"{result.get('attentive_count', 0)} / {result.get('inattentive_count', 0)}")
    c4.metric("Engagement drops", len(result.get("drop_points", [])))

    tl = [t for t in result.get("class_timeline", []) if t["end"] > tr[0] and t["start"] < tr[1]]
    if tl:
        df = pd.DataFrame(tl)
        df["minute"] = df["start"] / 60.0
        fig = go.Figure(go.Scatter(x=df["minute"], y=df["score"], mode="lines+markers", name="Class score",
                                   line=dict(color="#1f77b4", width=3)))
        drops = [d for d in result.get("drop_points", []) if tr[0] <= d["start"] <= tr[1]]
        if drops:
            fig.add_trace(go.Scatter(x=[d["start"] / 60 for d in drops], y=[d["score"] for d in drops],
                                     mode="markers", name="Engagement drop",
                                     marker=dict(color="red", size=14, symbol="triangle-down"),
                                     text=[f"-{d['drop']:.0f} pts" for d in drops]))
        fig.update_layout(title="Class attention curve", xaxis_title="Minute", yaxis_title="Score (0-100)",
                          yaxis_range=[0, 105], height=360)
        st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    totals = {b: v for b, v in result.get("behaviour_totals", {}).items() if v > 0 and b in beh}
    if totals:
        fig = px.pie(names=[DISPLAY[b] for b in totals], values=[v / 60 for v in totals.values()], hole=0.5,
                     color=[DISPLAY[b] for b in totals],
                     color_discrete_map={DISPLAY[b]: COLORS[b] for b in totals},
                     title="Class time per behaviour (minutes)")
        left.plotly_chart(fig, width="stretch")
    dist = result.get("distractions", [])
    with right:
        st.markdown("**Most common distractions**")
        st.dataframe(pd.DataFrame(dist)[["display", "minutes", "students"]] if dist else pd.DataFrame(),
                     hide_index=True, width="stretch")
        if result.get("drop_points"):
            st.markdown("**Engagement drop points** (match them to the lecture content)")
            st.dataframe(pd.DataFrame([{"From": fmt_ts(d["start"]), "To": fmt_ts(d["end"]), "Score": d["score"],
                                        "Previous": d["previous_mean"], "Drop": d["drop"]}
                                       for d in result["drop_points"]]), hide_index=True, width="stretch")
    if class_only:
        st.info("Class-level only mode: per-student charts are hidden.")
        return
    scored = [s for s in subjects if s["score"] is not None]
    if not scored:
        return
    seat = [s for s in scored if s.get("position")]
    if seat:
        fig = go.Figure()
        bg = _first_frame((result.get("files") or {}).get("video_preview"))
        if bg is not None:
            from PIL import Image
            fig.add_layout_image(dict(source=Image.fromarray(bg), xref="x", yref="y", x=0, y=0, sizex=1, sizey=1,
                                      sizing="stretch", layer="below", opacity=0.45))
        fig.add_trace(go.Scatter(
            x=[s["position"][0] for s in seat], y=[s["position"][1] for s in seat], mode="markers+text",
            text=[s["name"] for s in seat], textposition="top center",
            marker=dict(size=34, color=[s["score"] for s in seat], colorscale="RdYlGn", cmin=0, cmax=100,
                        showscale=True, colorbar=dict(title="Score"), line=dict(color="black", width=1)),
            hovertemplate="%{text}<br>score %{marker.color:.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, 1], visible=False)
        fig.update_yaxes(range=[1, 0], visible=False, scaleanchor="x",
                         scaleratio=result.get("frame_h", 9) / max(1, result.get("frame_w", 16)))
        fig.update_layout(title="Seating heatmap (attention at each student's position)", height=460)
        st.plotly_chart(fig, width="stretch")
    ranked = sorted(scored, key=lambda s: s["score"])
    fig = px.bar(x=[s["score"] for s in ranked], y=[f"{s['name']} ({s['key']})" for s in ranked], orientation="h",
                 color=[s["score"] for s in ranked], color_continuous_scale="RdYlGn", range_color=[0, 100],
                 labels={"x": "Attentiveness score", "y": "", "color": "Score"}, title="Students ranked")
    fig.update_layout(height=max(300, 26 * len(ranked) + 120))
    st.plotly_chart(fig, width="stretch")


def student_view(db: Any, cfg: Any, user: str, sid: int, result: dict[str, Any], subjects: list[dict[str, Any]],
                 tr: tuple[float, float], beh: list[str]) -> None:
    from src.behaviour.analyzer import rescore_session
    avail = [s for s in subjects if s["score"] is not None]
    if not avail:
        st.info("No per-student data (absent, no analytics consent, or class-level mode).")
        return
    s = st.selectbox("Student", avail, format_func=lambda x: f"{x['key']} - {x['name']}", key=f"stu_{sid}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{s['score']:.0f}", s["label"])
    c2.metric("Confidence", f"{s['confidence']:.2f}")
    c3.metric("Observed", fmt_ts(s["observed_sec"]))
    c4.metric("Hand raises", s["counts"]["hand_raises"])
    bin_sec = float(result.get("bin_sec", 60))
    if s["bins"]:
        df = pd.DataFrame({"minute": [i * bin_sec / 60 for i in range(len(s["bins"]))], "score": s["bins"]})
        cls = [t["score"] for t in result.get("class_timeline", [])]
        fig = go.Figure(go.Scatter(x=df["minute"], y=df["score"], mode="lines+markers", name=s["name"]))
        if cls:
            fig.add_trace(go.Scatter(x=[i * bin_sec / 60 for i in range(len(cls))], y=cls, mode="lines",
                                     name="Class", line=dict(dash="dot", color="grey")))
        fig.update_layout(title="Attention timeline", xaxis_title="Minute", yaxis_range=[0, 105], height=320)
        st.plotly_chart(fig, width="stretch")
    events = [e for e in result.get("events", []) if e["key"] == s["key"] and e["behaviour"] in beh
              and e["end"] > tr[0] and e["start"] < tr[1]]
    if events:
        g = pd.DataFrame([{"Behaviour": e["display"], "Start": BASE + timedelta(seconds=e["start"]),
                           "End": BASE + timedelta(seconds=max(e["end"], e["start"] + 0.5)), "state": e["behaviour"],
                           "Confidence": e["confidence"]} for e in events])
        fig = px.timeline(g, x_start="Start", x_end="End", y="Behaviour", color="Behaviour",
                          color_discrete_map={DISPLAY[b]: COLORS[b] for b in BEHAVIOURS},
                          hover_data=["Confidence"], title="Behaviour timeline")
        fig.update_xaxes(tickformat="%H:%M:%S")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    pct = {b: s["behaviour_pct"][b] for b in BEHAVIOURS if b != "out_of_frame"}
    fig = px.line_polar(r=list(pct.values()) + [list(pct.values())[0]],
                        theta=[DISPLAY[b] for b in pct] + [DISPLAY[list(pct)[0]]], line_close=True,
                        title="Behaviour profile (% of time)")
    fig.update_traces(fill="toself")
    left.plotly_chart(fig, width="stretch")
    if s.get("trail"):
        t = pd.DataFrame(s["trail"], columns=["t", "x", "y"])
        t = t[(t["t"] >= tr[0]) & (t["t"] <= tr[1])]
        fig = px.scatter(t, x="x", y="y", color="t", color_continuous_scale="Viridis", title="Movement trail",
                         labels={"t": "second"})
        fig.add_trace(go.Scatter(x=t["x"], y=t["y"], mode="lines", line=dict(color="grey", width=1),
                                 showlegend=False))
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[1, 0])
        right.plotly_chart(fig, width="stretch")
        m = s["movement"]
        right.caption(f"Displacement {m['displacement_face_heights']} face heights · posture changes "
                      f"{m['posture_changes']} · head turns {m['head_turns']} · stand-ups {m['stand_ups']}")

    st.markdown("**Events** - select a row to jump the video there, or mark it as a false detection.")
    if not events:
        st.caption("No events in the current filter.")
        return
    ev_df = pd.DataFrame([{"Behaviour": e["display"], "Start": fmt_ts(e["start"]), "End": fmt_ts(e["end"]),
                           "Duration (s)": e["duration"], "Confidence": e["confidence"],
                           "Corrected": "yes" if e.get("corrected") else ""} for e in events])
    sel = st.dataframe(ev_df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                       key=f"ev_{sid}_{s['key']}")
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    if rows:
        e = events[rows[0]]
        st.session_state[f"vstart_{sid}"] = int(e["start"])
        _video(result, sid, height_note=f"Jumped to {fmt_ts(e['start'])} ({e['display']})")
        with st.form(f"corr_{sid}_{s['key']}_{rows[0]}"):
            st.write(f"Mark **{e['display']}** {fmt_ts(e['start'])}-{fmt_ts(e['end'])} as wrong")
            to = st.selectbox("Correct behaviour", BEHAVIOURS, index=BEHAVIOURS.index("other"),
                              format_func=DISPLAY.get)
            reason = st.text_input("Reason (saved in the audit log)")
            if st.form_submit_button("Save correction & re-score"):
                db.add_behaviour_correction(sid, s["key"], e["behaviour"], e["start"], e["end"], to, user, reason)
                with st.spinner("Re-scoring from cached signals..."):
                    rescore_session(db, cfg, sid, user)
                st.success("Correction saved; scores updated.")
                st.rerun()


def compare_view(db: Any, result: dict[str, Any], subjects: list[dict[str, Any]], class_only: bool) -> None:
    st.markdown("**Compare students**")
    avail = [s for s in subjects if s["score"] is not None]
    if not class_only and len(avail) >= 2:
        keys = st.multiselect("Choose 2-5 students", [s["key"] for s in avail], default=[s["key"] for s in avail[:2]],
                              max_selections=5, format_func=lambda k: next(f"{k} - {s['name']}" for s in avail
                                                                           if s["key"] == k))
        chosen = [s for s in avail if s["key"] in keys]
        if len(chosen) >= 2:
            bin_sec = float(result.get("bin_sec", 60))
            fig = go.Figure()
            for s in chosen:
                fig.add_trace(go.Scatter(x=[i * bin_sec / 60 for i in range(len(s["bins"]))], y=s["bins"],
                                         mode="lines+markers", name=s["name"]))
            fig.update_layout(title="Attention timelines", xaxis_title="Minute", yaxis_range=[0, 105], height=340)
            st.plotly_chart(fig, width="stretch")
            rows = [{"Student": s["name"], "Behaviour": DISPLAY[b], "Percent": s["behaviour_pct"][b]}
                    for s in chosen for b in BEHAVIOURS if s["behaviour_pct"][b] > 0]
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="Behaviour", y="Percent", color="Student", barmode="group",
                                   title="Time per behaviour (%)"), width="stretch")
    elif class_only:
        st.info("Student comparison is disabled in class-level only mode.")
    st.markdown("**This class across sessions**")
    cls = result.get("class_name")
    trend = []
    per_student: dict[str, list[dict[str, Any]]] = {}
    for sess in db.list_sessions(cls):
        r = db.get_behaviour_result(sess["id"])
        if not r:
            continue
        trend.append({"Session": f"#{sess['id']} {sess['subject']}", "Date": sess["started_at"][:16],
                      "Subject": sess["subject"], "Class score": r.get("class_score")})
        if not class_only:
            for s in r.get("subjects", []):
                if s["score"] is not None and s["kind"] == "student":
                    per_student.setdefault(s["key"], []).append({"Date": sess["started_at"][:16],
                                                                 "Score": s["score"], "Student": s["name"]})
    if len(trend) >= 1:
        st.plotly_chart(px.line(pd.DataFrame(trend), x="Date", y="Class score", color="Subject", markers=True,
                                range_y=[0, 105], title="Class attentiveness trend"), width="stretch")
    if per_student and not class_only:
        pick = st.multiselect("Students over time", sorted(per_student), default=sorted(per_student)[:3],
                              max_selections=5, key="trend_students")
        rows = [r for k in pick for r in per_student[k]]
        if rows:
            st.plotly_chart(px.line(pd.DataFrame(rows), x="Date", y="Score", color="Student", markers=True,
                                    range_y=[0, 105], title="Student attentiveness trend"), width="stretch")


def _video(result: dict[str, Any], sid: int, height_note: str = "") -> None:
    files = result.get("files") or {}
    vp = files.get("video_preview") or files.get("video_full")
    if vp and Path(vp).exists():
        if height_note:
            st.caption(height_note)
        st.video(vp, start_time=int(st.session_state.get(f"vstart_{sid}", 0)))
    else:
        st.caption("Annotated video not available (disabled or removed by the retention policy).")


def downloads(result: dict[str, Any], sid: int) -> None:
    files = result.get("files") or {}
    _video(result, sid)
    items = [("excel", "⬇ Behaviour Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
             ("csv_students", "⬇ Students CSV", "text/csv"), ("csv_events", "⬇ Events CSV", "text/csv"),
             ("csv_timeline", "⬇ Class timeline CSV", "text/csv"), ("json", "⬇ JSON", "application/json")]
    cols = st.columns(len(items))
    for col, (k, label, mime) in zip(cols, items):
        p = files.get(k)
        if p and Path(p).exists():
            col.download_button(label, Path(p).read_bytes(), Path(p).name, mime, key=f"dl_{k}_{sid}")
    c1, c2 = st.columns(2)
    for col, k, label in ((c1, "video_full", "⬇ Annotated video (full resolution)"),
                          (c2, "video_preview", "⬇ Annotated video (720p preview)")):
        p = files.get(k)
        if p and Path(p).exists():
            with open(p, "rb") as fh:
                col.download_button(label, fh, f"behaviour_{k}_{sid}.mp4", "video/mp4", key=f"dl_{k}_{sid}")


def corrections_view(db: Any, cfg: Any, user: str, sid: int) -> None:
    from src.behaviour.analyzer import rescore_session
    corr = db.get_behaviour_corrections(sid)
    if not corr:
        st.caption("No teacher corrections for this session.")
    for c in corr:
        cols = st.columns([5, 1])
        cols[0].write(f"#{c['id']} **{c['subject_key']}** {fmt_ts(c['start'])}-{fmt_ts(c['end'])}: "
                      f"{DISPLAY.get(c['behaviour'], c['behaviour'] or 'any')} → {DISPLAY.get(c['corrected_to'])}"
                      f" · by {c['username']} · {c['reason'] or ''}")
        if cols[1].button("Undo", key=f"undo_{c['id']}"):
            db.delete_behaviour_correction(c["id"], user)
            rescore_session(db, cfg, sid, user)
            st.rerun()
    if st.button("Re-score with current config.yaml weights / thresholds", key=f"rescore_{sid}"):
        with st.spinner("Re-scoring from cached signals (no video re-processing)..."):
            rescore_session(db, cfg, sid, user)
        st.success("Re-scored.")
        st.rerun()


def zone_editor(cfg: Any, db: Any, user: str) -> None:
    from src.behaviour.zones import get_zone, load_zones, save_zone
    st.markdown("Draw the **board / teacher zone** once per camera. Use the **lasso** or **box** tool on the image; "
                "the polygon is saved (normalised) to the zones file and used for 'looking at board'. Without a zone, "
                "facing the camera counts as looking at the board (camera mounted next to the board).")
    cam = st.text_input("Camera / room id", value="default", key="zone_cam")
    src = st.file_uploader("Reference frame: image or short video from this camera",
                           type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], key="zone_src")
    img = None
    if src is not None:
        data = src.getvalue()
        if Path(src.name).suffix.lower() in (".mp4", ".avi", ".mov"):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(src.name).suffix)
            tmp.write(data)
            tmp.close()
            img = _first_frame(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
        else:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr is not None else None
    current = get_zone(cfg, cam)
    if img is None:
        st.caption(f"Current zone for '{cam}': {current if current else 'none (facing-camera mode)'}")
    else:
        h, w = img.shape[:2]
        step = max(w, h) / 45.0
        gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        fig = px.imshow(img)
        fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=6, opacity=0.02),
                                 hoverinfo="skip", showlegend=False))
        if current:
            pts = np.array(current) * [w, h]
            fig.add_trace(go.Scatter(x=list(pts[:, 0]) + [pts[0, 0]], y=list(pts[:, 1]) + [pts[0, 1]],
                                     mode="lines", line=dict(color="lime", width=3), name="current zone"))
        fig.update_layout(dragmode="lasso", height=520, margin=dict(l=0, r=0, t=30, b=0))
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode=("lasso", "box"), key="zone_plot")
        poly = None
        sel = getattr(ev, "selection", None) if ev is not None else None
        if sel:
            lasso = sel.get("lasso") if isinstance(sel, dict) else getattr(sel, "lasso", None)
            box = sel.get("box") if isinstance(sel, dict) else getattr(sel, "box", None)
            points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
            if lasso:
                poly = list(zip(lasso[0]["x"], lasso[0]["y"]))
            elif box:
                (x0, x1), (y0, y1) = box[0]["x"], box[0]["y"]
                poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            elif points and len(points) >= 3:
                hull = cv2.convexHull(np.array([[p["x"], p["y"]] for p in points], np.float32))
                poly = [tuple(p[0]) for p in hull]
        if poly and len(poly) >= 3:
            norm = [[float(x) / w, float(y) / h] for x, y in poly]
            st.success(f"Selected polygon with {len(norm)} points.")
            if st.button("Save zone for this camera", type="primary"):
                save_zone(cfg, cam, norm)
                db.log_audit(user, "board_zone_saved", reason=cam)
                st.success("Saved. Re-score old sessions to apply it.")
    with st.expander("Enter / clear the zone manually"):
        txt = st.text_input("Points as x,y pairs in 0-1 (e.g. 0.3,0.0; 0.7,0.0; 0.7,0.2; 0.3,0.2)", key="zone_txt")
        c1, c2 = st.columns(2)
        if c1.button("Save typed zone"):
            try:
                pts = [[float(v) for v in p.split(",")] for p in txt.split(";") if p.strip()]
                save_zone(cfg, cam, pts)
                db.log_audit(user, "board_zone_saved", reason=cam)
                st.success("Saved.")
            except ValueError as exc:
                st.error(f"Invalid points: {exc}")
        if c2.button("Clear zone (use facing-camera mode)"):
            save_zone(cfg, cam, None)
            db.log_audit(user, "board_zone_cleared", reason=cam)
            st.success("Cleared.")
    zones = load_zones(cfg)
    if zones:
        st.caption("Saved zones: " + ", ".join(sorted(zones)))


# --------------------------------------------------------------------------- entry
def render(db: Any, cfg: Any, user: str, role: str) -> None:
    st.header("Attentiveness & Behaviour")
    st.warning("⚠ " + DISCLAIMER + " Every event shows its confidence; teachers can mark false detections.")
    tabs_spec = ["Class view", "Student view", "Compare", "Video & downloads", "Corrections"]
    if role == "admin":
        tabs_spec.append("Board zone setup")
    sessions = _sessions_with_results(db)
    if not sessions:
        st.info("No behaviour analyses yet. Process a video with mode **Attendance + Behaviour**.")
        if role == "admin":
            with st.expander("Board zone setup", expanded=True):
                zone_editor(cfg, db, user)
        return
    sess = st.selectbox("Session", sessions, format_func=lambda s: f"#{s['id']} · {s['started_at'][:16]} · "
                        f"{s['class_name']} · {s['subject']}")
    sid = sess["id"]
    result = db.get_behaviour_result(sid)
    forced = bool(cfg.behaviour.privacy.class_level_only) or bool(result.get("class_only"))
    class_only = st.toggle("Class-level only view (aggregate analytics, no per-student data)", value=forced,
                           disabled=forced, help="Use this when the screen is shared, e.g. in staff meetings.")
    if result.get("withheld_no_consent"):
        st.caption(f"{result['withheld_no_consent']} student(s) without guardian consent for behaviour analytics "
                   "are included only in class aggregates.")
    tr, beh, min_score = _filters(result, str(sid))
    subjects = [] if class_only else [s for s in result.get("subjects", [])
                                      if s["score"] is None or s["score"] >= min_score]
    if min_score > 0:
        subjects = [s for s in subjects if s["score"] is not None]
    tabs = st.tabs(tabs_spec)
    with tabs[0]:
        class_view(result, subjects, tr, beh, class_only)
    with tabs[1]:
        if class_only:
            st.info("Hidden in class-level only mode.")
        else:
            student_view(db, cfg, user, sid, result, subjects, tr, beh)
    with tabs[2]:
        compare_view(db, result, subjects, class_only)
    with tabs[3]:
        downloads(result, sid)
    with tabs[4]:
        corrections_view(db, cfg, user, sid)
    if role == "admin":
        with tabs[5]:
            zone_editor(cfg, db, user)


if __name__ == "__main__":  # opened directly as a Streamlit page -> no access without the main app's login
    st.info("Open the ClassSightAI main app (streamlit run app/streamlit_app.py) and log in to view analytics.")
