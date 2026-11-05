"""ClassSightAI web app (Streamlit).

Run:  streamlit run app/streamlit_app.py
Pages: Enroll Students | Upload Video / Connect CCTV | Live Processing View | Results & Download | History
"""
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, setup_logging  # noqa: E402
from src.db import STATUSES, Database  # noqa: E402
from src.privacy import cleanup_retention  # noqa: E402
from src.report_excel import fmt_ts, generate_report  # noqa: E402

PAGES = ["Enroll Students", "Upload Video / Connect CCTV", "Live Processing View", "Results & Download",
         "History & Analytics", "Attentiveness & Behaviour", "Admin"]

st.set_page_config(page_title="ClassSightAI", page_icon="🎓", layout="wide")


# ------------------------------------------------------------------ resources
@st.cache_resource
def get_cfg():
    cfg = load_config()
    setup_logging(cfg)
    cleanup_retention(cfg)
    return cfg


@st.cache_resource
def get_db() -> Database:
    d = Database.from_config(get_cfg())
    d.ensure_default_admin(get_cfg())   # built-in admin login, no registration needed
    return d


@st.cache_resource(show_spinner="Loading face models (first run downloads ~280 MB)...")
def get_engine():
    from src.detector import FaceEngine
    return FaceEngine(get_cfg())


cfg = get_cfg()
db = get_db()


# ------------------------------------------------------------------ auth
def auth_gate() -> None:
    if st.session_state.get("user"):
        return
    st.title("🎓 ClassSightAI")
    st.caption("AI-based classroom attendance from video / CCTV - all processing runs locally.")
    default_user = str((cfg.get("auth") or {}).get("default_admin_username") or "")
    if default_user:
        st.info(f"Admin login: **{default_user}** / password from `auth.default_admin_password` in config.yaml "
                "(default `admin123`).")
    with st.form("login"):
        u = st.text_input("Username", value=default_user)
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            if db.verify_user(u.strip(), p):
                st.session_state.user = u.strip()
                st.session_state.role = db.get_role(u.strip()) or "staff"
                db.log_audit(u.strip(), "login")
                st.rerun()
            else:
                time.sleep(1)
                st.error("Invalid username or password.")
    st.stop()


auth_gate()
USER: str = st.session_state.user
ROLE: str = st.session_state.get("role") or db.get_role(USER) or "staff"
CAN_ANALYTICS = ROLE in list(cfg.behaviour.allowed_roles)
VISIBLE = [p for p in PAGES if (p != "Attentiveness & Behaviour" or CAN_ANALYTICS) and (p != "Admin" or ROLE == "admin")]

with st.sidebar:
    st.title("🎓 ClassSight")
    st.caption(f"Logged in as **{USER}** ({ROLE})")
    if "_nav" in st.session_state:  # programmatic navigation must happen before the widget exists
        st.session_state.page = st.session_state.pop("_nav")
    if st.session_state.get("page") not in VISIBLE:
        st.session_state.page = VISIBLE[0]
    page = st.radio("Navigate", VISIBLE, key="page")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.caption("🔒 Faces never leave this machine. Embeddings are encrypted at rest.")


def goto(p: str) -> None:
    st.session_state._nav = p


def classes() -> list[str]:
    return db.list_classes()


# ------------------------------------------------------------------ page 1
def page_enroll() -> None:
    st.header("Enroll Students")
    tab1, tab2, tab3 = st.tabs(["Single student", "Bulk from folder", "Enrolled students"])
    with tab1:
        with st.form("enroll", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            roll = c1.text_input("Roll No *")
            name = c2.text_input("Name *")
            klass = c3.text_input("Class / Section *", value=str(cfg.enrollment.default_class))
            guardian = st.text_input("Guardian e-mail / phone (optional, for absence alerts)")
            photos = st.file_uploader("3-10 face photos", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                      accept_multiple_files=True)
            video = st.file_uploader("...or a short selfie video (optional)", type=["mp4", "mov", "avi", "webm"])
            st.markdown("**Consent (Digital Personal Data Protection Act, 2023)** - the student / guardian has been "
                        "informed that a face template (not the photo) will be stored, encrypted, and used only "
                        "for attendance, and that they may withdraw consent and request erasure at any time.")
            consent = st.checkbox("Informed consent obtained *")
            bconsent_one = st.checkbox("Guardian also consents to classroom behaviour analytics (optional - "
                                       "without it the student appears only in class-level aggregates)")
            submitted = st.form_submit_button("Enroll")
        if submitted:
            if not consent:
                st.error("Consent is mandatory.")
                return
            vpath = None
            if video is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.name).suffix)
                tmp.write(video.getvalue())
                tmp.close()
                vpath = tmp.name
            from src.enroll import Enroller
            with st.spinner("Detecting and embedding faces..."):
                res = Enroller(cfg, db, get_engine()).enroll(
                    roll, name, klass, images=[f.getvalue() for f in photos or []],
                    labels=[f.name for f in photos or []], video=vpath, consent=consent, guardian=guardian,
                    behaviour_consent=bconsent_one)
            if vpath:
                Path(vpath).unlink(missing_ok=True)
            (st.success if res.success else st.error)(res.message)
            for w in res.warnings:
                st.warning(w)
            for r in res.rejected:
                st.caption(f"❌ {r}")
            if res.success:
                db.log_audit(USER, "enroll", roll_no=roll, new_value=f"{res.accepted} images")
    with tab2:
        st.write("Folder layout: `dataset/<RollNo>_<Name>/*.jpg` (a selfie video per folder is also accepted).")
        folder = st.text_input("Dataset folder path (on this machine)", value=str(ROOT / "dataset"))
        bclass = st.text_input("Class for all students", value=str(cfg.enrollment.default_class), key="bclass")
        bconsent = st.checkbox("I confirm informed consent was obtained for every student in this folder")
        bbeh = st.checkbox("Guardians of all these students also consented to behaviour analytics")
        if st.button("Run bulk enrollment", disabled=not bconsent):
            from src.enroll import Enroller
            try:
                with st.spinner("Enrolling..."):
                    results = Enroller(cfg, db, get_engine()).enroll_folder(folder, bclass, consent=True,
                                                                             behaviour_consent=True if bbeh else None)
            except NotADirectoryError as exc:
                st.error(str(exc))
                return
            ok = sum(r.success for r in results)
            st.success(f"{ok}/{len(results)} students enrolled.")
            db.log_audit(USER, "enroll_bulk", new_value=f"{ok}/{len(results)}", reason=folder)
            st.dataframe(pd.DataFrame([{"Roll No": r.roll_no, "OK": r.success, "Images used": r.accepted,
                                        "Message": r.message, "Warnings": "; ".join(r.warnings),
                                        "Rejected": "; ".join(r.rejected)} for r in results]),
                         width="stretch")
    with tab3:
        fc = st.selectbox("Class", ["All"] + classes(), key="list_class")
        studs = db.list_students(None if fc == "All" else fc)
        st.dataframe(pd.DataFrame([{"Roll No": s.roll_no, "Name": s.name, "Class": s.class_name,
                                    "Images": s.num_images, "Consent date": s.consent_date,
                                    "Behaviour consent": s.behaviour_consent} for s in studs]),
                     width="stretch", hide_index=True)
        if studs:
            st.subheader("Behaviour-analytics consent")
            bc1, bc2 = st.columns([2, 1])
            who = bc1.selectbox("Student", [s.roll_no for s in studs], key="bc_roll")
            cur = next(s.behaviour_consent for s in studs if s.roll_no == who)
            if bc2.button("Withdraw consent" if cur else "Record guardian consent", key="bc_btn"):
                db.set_behaviour_consent(who, not cur, USER)
                st.rerun()
            st.subheader("Right to erasure")
            victim = st.selectbox("Delete a student's biometric data", [s.roll_no for s in studs])
            if st.button("Delete permanently", type="secondary"):
                db.delete_student(victim)
                db.log_audit(USER, "delete_student", roll_no=victim)
                st.success(f"Deleted {victim}.")
                st.rerun()


# ------------------------------------------------------------------ page 2
def page_source() -> None:
    st.header("Upload Video / Connect CCTV")
    cls = classes()
    if not cls:
        st.warning("Enroll students first.")
        return
    c1, c2 = st.columns(2)
    klass = c1.selectbox("Class", cls)
    subject = c2.text_input("Subject / session", value="Maths")
    kind = st.radio("Source", ["Upload video file", "RTSP / CCTV stream", "Webcam"], horizontal=True)
    m1, m2 = st.columns(2)
    modes = ["Attendance", "Attendance + Behaviour"] if CAN_ANALYTICS else ["Attendance"]
    mode = m1.radio("Mode", modes, horizontal=True)
    camera = m2.text_input("Camera / room id (selects the board zone)", value="default")
    upload = url = None
    cam = 0
    if kind == "Upload video file":
        upload = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
    elif kind == "RTSP / CCTV stream":
        url = st.text_input("Stream URL", placeholder="rtsp://user:pass@192.168.1.20:554/stream1")
    else:
        cam = st.number_input("Webcam index", 0, 10, 0)
    with st.expander("Processing options"):
        o1, o2, o3 = st.columns(3)
        skip = o1.number_input("Process every Nth frame", 1, 60, int(cfg.video.frame_skip))
        thr = o2.slider("Similarity threshold", 0.2, 0.8, float(cfg.recognition.similarity_threshold), 0.01)
        votes = o3.number_input("Min matched frames (votes)", 1, 50, int(cfg.recognition.min_votes))
        o4, o5, o6 = st.columns(3)
        tiled = o4.checkbox("Tiled detection (4K / wide-angle)", bool(cfg.video.tiled_detection))
        upscale = o5.number_input("Upscale factor", 1.0, 3.0, float(cfg.video.upscale_factor), 0.25)
        max_min = o6.number_input("Max minutes (live streams)", 1, 600, 60)
    if st.button("Start processing ▶", type="primary"):
        if kind == "Upload video file":
            if upload is None:
                st.error("Choose a video file.")
                return
            vdir = cfg.path("videos_dir") if cfg.privacy.retain_raw_video else Path(tempfile.gettempdir())
            vdir.mkdir(parents=True, exist_ok=True)
            dest = vdir / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(upload.name).name}"
            with open(dest, "wb") as fh:
                fh.write(upload.getbuffer())
            source, label = str(dest), upload.name
        elif kind == "RTSP / CCTV stream":
            if not url:
                st.error("Enter the stream URL.")
                return
            source, label = url.strip(), url.split("@")[-1]   # never log credentials
        else:
            source, label = int(cam), f"webcam:{cam}"
        st.session_state.job = {"source": source, "label": label, "class": klass, "subject": subject,
                                "skip": int(skip), "thr": float(thr), "votes": int(votes), "tiled": tiled,
                                "upscale": float(upscale), "max_sec": float(max_min) * 60,
                                "is_upload": kind == "Upload video file",
                                "behaviour": mode == "Attendance + Behaviour", "camera": camera.strip() or "default"}
        goto("Live Processing View")
        st.rerun()


# ------------------------------------------------------------------ page 3
def page_live() -> None:
    st.header("Live Processing View")
    job = st.session_state.get("job")
    if not job:
        st.info("No pending job. Configure one on **Upload Video / Connect CCTV**.")
        if st.session_state.get("last_session_id"):
            st.button("Open last results", on_click=goto, args=("Results & Download",))
        return
    st.write(f"**Class:** {job['class']} · **Subject:** {job['subject']} · **Source:** {job['label']}")
    stream = not job["is_upload"]
    if stream:
        st.caption("Live stream: processing stops after the max duration or when you press Stop. "
                   "Partial results are saved either way.")
    st.button("Stop", key="stop_btn")  # clicking reruns the script; the processor saves partial results
    prog = st.progress(0.0, text="Starting...")
    m1, m2, m3 = st.columns(3)
    present_m, eta_m, time_m = m1.empty(), m2.empty(), m3.empty()
    frame_box = st.empty()

    from src.config import load_config as _load
    from src.pipeline import AttendanceProcessor
    jcfg = _load(overrides={"video": {"frame_skip": job["skip"], "tiled_detection": job["tiled"],
                                      "upscale_factor": job["upscale"]},
                            "recognition": {"similarity_threshold": job["thr"], "min_votes": job["votes"]}})
    engine = get_engine()
    engine.tiled, engine.upscale = job["tiled"], job["upscale"]
    st.session_state.pop("job")  # consume the job so a rerun does not restart it
    last_draw = {"t": 0.0}

    def on_progress(p: dict) -> None:
        if "sid" not in last_draw:
            last_draw["sid"] = True
            sessions = db.list_sessions(job["class"])
            if sessions:
                st.session_state.last_session_id = sessions[-1]["id"]
        frac = p["fraction"]
        eta = f"{p['eta_sec']:.0f}s" if p["eta_sec"] is not None else "-"
        prog.progress(frac if not stream else min(1.0, p["ts"] / job["max_sec"]),
                      text=f"{frac * 100:.0f}% · ETA {eta}" if not stream else f"Live · {fmt_ts(p['ts'])}")
        present_m.metric("Present", f"{p['present']} / {p['total']}")
        eta_m.metric("ETA", eta)
        time_m.metric("Video time", fmt_ts(p["ts"]))

    def on_frame(frame) -> None:
        now = time.monotonic()
        if now - last_draw["t"] < 0.3:
            return
        last_draw["t"] = now
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        frame_box.image(frame, channels="BGR", width="stretch")

    try:
        res = AttendanceProcessor(jcfg, db, engine=engine).run(
            job["source"], job["class"], job["subject"], progress_cb=on_progress, frame_cb=on_frame,
            source_label=job["label"], max_seconds=job["max_sec"],
            delete_source_after=job["is_upload"] and not cfg.privacy.retain_raw_video,
            behaviour=job.get("behaviour", False), camera_id=job.get("camera", "default"))
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    st.session_state.last_session_id = res.session_id
    prog.progress(1.0, text="Done")
    present = sum(1 for r in res.rows if r["status"] in ("Present", "Late"))
    st.success(f"Session #{res.session_id} {res.status}: {present}/{len(res.rows)} present, "
               f"{res.unknown_count} unknown faces, {res.conflicts} duplicate flags.")
    st.button("Review & download results →", type="primary", on_click=goto, args=("Results & Download",))


# ------------------------------------------------------------------ page 4
def session_label(s: dict) -> str:
    return f"#{s['id']} · {s['started_at'][:16]} · {s['class_name']} · {s['subject']} · {s['status']}"


def page_results() -> None:
    st.header("Results & Download")
    sessions = db.list_sessions()
    if not sessions:
        st.info("No sessions yet.")
        return
    sessions = sessions[::-1]
    ids = [s["id"] for s in sessions]
    default = st.session_state.get("last_session_id")
    idx = ids.index(default) if default in ids else 0
    sess = st.selectbox("Session", sessions, index=idx, format_func=session_label)
    sid = sess["id"]
    students = {s.roll_no: s for s in db.list_students(sess["class_name"])}
    att = {a["roll_no"]: a for a in db.get_attendance(sid)}
    df = pd.DataFrame([{
        "Roll No": roll, "Name": students[roll].name if roll in students else "(deleted)",
        "Status": att.get(roll, {}).get("status", "Absent"),
        "First Seen": fmt_ts(att.get(roll, {}).get("first_seen")),
        "Last Seen": fmt_ts(att.get(roll, {}).get("last_seen")),
        "Visible": fmt_ts(att.get(roll, {}).get("time_visible") or 0),
        "Confidence": round(float(att.get(roll, {}).get("confidence") or 0), 3),
        "Remarks": att.get(roll, {}).get("remarks") or "",
        "Overridden": bool(att.get(roll, {}).get("overridden", 0)),
    } for roll in sorted(set(students) | set(att))])
    present = int(df["Status"].isin(["Present", "Late"]).sum()) if len(df) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Present", f"{present} / {len(df)}")
    c2.metric("Late", int((df["Status"] == "Late").sum()) if len(df) else 0)
    c3.metric("Absent", int((df["Status"] == "Absent").sum()) if len(df) else 0)
    c4.metric("Duration", fmt_ts(sess.get("duration_sec") or 0))

    st.subheader("Attendance (edit Status to override)")
    edited = st.data_editor(df, hide_index=True, width="stretch", key=f"editor_{sid}",
                            disabled=[c for c in df.columns if c != "Status"],
                            column_config={"Status": st.column_config.SelectboxColumn(options=list(STATUSES),
                                                                                      required=True)})
    reason = st.text_input("Reason for override (saved in the audit trail)")
    if st.button("Save overrides & regenerate report"):
        changes = 0
        for _, row in edited.iterrows():
            orig = df.loc[df["Roll No"] == row["Roll No"], "Status"].iloc[0]
            if row["Status"] != orig:
                changes += db.set_status(sid, row["Roll No"], row["Status"], USER, reason)
        generate_report(db, sid, cfg)
        st.success(f"{changes} override(s) saved; report regenerated.")
        st.rerun()

    st.subheader("Downloads")
    sess = db.get_session(sid)
    rpath = Path(sess["report_path"]) if sess.get("report_path") else None
    if rpath is None or not rpath.exists():
        if st.button("Generate report"):
            generate_report(db, sid, cfg)
            st.rerun()
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("⬇ Download Excel", rpath.read_bytes(), rpath.name,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        for col, ext, mime in ((d2, ".csv", "text/csv"), (d3, ".json", "application/json")):
            p = rpath.with_suffix(ext)
            if p.exists():
                col.download_button(f"⬇ {ext[1:].upper()}", p.read_bytes(), p.name, mime)
        vpath = Path(sess["video_path"]) if sess.get("video_path") else None
        if vpath and vpath.exists():
            with open(vpath, "rb") as fh:
                d4.download_button("⬇ Download Annotated Video", fh, f"annotated_session_{sid}.mp4", "video/mp4")
        else:
            d4.caption("No annotated video (disabled or deleted by retention policy).")
    if st.button("📧 Send notifications (teacher / parents)"):
        from src.notify import notify_session
        for m in notify_session(db, cfg, sid):
            st.write(m)

    conflicts = sess.get("meta", {}).get("conflicts") or []
    if conflicts:
        st.subheader("⚠ Possible spoof / duplicate detections")
        st.dataframe(pd.DataFrame(conflicts), hide_index=True, width="stretch")

    st.subheader("Unknown faces")
    unknowns = db.get_unknowns(sid)
    if not unknowns:
        st.caption("No unknown faces in this session.")
    else:
        clusters: dict[int, list[dict]] = {}
        for u in unknowns:
            clusters.setdefault(u["cluster_id"], []).append(u)
        rolls = sorted(students)
        for cid, items in clusters.items():
            with st.container(border=True):
                cols = st.columns([3, 2])
                imgs = [u["thumb_path"] for u in items if u["thumb_path"] and Path(u["thumb_path"]).exists()]
                if imgs:
                    cols[0].image(imgs[:6], width=96)
                assigned = items[0].get("assigned_roll")
                cols[0].caption(f"Cluster {cid} · {len(items)} track(s) · first seen "
                                f"{fmt_ts(min(u['first_seen'] for u in items))}"
                                + (f" · assigned to **{assigned}**" if assigned else ""))
                who = cols[1].selectbox("Assign to student", rolls, key=f"as_{sid}_{cid}",
                                        format_func=lambda r: f"{r} - {students[r].name}")
                upd = cols[1].checkbox("Also improve this student's face template", key=f"up_{sid}_{cid}")
                if cols[1].button("Assign", key=f"ab_{sid}_{cid}"):
                    from src.pipeline import assign_unknown
                    assign_unknown(db, cfg, sid, cid, who, USER, upd)
                    st.success(f"Cluster {cid} assigned to {who}; report regenerated.")
                    st.rerun()

    with st.expander("Audit trail for this session"):
        st.dataframe(pd.DataFrame(db.get_audit_log(sid)), hide_index=True, width="stretch")
    with st.expander("Danger zone"):
        if st.button("Delete this session"):
            db.delete_session(sid)
            db.log_audit(USER, "delete_session", session_id=sid)
            st.session_state.pop("last_session_id", None)
            st.rerun()


# ------------------------------------------------------------------ page 5
def page_history() -> None:
    st.header("History & Analytics")
    cls = classes()
    if not cls:
        st.info("No classes yet.")
        return
    klass = st.selectbox("Class", cls)
    sessions = [s for s in db.list_sessions(klass) if s["status"] in ("completed", "interrupted", "stopped")]
    if not sessions:
        st.info("No completed sessions for this class.")
    else:
        trend = []
        for s in sessions:
            att = db.get_attendance(s["id"])
            counted = [a for a in att if a["status"] != "Excused"]
            pres = sum(1 for a in counted if a["status"] in ("Present", "Late"))
            trend.append({"Session": f"#{s['id']} {s['subject']}", "Date": s["started_at"][:16],
                          "Subject": s["subject"], "Attendance %": round(100 * pres / len(counted), 1)
                          if counted else 0.0})
        tdf = pd.DataFrame(trend)
        st.plotly_chart(px.line(tdf, x="Date", y="Attendance %", color="Subject", markers=True,
                                title="Attendance trend", range_y=[0, 105]), width="stretch")
    thr = float(cfg.report.history_threshold)
    hist = pd.DataFrame(db.history(klass))
    if len(hist):
        hist = hist[hist["sessions"] > 0]
    st.subheader(f"Chronically absent (< {thr:.0f}%)")
    if len(hist):
        low = hist[hist["percentage"] < thr].sort_values("percentage")
        st.dataframe(low.rename(columns={"roll_no": "Roll No", "name": "Name", "percentage": "Attendance %"}),
                     hide_index=True, width="stretch")
        st.subheader("Cumulative attendance per student")
        st.plotly_chart(px.bar(hist.sort_values("percentage"), x="name", y="percentage",
                               color=hist.sort_values("percentage")["percentage"] < thr,
                               color_discrete_map={True: "#d62728", False: "#2ca02c"},
                               labels={"name": "Student", "percentage": "Attendance %", "color": f"< {thr:.0f}%"}),
                        width="stretch")
    st.subheader("Class-wise comparison")
    comp = []
    for c in cls:
        h = [r for r in db.history(c) if r["sessions"] > 0]
        if h:
            comp.append({"Class": c, "Average attendance %": round(sum(r["percentage"] for r in h) / len(h), 1),
                         "Sessions": len(db.list_sessions(c))})
    if comp:
        st.plotly_chart(px.bar(pd.DataFrame(comp), x="Class", y="Average attendance %", text="Average attendance %",
                               range_y=[0, 105]), width="stretch")
    with st.expander("Full audit log"):
        st.dataframe(pd.DataFrame(db.get_audit_log()), hide_index=True, width="stretch")


def page_behaviour() -> None:
    if not CAN_ANALYTICS:  # defence in depth: the page is also hidden from the menu
        st.error("Your role is not authorised to view behaviour analytics.")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("attentiveness_page", ROOT / "app" / "pages" / "attentiveness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.render(db, cfg, USER, ROLE)


def page_admin() -> None:
    st.header("Admin")
    if ROLE != "admin":
        st.error("Admins only.")
        return
    st.subheader("Users & roles")
    st.caption("admin: everything · teacher: attendance + behaviour analytics · staff: attendance only")
    st.dataframe(pd.DataFrame(db.list_users()), hide_index=True, width="stretch")
    with st.form("new_user"):
        c1, c2, c3 = st.columns(3)
        nu = c1.text_input("Username")
        npw = c2.text_input("Password (min 8)", type="password")
        nr = c3.selectbox("Role", ["teacher", "staff", "admin"])
        if st.form_submit_button("Create / reset user"):
            try:
                db.create_user(nu.strip(), npw, nr)
                db.log_audit(USER, "create_user", new_value=f"{nu.strip()}:{nr}")
                st.success(f"Saved {nu.strip()} ({nr}).")
            except ValueError as exc:
                st.error(str(exc))
    others = [u["username"] for u in db.list_users() if u["username"] != USER]
    if others:
        c1, c2 = st.columns([2, 1])
        victim = c1.selectbox("Delete user", others)
        if c2.button("Delete user"):
            db.delete_user(victim)
            db.log_audit(USER, "delete_user", new_value=victim)
            st.rerun()


{"Enroll Students": page_enroll, "Upload Video / Connect CCTV": page_source, "Live Processing View": page_live,
 "Results & Download": page_results, "History & Analytics": page_history,
 "Attentiveness & Behaviour": page_behaviour, "Admin": page_admin}[page]()
