"""ClassSightAI command-line interface.

Examples:
    python main.py --video class.mp4 --class 10A --subject Maths
    python main.py process --source rtsp://user:pass@cam/stream --class 10A --subject Physics --max-minutes 50
    python main.py process --source 0 --class 10A --subject Demo           # webcam
    python main.py enroll --roll 23 --name "Aditya Kumar" --class 10A --images a.jpg b.jpg c.jpg --consent
    python main.py enroll-bulk --dataset dataset --class 10A --consent
    python main.py list-students --class 10A
    python main.py unknowns --session 3
    python main.py assign-unknown --session 3 --cluster 2 --roll 23 [--update-gallery]
    python main.py override --session 3 --roll 23 --status Present --reason "Medical certificate"
    python main.py report --session 3
    python main.py notify --session 3
    python main.py create-admin --username admin
    python main.py cleanup
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys

from src.config import load_config, setup_logging

COMMANDS = {"process", "enroll", "enroll-bulk", "list-students", "delete-student", "unknowns", "assign-unknown",
            "override", "report", "notify", "create-admin", "cleanup", "sessions", "set-behaviour-consent",
            "behaviour-rescore", "behaviour-correct", "behaviour-summary"}
log = logging.getLogger("ClassSightAI")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ClassSightAI", description="AI classroom attendance from video / CCTV")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("process", help="Process a video file / RTSP stream / webcam")
    src = pr.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Video file (mp4/avi/mov/...)")
    src.add_argument("--source", help="RTSP/HTTP URL or webcam index")
    pr.add_argument("--class", dest="class_name", required=True)
    pr.add_argument("--subject", required=True)
    pr.add_argument("--frame-skip", type=int, help="Override video.frame_skip")
    pr.add_argument("--threshold", type=float, help="Override recognition.similarity_threshold")
    pr.add_argument("--tiled", action="store_true", help="Enable tiled detection for wide/4K cameras")
    pr.add_argument("--upscale", type=float, help="Upscale factor for small faces")
    pr.add_argument("--max-minutes", type=float, help="Stop a live stream after N minutes")
    pr.add_argument("--no-video", action="store_true", help="Do not write the annotated video")
    pr.add_argument("--notify", action="store_true", help="Send configured notifications afterwards")
    pr.add_argument("--mode", default="attendance", choices=["attendance", "attendance+behaviour", "behaviour"],
                    help="attendance+behaviour also runs Class Attentiveness & Behaviour Analytics")
    pr.add_argument("--camera", default="default", help="Camera / room id (selects the board zone)")
    pr.add_argument("--class-level-only", action="store_true", help="Behaviour privacy mode: aggregates only")

    en = sub.add_parser("enroll", help="Enroll one student")
    en.add_argument("--roll", required=True)
    en.add_argument("--name", required=True)
    en.add_argument("--class", dest="class_name", required=True)
    en.add_argument("--images", nargs="*", default=[])
    en.add_argument("--video", help="Short selfie video instead of / in addition to photos")
    en.add_argument("--guardian", default="", help="Guardian e-mail/phone (optional)")
    en.add_argument("--consent", action="store_true", help="Confirm informed consent was obtained (required)")
    en.add_argument("--behaviour-consent", action="store_true",
                    help="Guardian also consented to behaviour analytics (otherwise aggregate-only)")

    eb = sub.add_parser("enroll-bulk", help="Enroll from dataset/<RollNo>_<Name>/*.jpg")
    eb.add_argument("--dataset", required=True)
    eb.add_argument("--class", dest="class_name")
    eb.add_argument("--consent", action="store_true", help="Confirm consent was obtained for all students")
    eb.add_argument("--behaviour-consent", action="store_true",
                    help="Guardians of all students also consented to behaviour analytics")

    bc = sub.add_parser("set-behaviour-consent", help="Record / withdraw guardian consent for behaviour analytics")
    bc.add_argument("--roll", required=True)
    bc.add_argument("--value", choices=["yes", "no"], required=True)
    bc.add_argument("--user", default="cli")

    br = sub.add_parser("behaviour-rescore", help="Recompute behaviour scores from cached signals (new weights)")
    br.add_argument("--session", type=int, required=True)
    br.add_argument("--user", default="cli")
    bx = sub.add_parser("behaviour-correct", help="Mark a detected behaviour as wrong (re-scores, audited)")
    bx.add_argument("--session", type=int, required=True)
    bx.add_argument("--subject", required=True, help="Roll No or Unknown_<id>")
    bx.add_argument("--start", type=float, required=True, help="seconds")
    bx.add_argument("--end", type=float, required=True, help="seconds")
    bx.add_argument("--behaviour", help="Only replace this detected behaviour (default: any)")
    bx.add_argument("--to", dest="corrected_to", default="other", help="Correct behaviour (default: other)")
    bx.add_argument("--reason", default="")
    bx.add_argument("--user", default="cli")
    bs = sub.add_parser("behaviour-summary", help="Print the behaviour summary of a session")
    bs.add_argument("--session", type=int, required=True)

    ls = sub.add_parser("list-students")
    ls.add_argument("--class", dest="class_name")
    ds = sub.add_parser("delete-student", help="Erase a student's biometric data (right to erasure)")
    ds.add_argument("--roll", required=True)
    se = sub.add_parser("sessions")
    se.add_argument("--class", dest="class_name")

    uk = sub.add_parser("unknowns", help="List unknown-face clusters of a session")
    uk.add_argument("--session", type=int, required=True)
    au = sub.add_parser("assign-unknown", help="Assign an unknown cluster to a student")
    au.add_argument("--session", type=int, required=True)
    au.add_argument("--cluster", type=int, required=True)
    au.add_argument("--roll", required=True)
    au.add_argument("--update-gallery", action="store_true", help="Also add these faces to the student's embedding")
    au.add_argument("--user", default="cli")

    ov = sub.add_parser("override", help="Manually change a student's status (audited)")
    ov.add_argument("--session", type=int, required=True)
    ov.add_argument("--roll", required=True)
    ov.add_argument("--status", required=True, choices=["Present", "Late", "Absent", "Excused"])
    ov.add_argument("--reason", default="")
    ov.add_argument("--user", default="cli")

    rp = sub.add_parser("report", help="Regenerate the Excel/CSV/JSON report")
    rp.add_argument("--session", type=int, required=True)
    rp.add_argument("--no-images", action="store_true")
    nt = sub.add_parser("notify", help="E-mail / WhatsApp the report")
    nt.add_argument("--session", type=int, required=True)

    ca = sub.add_parser("create-admin", help="Create/reset a web-app account (default role admin)")
    ca.add_argument("--username", default="admin")
    ca.add_argument("--role", default="admin", choices=["admin", "teacher", "staff"])
    sub.add_parser("cleanup", help="Apply data-retention policy now")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # allow the short form: python main.py --video x.mp4 --class 10A --subject Maths
    cfg_args: list[str] = []
    if argv and argv[0] == "--config" and len(argv) >= 2:
        cfg_args, argv = argv[:2], argv[2:]
    if argv and argv[0] not in COMMANDS and any(a in ("--video", "--source") for a in argv):
        argv = ["process"] + argv
    argv = cfg_args + argv
    args = build_parser().parse_args(argv)

    overrides: dict = {}
    if args.cmd == "process":
        v = overrides.setdefault("video", {})
        if args.frame_skip:
            v["frame_skip"] = args.frame_skip
        if args.tiled:
            v["tiled_detection"] = True
        if args.upscale:
            v["upscale_factor"] = args.upscale
        if args.threshold is not None:
            overrides["recognition"] = {"similarity_threshold": args.threshold}
        if args.no_video:
            overrides["annotate"] = {"enabled": False}
            overrides["behaviour"] = {"video": {"enabled": False}}
        if args.class_level_only:
            overrides.setdefault("behaviour", {})["privacy"] = {"class_level_only": True}
    cfg = load_config(args.config, overrides)
    setup_logging(cfg)

    from src.db import Database
    from src.privacy import cleanup_retention
    db = Database.from_config(cfg)
    try:
        db.ensure_default_admin(cfg)   # built-in login: see auth.* in config.yaml (default admin / admin123)
        cleanup_retention(cfg)
        return dispatch(args, cfg, db)
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        log.error("%s", exc)
        return 2
    finally:
        db.close()


def _print_behaviour(r: dict) -> None:
    print(f"Behaviour    : class attentiveness {r.get('class_score')} ({r.get('class_label')}), "
          f"{r.get('students_analysed')} students analysed, {len(r.get('events', []))} events, "
          f"{len(r.get('drop_points', []))} engagement drop points")
    for s in sorted([s for s in r.get("subjects", []) if s["score"] is not None], key=lambda s: -s["score"]):
        print(f"   {s['key']:>12}  {s['name'][:24]:<24} {s['score']:6.1f}  {s['label']}")
    files = r.get("files") or {}
    print(f"Behaviour report: {files.get('excel')}")
    print(f"Behaviour video : {files.get('video_full')}  (preview: {files.get('video_preview')})")
    print("NOTE: indicative estimates of observable behaviour only - not for disciplinary use.")


def dispatch(args: argparse.Namespace, cfg, db) -> int:  # noqa: C901 - simple command switch
    if args.cmd == "process":
        from src.pipeline import AttendanceProcessor
        last = {"pct": -1}

        def progress(p: dict) -> None:
            pct = int(p["fraction"] * 100)
            if p["total_frames"] and pct != last["pct"] and pct % 5 == 0:
                last["pct"] = pct
                eta = f"{p['eta_sec']:.0f}s" if p["eta_sec"] is not None else "?"
                print(f"\r[{pct:3d}%] present {p['present']}/{p['total']}  ETA {eta}   ", end="", flush=True)
        source = args.video if args.video else args.source
        proc = AttendanceProcessor(cfg, db)
        res = proc.run(source, args.class_name, args.subject, progress_cb=progress,
                       max_seconds=args.max_minutes * 60 if args.max_minutes else None,
                       behaviour=args.mode != "attendance", camera_id=args.camera)
        print()
        present = sum(1 for r in res.rows if r["status"] in ("Present", "Late"))
        print(f"Session {res.session_id} ({res.status}): {present}/{len(res.rows)} present, "
              f"{res.unknown_count} unknown face tracks, {res.conflicts} duplicate flags")
        print(f"Excel report : {res.report_path}")
        print(f"Annotated    : {res.video_path}")
        if res.behaviour:
            _print_behaviour(res.behaviour)
        elif args.mode != "attendance":
            print("Behaviour analysis failed - see the log.")
        print(f"Output folder: {res.output_dir}")
        if args.notify:
            from src.notify import notify_session
            for m in notify_session(db, cfg, res.session_id):
                print(m)
        return 0

    if args.cmd in ("enroll", "enroll-bulk"):
        from src.detector import FaceEngine
        from src.enroll import Enroller
        enroller = Enroller(cfg, db, FaceEngine(cfg))
        if args.cmd == "enroll":
            results = [enroller.enroll(args.roll, args.name, args.class_name, images=args.images, video=args.video,
                                       consent=args.consent, guardian=args.guardian,
                                       behaviour_consent=True if args.behaviour_consent else None)]
        else:
            if not args.consent:
                raise PermissionError("Pass --consent to confirm consent was obtained for every student (DPDP Act).")
            results = enroller.enroll_folder(args.dataset, args.class_name, consent=True,
                                             behaviour_consent=True if args.behaviour_consent else None)
        ok = 0
        for r in results:
            print(("[OK]   " if r.success else "[FAIL] ") + f"{r.roll_no}: {r.message}")
            for w in r.warnings:
                print(f"         warning: {w}")
            for rej in r.rejected:
                print(f"         rejected: {rej}")
            ok += r.success
        print(f"{ok}/{len(results)} students enrolled.")
        return 0 if ok == len(results) else 1

    if args.cmd == "list-students":
        for s in db.list_students(args.class_name):
            print(f"{s.roll_no:>8}  {s.name:<30} {s.class_name:<8} images={s.num_images} consent={s.consent_date}")
        return 0
    if args.cmd == "delete-student":
        db.delete_student(args.roll)
        db.log_audit("cli", "delete_student", roll_no=args.roll)
        print(f"Deleted {args.roll}")
        return 0
    if args.cmd == "sessions":
        for s in db.list_sessions(args.class_name):
            print(f"#{s['id']:<4} {s['started_at']}  {s['class_name']:<6} {s['subject']:<12} {s['status']:<11} "
                  f"{s['report_path'] or ''}")
        return 0
    if args.cmd == "unknowns":
        for u in db.get_unknowns(args.session):
            print(f"cluster {u['cluster_id']:>3}  track {u['track_id']:>5}  {u['first_seen']:.1f}-{u['last_seen']:.1f}s"
                  f"  assigned={u['assigned_roll'] or '-'}  {u['thumb_path']}")
        return 0
    if args.cmd == "assign-unknown":
        from src.pipeline import assign_unknown
        path = assign_unknown(db, cfg, args.session, args.cluster, args.roll, args.user, args.update_gallery)
        print(f"Cluster {args.cluster} -> {args.roll}. Report regenerated: {path}")
        return 0
    if args.cmd == "override":
        from src.report_excel import generate_report
        changed = db.set_status(args.session, args.roll, args.status, args.user, args.reason)
        print("Status updated." if changed else "No change.")
        print(f"Report: {generate_report(db, args.session, cfg)}")
        return 0
    if args.cmd == "report":
        from src.report_excel import generate_report
        print(generate_report(db, args.session, cfg, embed_images=False if args.no_images else None))
        return 0
    if args.cmd == "notify":
        from src.notify import notify_session
        for m in notify_session(db, cfg, args.session):
            print(m)
        return 0
    if args.cmd == "create-admin":
        pw = getpass.getpass("New password (min 8 chars): ")
        if pw != getpass.getpass("Repeat password: "):
            raise ValueError("Passwords do not match")
        db.create_user(args.username, pw, args.role)
        print(f"User '{args.username}' ({args.role}) saved.")
        return 0
    if args.cmd == "set-behaviour-consent":
        db.set_behaviour_consent(args.roll, args.value == "yes", args.user)
        print(f"Behaviour-analytics consent for {args.roll}: {args.value}")
        return 0
    if args.cmd in ("behaviour-rescore", "behaviour-correct"):
        from src.behaviour.analyzer import rescore_session
        if args.cmd == "behaviour-correct":
            db.add_behaviour_correction(args.session, args.subject, args.behaviour, args.start, args.end,
                                        args.corrected_to, args.user, args.reason)
        _print_behaviour(rescore_session(db, cfg, args.session, args.user))
        return 0
    if args.cmd == "behaviour-summary":
        r = db.get_behaviour_result(args.session)
        if r is None:
            raise ValueError(f"No behaviour analysis for session {args.session}")
        _print_behaviour(r)
        return 0
    if args.cmd == "cleanup":
        from src.privacy import cleanup_retention
        print(json.dumps(cleanup_retention(cfg)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
