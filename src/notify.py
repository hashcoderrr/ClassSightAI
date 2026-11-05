"""Optional e-mail / WhatsApp notifications.

Privacy: by default the e-mailed report is regenerated *without* face
thumbnails, and WhatsApp messages carry text only. No face data or embeddings
are ever sent to a third-party API.
"""
from __future__ import annotations

import logging
import os
import smtplib
import tempfile
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def _summary_text(db: Any, session_id: int) -> tuple[str, list[dict[str, Any]]]:
    sess = db.get_session(session_id)
    att = db.get_attendance(session_id)
    present = sum(1 for a in att if a["status"] in ("Present", "Late"))
    absent = [a for a in att if a["status"] == "Absent"]
    text = (f"SmartAttend - {sess['class_name']} / {sess['subject']} ({sess['started_at']})\n"
            f"Present: {present} / {len(att)}\nAbsent: {', '.join(a['roll_no'] for a in absent) or 'none'}")
    return text, absent


def send_email(cfg: Any, to: str, subject: str, body: str, attachments: Optional[list[Path]] = None) -> None:
    ecfg = cfg.notify.email
    password = os.environ.get(str(ecfg.password_env), "")
    msg = EmailMessage()
    msg["From"] = ecfg.sender or ecfg.username
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    for p in attachments or []:
        msg.add_attachment(Path(p).read_bytes(), maintype="application",
                           subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=Path(p).name)
    with smtplib.SMTP(ecfg.smtp_host, int(ecfg.smtp_port), timeout=30) as s:
        if ecfg.use_tls:
            s.starttls()
        if ecfg.username:
            s.login(ecfg.username, password)
        s.send_message(msg)
    log.info("E-mail sent to %s", to)


def send_whatsapp(cfg: Any, to: str, body: str) -> None:
    """Text-only WhatsApp message via Twilio's REST API."""
    import urllib.parse
    import urllib.request
    import base64
    wcfg = cfg.notify.whatsapp
    token = os.environ.get(str(wcfg.auth_token_env), "")
    url = f"https://api.twilio.com/2010-04-01/Accounts/{wcfg.account_sid}/Messages.json"
    data = urllib.parse.urlencode({"From": wcfg.from_number, "To": to, "Body": body}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{wcfg.account_sid}:{token}".encode()).decode())
    with urllib.request.urlopen(req, timeout=30) as resp:
        log.info("WhatsApp sent to %s (HTTP %s)", to, resp.status)


def notify_session(db: Any, cfg: Any, session_id: int) -> list[str]:
    """Send configured notifications for a session. Returns a list of status messages."""
    from .report_excel import generate_report
    msgs: list[str] = []
    text, absent = _summary_text(db, session_id)
    ecfg, wcfg = cfg.notify.email, cfg.notify.whatsapp
    if ecfg.enabled and ecfg.teacher_email:
        try:
            if ecfg.attach_snapshots:
                report = Path(db.get_session(session_id)["report_path"])
            else:
                tmp = Path(tempfile.mkdtemp()) / "Attendance_report_no_faces.xlsx"
                report = generate_report(db, session_id, cfg, embed_images=False, out_path=tmp, record=False)
            send_email(cfg, ecfg.teacher_email, "Attendance report", text, [report])
            msgs.append(f"E-mail sent to {ecfg.teacher_email}")
        except Exception as exc:
            log.exception("E-mail failed")
            msgs.append(f"E-mail failed: {exc}")
        if ecfg.parent_alerts:
            for a in absent:
                s = db.get_student(a["roll_no"])
                if s and s.guardian and "@" in s.guardian:
                    try:
                        send_email(cfg, s.guardian, f"Absence alert: {s.name}",
                                   f"Dear parent/guardian, {s.name} ({s.roll_no}) was marked absent in "
                                   f"{db.get_session(session_id)['subject']} today.")
                        msgs.append(f"Absence alert sent for {s.roll_no}")
                    except Exception as exc:
                        msgs.append(f"Absence alert for {s.roll_no} failed: {exc}")
    if wcfg.enabled and wcfg.teacher_number:
        try:
            send_whatsapp(cfg, wcfg.teacher_number, text)
            msgs.append(f"WhatsApp sent to {wcfg.teacher_number}")
        except Exception as exc:
            log.exception("WhatsApp failed")
            msgs.append(f"WhatsApp failed: {exc}")
    if not msgs:
        msgs.append("No notification channel enabled in config.yaml.")
    return msgs
