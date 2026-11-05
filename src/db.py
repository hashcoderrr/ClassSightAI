"""SQLite persistence layer with field-level encryption at rest.

Sensitive fields (student names, guardian contacts and every face embedding)
are encrypted with Fernet (AES-128-CBC + HMAC-SHA256). The key comes from the
``SMARTATTEND_KEY`` environment variable or a key file created on first run.
Raw images are never stored in the database.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    roll_no TEXT PRIMARY KEY,
    name_enc BLOB NOT NULL,
    class_name TEXT NOT NULL,
    embedding_enc BLOB NOT NULL,
    num_images INTEGER NOT NULL DEFAULT 0,
    consent INTEGER NOT NULL DEFAULT 0,
    consent_date TEXT,
    guardian_enc BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL,
    subject TEXT NOT NULL,
    started_at TEXT NOT NULL,
    source TEXT,
    duration_sec REAL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed',
    output_dir TEXT,
    report_path TEXT,
    video_path TEXT,
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS attendance (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    roll_no TEXT NOT NULL,
    status TEXT NOT NULL,
    first_seen REAL,
    last_seen REAL,
    time_visible REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    snapshot_path TEXT,
    remarks TEXT DEFAULT '',
    overridden INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, roll_no)
);
CREATE TABLE IF NOT EXISTS unknown_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    first_seen REAL,
    last_seen REAL,
    thumb_path TEXT,
    embedding_enc BLOB,
    assigned_roll TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    username TEXT NOT NULL,
    action TEXT NOT NULL,
    session_id INTEGER,
    roll_no TEXT,
    old_value TEXT,
    new_value TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    pw_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS behaviour_results (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    data_enc BLOB NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS behaviour_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    subject_key TEXT NOT NULL,
    behaviour TEXT,
    start REAL NOT NULL,
    end REAL NOT NULL,
    corrected_to TEXT NOT NULL,
    username TEXT NOT NULL,
    reason TEXT,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attendance_roll ON attendance(roll_no);
CREATE INDEX IF NOT EXISTS idx_unknown_session ON unknown_faces(session_id);
"""

STATUSES = ("Present", "Late", "Absent", "Excused")
ROLES = ("admin", "teacher", "staff")
ATTENDED_STATUSES = ("Present", "Late")
PBKDF2_ITERATIONS = 200_000


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Crypto:
    """Fernet wrapper used for all encrypted columns."""

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    @classmethod
    def from_env_or_file(cls, key_path: Path) -> "Crypto":
        env_key = os.environ.get("SMARTATTEND_KEY")
        if env_key:
            return cls(env_key.encode())
        if key_path.exists():
            return cls(key_path.read_bytes().strip())
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        log.warning("Generated new encryption key at %s - back it up and keep it out of version "
                    "control (or set SMARTATTEND_KEY).", key_path)
        return cls(key)

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        try:
            return self._fernet.decrypt(token)
        except InvalidToken as exc:
            raise ValueError("Cannot decrypt database field - wrong SMARTATTEND_KEY / key file?") from exc

    def encrypt_str(self, text: str) -> bytes:
        return self.encrypt(text.encode("utf-8"))

    def decrypt_str(self, token: bytes) -> str:
        return self.decrypt(token).decode("utf-8")

    def encrypt_vec(self, vec: np.ndarray) -> bytes:
        return self.encrypt(np.asarray(vec, dtype=np.float32).tobytes())

    def decrypt_vec(self, token: bytes) -> np.ndarray:
        return np.frombuffer(self.decrypt(token), dtype=np.float32).copy()


@dataclass
class Student:
    roll_no: str
    name: str
    class_name: str
    embedding: np.ndarray
    num_images: int = 0
    consent: bool = False
    consent_date: Optional[str] = None
    guardian: str = ""
    behaviour_consent: bool = False


class Database:
    """Thread-safe access to the SmartAttend SQLite database."""

    def __init__(self, db_path: str | Path, key_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.crypto = Crypto.from_env_or_file(Path(key_path))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(students)")}
        if "behaviour_consent" not in cols:
            self._conn.execute("ALTER TABLE students ADD COLUMN behaviour_consent INTEGER NOT NULL DEFAULT 0")
            self._conn.execute("ALTER TABLE students ADD COLUMN behaviour_consent_date TEXT")

    @classmethod
    def from_config(cls, cfg: Any) -> "Database":
        return cls(cfg.path("db_path"), cfg.path("key_path"))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ students
    def upsert_student(self, s: Student) -> None:
        if not s.consent:
            raise PermissionError(f"Consent is required to enroll {s.roll_no} (DPDP Act).")
        emb = np.asarray(s.embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        ts = now_iso()
        with self.tx() as c:
            c.execute(
"""INSERT INTO students (roll_no, name_enc, class_name, embedding_enc, num_images, consent,
                       consent_date, guardian_enc, created_at, updated_at, behaviour_consent, behaviour_consent_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(roll_no) DO UPDATE SET name_enc=excluded.name_enc,
                       class_name=excluded.class_name, embedding_enc=excluded.embedding_enc,
                       num_images=excluded.num_images, consent=excluded.consent,
                       consent_date=excluded.consent_date, guardian_enc=excluded.guardian_enc,
                       updated_at=excluded.updated_at, behaviour_consent=excluded.behaviour_consent,
                       behaviour_consent_date=COALESCE(students.behaviour_consent_date,
                                                       excluded.behaviour_consent_date)""",
                (s.roll_no, self.crypto.encrypt_str(s.name), s.class_name, self.crypto.encrypt_vec(emb),
                 s.num_images, int(s.consent), s.consent_date or ts,
                 self.crypto.encrypt_str(s.guardian or ""), ts, ts, int(s.behaviour_consent),
                 ts if s.behaviour_consent else None))

    def _row_to_student(self, r: sqlite3.Row) -> Student:
        guardian = self.crypto.decrypt_str(r["guardian_enc"]) if r["guardian_enc"] else ""
        return Student(roll_no=r["roll_no"], name=self.crypto.decrypt_str(r["name_enc"]),
                       class_name=r["class_name"], embedding=self.crypto.decrypt_vec(r["embedding_enc"]),
                       num_images=r["num_images"], consent=bool(r["consent"]),
                       consent_date=r["consent_date"], guardian=guardian,
                       behaviour_consent=bool(r["behaviour_consent"]))

    def get_student(self, roll_no: str) -> Optional[Student]:
        rows = self._query("SELECT * FROM students WHERE roll_no=?", (roll_no,))
        return self._row_to_student(rows[0]) if rows else None

    def list_students(self, class_name: Optional[str] = None) -> list[Student]:
        if class_name:
            rows = self._query("SELECT * FROM students WHERE class_name=? ORDER BY roll_no", (class_name,))
        else:
            rows = self._query("SELECT * FROM students ORDER BY roll_no")
        return [self._row_to_student(r) for r in rows]

    def list_classes(self) -> list[str]:
        return [r[0] for r in self._query("SELECT DISTINCT class_name FROM students ORDER BY class_name")]

    def set_behaviour_consent(self, roll_no: str, value: bool, username: str) -> None:
        """Record (or withdraw) guardian consent for behaviour analytics."""
        with self.tx() as c:
            cur = c.execute("UPDATE students SET behaviour_consent=?, behaviour_consent_date=? WHERE roll_no=?",
                            (int(value), now_iso() if value else None, roll_no))
            if cur.rowcount == 0:
                raise ValueError(f"Student {roll_no} not enrolled")
        self.log_audit(username, "behaviour_consent", roll_no=roll_no, new_value=int(value))

    def delete_student(self, roll_no: str) -> None:
        """Right to erasure: remove the student's embedding and attendance rows."""
        with self.tx() as c:
            c.execute("DELETE FROM attendance WHERE roll_no=?", (roll_no,))
            c.execute("UPDATE unknown_faces SET assigned_roll=NULL WHERE assigned_roll=?", (roll_no,))
            c.execute("DELETE FROM students WHERE roll_no=?", (roll_no,))

    # ------------------------------------------------------------------ sessions
    def create_session(self, class_name: str, subject: str, source: str, started_at: Optional[str] = None,
                       output_dir: str = "", meta: Optional[dict] = None) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO sessions (class_name, subject, started_at, source, status, output_dir, meta_json)"
                " VALUES (?,?,?,?,?,?,?)",
                (class_name, subject, started_at or now_iso(), source, "processing", output_dir,
                 json.dumps(meta or {})))
            return int(cur.lastrowid)

    def update_session(self, session_id: int, **fields: Any) -> None:
        allowed = {"duration_sec", "status", "output_dir", "report_path", "video_path", "meta_json"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown session fields: {bad}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE sessions SET {sets} WHERE id=?", (*fields.values(), session_id))

    def get_session(self, session_id: int) -> Optional[dict[str, Any]]:
        rows = self._query("SELECT * FROM sessions WHERE id=?", (session_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
        return d

    def list_sessions(self, class_name: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sessions"
        params: tuple = ()
        if class_name:
            sql += " WHERE class_name=?"
            params = (class_name,)
        out = []
        for r in self._query(sql + " ORDER BY started_at, id", params):
            d = dict(r)
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            out.append(d)
        return out

    def delete_session(self, session_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ---------------------------------------------------------------- attendance
    def save_attendance(self, session_id: int, rows: list[dict[str, Any]]) -> None:
        with self.tx() as c:
            c.executemany(
                """INSERT OR REPLACE INTO attendance (session_id, roll_no, status, first_seen, last_seen,
                       time_visible, confidence, snapshot_path, remarks, overridden)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [(session_id, r["roll_no"], r["status"], r.get("first_seen"), r.get("last_seen"),
                  float(r.get("time_visible") or 0.0), float(r.get("confidence") or 0.0),
                  r.get("snapshot_path"), r.get("remarks", ""), int(r.get("overridden", 0)))
                 for r in rows])

    def get_attendance(self, session_id: int) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM attendance WHERE session_id=? ORDER BY roll_no", (session_id,))
        return [dict(r) for r in rows]

    def set_status(self, session_id: int, roll_no: str, new_status: str, username: str,
                   reason: str = "", **extra: Any) -> bool:
        """Manually override a status; every change is written to the audit log.

        Returns True if something changed.
        """
        if new_status not in STATUSES:
            raise ValueError(f"Invalid status {new_status!r}; expected one of {STATUSES}")
        rows = self._query("SELECT status FROM attendance WHERE session_id=? AND roll_no=?",
                           (session_id, roll_no))
        old = rows[0]["status"] if rows else None
        if old == new_status and not extra:
            return False
        with self.tx() as c:
            if rows:
                c.execute("UPDATE attendance SET status=?, overridden=1 WHERE session_id=? AND roll_no=?",
                          (new_status, session_id, roll_no))
            else:
                c.execute("INSERT INTO attendance (session_id, roll_no, status, overridden) VALUES (?,?,?,1)",
                          (session_id, roll_no, new_status))
            cols = {k: v for k, v in extra.items()
                    if k in {"first_seen", "last_seen", "time_visible", "confidence", "snapshot_path", "remarks"}}
            if cols:
                sets = ", ".join(f"{k}=?" for k in cols)
                c.execute(f"UPDATE attendance SET {sets} WHERE session_id=? AND roll_no=?",
                          (*cols.values(), session_id, roll_no))
            c.execute("INSERT INTO audit_log (ts, username, action, session_id, roll_no, old_value, new_value,"
                      " reason) VALUES (?,?,?,?,?,?,?,?)",
                      (now_iso(), username, "status_override", session_id, roll_no, old, new_status, reason))
        return True

    def log_audit(self, username: str, action: str, session_id: Optional[int] = None,
                  roll_no: Optional[str] = None, old_value: Any = None, new_value: Any = None,
                  reason: str = "") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO audit_log (ts, username, action, session_id, roll_no, old_value, new_value,"
                      " reason) VALUES (?,?,?,?,?,?,?,?)",
                      (now_iso(), username, action, session_id, roll_no,
                       None if old_value is None else str(old_value),
                       None if new_value is None else str(new_value), reason))

    def get_audit_log(self, session_id: Optional[int] = None) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self._query("SELECT * FROM audit_log ORDER BY id DESC")
        else:
            rows = self._query("SELECT * FROM audit_log WHERE session_id=? ORDER BY id DESC", (session_id,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- unknown faces
    def save_unknown(self, session_id: int, cluster_id: int, track_id: int, first_seen: float,
                     last_seen: float, thumb_path: str, embedding: Optional[np.ndarray]) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO unknown_faces (session_id, cluster_id, track_id, first_seen, last_seen, thumb_path,"
                " embedding_enc) VALUES (?,?,?,?,?,?,?)",
                (session_id, cluster_id, track_id, first_seen, last_seen, thumb_path,
                 self.crypto.encrypt_vec(embedding) if embedding is not None else None))
            return int(cur.lastrowid)

    def get_unknowns(self, session_id: int, include_embeddings: bool = False) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM unknown_faces WHERE session_id=? ORDER BY cluster_id, track_id",
                           (session_id,))
        out = []
        for r in rows:
            d = dict(r)
            enc = d.pop("embedding_enc")
            if include_embeddings:
                d["embedding"] = self.crypto.decrypt_vec(enc) if enc else None
            out.append(d)
        return out

    def assign_unknown_cluster(self, session_id: int, cluster_id: int, roll_no: str) -> int:
        with self.tx() as c:
            cur = c.execute("UPDATE unknown_faces SET assigned_roll=? WHERE session_id=? AND cluster_id=?",
                            (roll_no, session_id, cluster_id))
            return cur.rowcount

    # -------------------------------------------------------------------- users
    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS).hex()

    def has_users(self) -> bool:
        return bool(self._query("SELECT 1 FROM users LIMIT 1"))

    def create_user(self, username: str, password: str, role: str = "admin") -> None:
        if role not in ROLES:
            raise ValueError(f"Role must be one of {ROLES}")
        if not username.strip():
            raise ValueError("Username required.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        salt = secrets.token_hex(16)
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO users (username, pw_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
                      (username, self._hash_password(password, salt), salt, role, now_iso()))

    def ensure_default_admin(self, cfg: Any) -> Optional[str]:
        """Create the built-in admin from ``auth`` in config.yaml (or restore its password/role).

        Returns the username, or None when no default admin is configured.
        """
        acfg = cfg.get("auth") or {}
        username = str(acfg.get("default_admin_username") or "").strip()
        password = str(acfg.get("default_admin_password") or "")
        if not username or not password:
            return None
        exists = self.get_role(username) is not None
        if not exists:
            self.create_user(username, password, "admin")
            log.info("Default admin account '%s' created.", username)
        elif bool(acfg.get("enforce_default_admin", True)) and (
                self.get_role(username) != "admin" or not self.verify_user(username, password)):
            self.create_user(username, password, "admin")
            log.info("Default admin account '%s' password/role restored from config.yaml.", username)
        return username

    def get_role(self, username: str) -> Optional[str]:
        rows = self._query("SELECT role FROM users WHERE username=?", (username,))
        return rows[0]["role"] if rows else None

    def list_users(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query("SELECT username, role, created_at FROM users ORDER BY username")]

    def delete_user(self, username: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM users WHERE username=?", (username,))

    # --------------------------------------------------------------- behaviour
    def save_behaviour_result(self, session_id: int, result: dict[str, Any]) -> None:
        blob = self.crypto.encrypt(json.dumps(result, default=str).encode("utf-8"))
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO behaviour_results (session_id, data_enc, updated_at) VALUES (?,?,?)",
                      (session_id, blob, now_iso()))

    def get_behaviour_result(self, session_id: int) -> Optional[dict[str, Any]]:
        rows = self._query("SELECT data_enc FROM behaviour_results WHERE session_id=?", (session_id,))
        return json.loads(self.crypto.decrypt(rows[0]["data_enc"])) if rows else None

    def behaviour_session_ids(self) -> list[int]:
        return [r[0] for r in self._query("SELECT session_id FROM behaviour_results ORDER BY session_id")]

    def add_behaviour_correction(self, session_id: int, subject_key: str, behaviour: Optional[str], start: float,
                                 end: float, corrected_to: str, username: str, reason: str = "") -> int:
        if end < start:
            raise ValueError("end must be >= start")
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO behaviour_corrections (session_id, subject_key, behaviour, start, end, corrected_to,"
                " username, reason, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, subject_key, behaviour, float(start), float(end), corrected_to, username, reason,
                 now_iso()))
            cid = int(cur.lastrowid)
        self.log_audit(username, "behaviour_correction", session_id, subject_key, behaviour, corrected_to,
                       f"{start:.1f}-{end:.1f}s {reason}".strip())
        return cid

    def get_behaviour_corrections(self, session_id: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query("SELECT * FROM behaviour_corrections WHERE session_id=? ORDER BY id",
                                             (session_id,))]

    def delete_behaviour_correction(self, correction_id: int, username: str) -> None:
        rows = self._query("SELECT * FROM behaviour_corrections WHERE id=?", (correction_id,))
        if not rows:
            return
        with self.tx() as c:
            c.execute("DELETE FROM behaviour_corrections WHERE id=?", (correction_id,))
        r = rows[0]
        self.log_audit(username, "behaviour_correction_undo", r["session_id"], r["subject_key"], r["corrected_to"],
                       r["behaviour"])

    def verify_user(self, username: str, password: str) -> bool:
        rows = self._query("SELECT pw_hash, salt FROM users WHERE username=?", (username,))
        if not rows:
            # constant-ish time to avoid user enumeration
            self._hash_password(password, "00" * 16)
            return False
        return hmac.compare_digest(rows[0]["pw_hash"], self._hash_password(password, rows[0]["salt"]))

    # ------------------------------------------------------------------ history
    def history(self, class_name: Optional[str] = None) -> list[dict[str, Any]]:
        """Cumulative attendance per student across completed sessions."""
        students = {s.roll_no: s for s in self.list_students(class_name)}
        sessions = [s for s in self.list_sessions(class_name) if s["status"] in ("completed", "interrupted")]
        sids = [s["id"] for s in sessions]
        totals: dict[str, dict[str, Any]] = {}
        for roll, s in students.items():
            totals[roll] = {"roll_no": roll, "name": s.name, "class_name": s.class_name,
                            "sessions": 0, "attended": 0, "late": 0, "absent": 0, "excused": 0}
        if sids:
            ph = ",".join("?" * len(sids))
            rows = self._query(f"SELECT roll_no, status FROM attendance WHERE session_id IN ({ph})", tuple(sids))
            for r in rows:
                t = totals.get(r["roll_no"])
                if t is None:
                    continue
                t["sessions"] += 1
                if r["status"] in ATTENDED_STATUSES:
                    t["attended"] += 1
                if r["status"] == "Late":
                    t["late"] += 1
                elif r["status"] == "Absent":
                    t["absent"] += 1
                elif r["status"] == "Excused":
                    t["excused"] += 1
        for t in totals.values():
            counted = t["sessions"] - t["excused"]
            t["percentage"] = round(100.0 * t["attended"] / counted, 1) if counted > 0 else 0.0
        return list(totals.values())
