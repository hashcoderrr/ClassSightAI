"""Shared fixtures: an isolated config/DB in a temp dir and a fake face engine."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.db import Database, Student  # noqa: E402
from src.detector import Detection  # noqa: E402


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTATTEND_KEY", Fernet.generate_key().decode())
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data["paths"] = {k: str(tmp_path / Path(v).name) if k not in ("data_dir", "outputs_dir") else str(tmp_path / k)
                     for k, v in data["paths"].items()}
    data["paths"]["videos_dir"] = str(tmp_path / "videos")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(p)


@pytest.fixture()
def db(cfg):
    d = Database.from_config(cfg)
    yield d
    d.close()


@pytest.fixture()
def enrolled(db):
    students = [("1", "Asha Patil"), ("2", "Bilal Khan"), ("3", "Chitra Rao")]
    for i, (roll, name) in enumerate(students):
        db.upsert_student(Student(roll, name, "10A", unit(100 + i), num_images=5, consent=True))
    return db


class FakeEngine:
    """Stands in for FaceEngine: two faces per frame.

    Face at x~100 is student "1"; face at x~400 is an unknown person.
    The aligned 'crop' encodes the landmark x position so embed_aligned can
    return the matching identity embedding.
    """

    def __init__(self, faces=((100, 100), (400, 100))) -> None:
        self.faces = faces

    def detect(self, frame, tiled=None, upscale=None):
        out = []
        for x, y in self.faces:
            bbox = np.array([x, y, x + 80, y + 80], dtype=np.float32)
            kps = np.array([[x + 25, y + 30], [x + 55, y + 30], [x + 40, y + 45], [x + 28, y + 60],
                            [x + 52, y + 60]], dtype=np.float32)
            out.append(Detection(bbox, 0.9, kps))
        return out

    def align(self, frame, kps):
        a = np.zeros((112, 112, 3), dtype=np.uint8)
        a[0, 0, 0] = 1 if kps[0, 0] < 250 else 2
        return a

    def embed_aligned(self, aligned):
        rng = np.random.default_rng(7)
        out = []
        for a in aligned:
            base = unit(100) if a[0, 0, 0] == 1 else unit(999)
            v = base + rng.normal(scale=0.01, size=base.shape).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)

    def embed(self, frame, det):
        return self.embed_aligned([self.align(frame, det.kps)])[0]
