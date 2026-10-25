"""Runs the Lab 7 service smoke test in a subprocess with the stand-in model.

A subprocess keeps its environment (isolated cache, fake provider shadowing
litellm) from leaking into the other tests in this session.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lab7_service_smoke():
    proc = subprocess.run([sys.executable, str(ROOT / "tests/lab7_smoke.py")],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
