#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""BOK 원문 수집부터 분석용 clean text 생성까지 순차 실행."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.h2.data_io import ROOT

STEPS = [
    "collect_bok.py",
    "fix_duplicate_meetings.py",
    "recover_bok_empty.py",
    "build_bok_text_full.py",
    "clean_bok_text.py",
]

def main():
    for step in STEPS:
        print(f"\n>>> running {step}",flush=True)
        subprocess.run([sys.executable, '-B', '-m',
                        'src.h2.preprocessing.'+Path(step).stem],cwd=ROOT,check=True)
    print("\n=== BOK collection/preprocessing pipeline completed ===")

if __name__ == '__main__':main()
