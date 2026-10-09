#!/usr/bin/env python3
"""Reconstruct two large, losslessly packaged CSVs from repository-hosted gzip bytes.

Use only bundled files. SHA-256 verifies the compressed Git snapshot and the
exact restored CSV payload against FILE_SHA256.json. No network access needed.
"""
from __future__ import annotations
from pathlib import Path
import gzip
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    ("results/r6/R6_MATCHED_PAIRS_0p5.csv", False),
    ("results/r9/SHARED_SUPPORT_PAIRS.csv", True),
]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fd:
        for block in iter(lambda: fd.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def restore() -> None:
    manifest=json.loads((ROOT/'FILE_SHA256.json').read_text(encoding='utf-8'))
    for rel,split in FILES:
        out=ROOT/rel
        if out.exists():
            assert sha(out)==manifest[rel], f"Unexpected existing file: {rel}"
            print('ALREADY_VERIFIED',rel)
            continue
        base=ROOT/(rel+'.gz')
        if split:
            paths=sorted(base.parent.glob(base.name+'.part*'))
            assert [p.name.rsplit('.part',1)[1] for p in paths]==[f'{i:02d}' for i in range(5)], f"Missing/nonconsecutive parts: {rel}"
        else:
            paths=[base]
        assert all(p.is_file() for p in paths), f"Missing compressed input for {rel}"
        for path in paths:
            relative=path.relative_to(ROOT).as_posix()
            assert sha(path)==manifest[relative], f"Corrupted compressed file: {relative}"
        compressed=b''.join(p.read_bytes() for p in paths)
        raw=gzip.decompress(compressed)
        assert hashlib.sha256(raw).hexdigest()==manifest[rel], f"Restored CSV does not match original: {rel}"
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(raw)
        assert sha(out)==manifest[rel]
        print('RESTORED',rel,len(raw),'bytes SHA-256 PASS')

if __name__=='__main__':
    restore()
