#!/usr/bin/env python3
"""Measure scan accuracy against a labelled fixture set.

Sends each image to a *running* orc-service over HTTP (same path a real caller /
curl uses), compares the 6 returned fields against ground truth, and prints a
per-field + per-document accuracy report.

Usage:
    # start the service first (docker compose up -d), then:
    export API_KEY=<same secret the service runs with>
    python tools/accuracy.py \
        --labels .test-fixtures/labels.json \
        --url http://localhost:8000

Exit code is non-zero if accuracy falls below --min-field / --min-doc, so this
doubles as a CI regression gate. No third-party deps — pure stdlib.

labels.json format — see .test-fixtures/labels.example.json. Each entry keys an
image filename (resolved relative to the labels file's directory unless
--fixtures-dir is given) to its expected fields. Omit or set a field to null to
skip scoring it (useful when a card genuinely has no value for it).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# Fields we score, in report order.
FIELDS = [
    "first_name",
    "last_name",
    "document_number",
    "date_of_birth",
    "sex",
    "country",
]

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def content_type(name: str) -> str:
    return CONTENT_TYPES.get(Path(name).suffix.lower(), "image/jpeg")


def normalize(field: str, value) -> str:
    """Canonicalise a value so cosmetic differences don't count as errors."""
    if value is None:
        return ""
    s = str(value).strip()
    if field == "document_number":
        return re.sub(r"[\s-]", "", s)  # digits/letters only
    if field == "date_of_birth":
        return s[:10]  # ISO date; ignore any time component
    if field == "sex":
        return s.upper()
    if field == "country":
        return s.upper()
    # names: NFC unicode, collapse internal whitespace, casefold
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _multipart(field_name: str, path: Path) -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body carrying one image file."""
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {content_type(path.name)}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


def scan(url: str, api_key: str, path: Path, doc_type: str, timeout: float):
    """Return (status_code, body_dict). doc_type is 'thai_id' or 'passport'."""
    endpoint = f"{url.rstrip('/')}/scan/{doc_type.replace('_', '-')}"
    body, ctype = _multipart("image", path)
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"X-API-Key": api_key, "Content-Type": ctype, "Content-Length": str(len(body))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read(), resp.status
    except urllib.error.HTTPError as e:  # 4xx/5xx still carry a JSON body
        raw, status = e.read(), e.code
    except urllib.error.URLError as e:
        return 0, {"error": "connection_failed", "raw": str(e.reason)}
    try:
        return status, json.loads(raw.decode("utf-8"))
    except Exception:
        return status, {"error": "non_json_response", "raw": raw[:200].decode("utf-8", "replace")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path, help="path to labels.json")
    ap.add_argument("--url", default=os.environ.get("ORC_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"))
    ap.add_argument("--fixtures-dir", type=Path, default=None,
                    help="dir holding the images (default: labels.json's directory)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--min-field", type=float, default=0.0,
                    help="fail (exit 1) if overall field accuracy < this (0..1)")
    ap.add_argument("--min-doc", type=float, default=0.0,
                    help="fail (exit 1) if document (all-fields-correct) accuracy < this (0..1)")
    ap.add_argument("--json", dest="as_json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    if not args.api_key:
        return _die("--api-key or $API_KEY is required (must match the running service)")
    if not args.labels.exists():
        return _die(f"labels file not found: {args.labels}")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    fixtures_dir = args.fixtures_dir or args.labels.parent

    field_correct = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    docs_total = 0
    docs_perfect = 0
    rows = []  # per-document detail for the report

    for name, expected in labels.items():
        if name.startswith("_"):
            continue  # allow "_comment" keys in the labels file
        img_path = fixtures_dir / name
        doc_type = expected.get("type")
        if doc_type not in ("thai_id", "passport"):
            rows.append({"name": name, "error": f"bad 'type': {doc_type!r}"})
            continue
        if not img_path.exists():
            rows.append({"name": name, "error": f"image missing: {img_path}"})
            continue

        docs_total += 1
        status, body = scan(args.url, args.api_key, img_path, doc_type, args.timeout)

        per_field = {}
        all_correct = True
        for f in FIELDS:
            if f not in expected or expected[f] is None:
                continue  # not labelled → not scored
            field_total[f] += 1
            got = body.get(f) if status == 200 else None
            ok = normalize(f, expected[f]) == normalize(f, got)
            per_field[f] = {"expected": expected[f], "got": got, "ok": ok}
            if ok:
                field_correct[f] += 1
            else:
                all_correct = False

        if status != 200:
            all_correct = False
        if all_correct and status == 200:
            docs_perfect += 1

        rows.append({
            "name": name,
            "status": status,
            "error": body.get("error") if status != 200 else None,
            "fields": per_field,
            "all_correct": all_correct and status == 200,
        })

    scored_fields = sum(field_total.values())
    field_acc = (sum(field_correct.values()) / scored_fields) if scored_fields else 0.0
    doc_acc = (docs_perfect / docs_total) if docs_total else 0.0

    if args.as_json:
        print(json.dumps({
            "field_accuracy": field_acc,
            "document_accuracy": doc_acc,
            "documents": docs_total,
            "documents_perfect": docs_perfect,
            "per_field": {f: {"correct": field_correct[f], "total": field_total[f]} for f in FIELDS},
            "rows": rows,
        }, ensure_ascii=False, indent=2))
    else:
        _print_report(rows, field_correct, field_total, field_acc, doc_acc, docs_total, docs_perfect)

    below = (field_acc < args.min_field) or (doc_acc < args.min_doc)
    if below and not args.as_json:
        print(f"\n✗ below threshold (field {field_acc:.1%} < {args.min_field:.0%} "
              f"or doc {doc_acc:.1%} < {args.min_doc:.0%})", file=sys.stderr)
    return 1 if below else 0


def _print_report(rows, field_correct, field_total, field_acc, doc_acc, docs_total, docs_perfect):
    print("=" * 64)
    print("SCAN ACCURACY REPORT")
    print("=" * 64)
    for r in rows:
        if r.get("error") and "fields" not in r:
            print(f"\n  {r['name']}: SKIPPED — {r['error']}")
            continue
        mark = "✓" if r["all_correct"] else "✗"
        extra = f" [{r['status']} {r['error']}]" if r["status"] != 200 else ""
        print(f"\n{mark} {r['name']}{extra}")
        for f, d in r["fields"].items():
            fm = "✓" if d["ok"] else "✗"
            if d["ok"]:
                print(f"    {fm} {f}: {d['got']!r}")
            else:
                print(f"    {fm} {f}: got {d['got']!r}  expected {d['expected']!r}")

    print("\n" + "-" * 64)
    print("Per-field accuracy:")
    for f in FIELDS:
        tot = field_total[f]
        if not tot:
            print(f"    {f:16s} —  (not labelled)")
            continue
        acc = field_correct[f] / tot
        print(f"    {f:16s} {field_correct[f]:3d}/{tot:<3d}  {acc:6.1%}")
    print("-" * 64)
    print(f"Overall field accuracy : {field_acc:.1%}")
    print(f"Document accuracy      : {doc_acc:.1%}  ({docs_perfect}/{docs_total} images fully correct)")


def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
