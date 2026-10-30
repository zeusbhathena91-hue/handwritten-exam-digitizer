"""Command line:  python -m exam_digitizer.cli digitize page1.png page2.png --booklet-id T001 --roster roster.txt
                python -m exam_digitizer.cli serve --port 8000"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exam_digitizer", description="Handwritten Examination Digitizer (Group 1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digitize", help="digitize one booklet (page images in order, or one PDF)")
    d.add_argument("pages", nargs="+", help="page image files (PNG/JPEG) in reading order, or a single PDF")
    d.add_argument("--booklet-id", default="")
    d.add_argument("--roster", help="file with one roll number per line, or a comma-separated list")
    d.add_argument("--roll-policy", default="digits-exact", choices=["digits-exact", "strict", "snap", "format-only"])
    d.add_argument("--include-lines", action="store_true", help="also return per-line boxes and confidences")
    d.add_argument("--review-threshold", type=float, default=0.5)
    d.add_argument("--out", help="write JSON here instead of stdout")
    s = sub.add_parser("serve", help="start the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("exam_digitizer.api.app:app", host=a.host, port=a.port)
        return 0

    from .digitizer import Digitizer
    roster = None
    if a.roster:
        roster = ([ln.strip() for ln in Path(a.roster).read_text().splitlines() if ln.strip()] if Path(a.roster).exists()
                  else [x.strip() for x in a.roster.split(",") if x.strip()])
    res = Digitizer().digitize(a.pages, booklet_id=a.booklet_id or Path(a.pages[0]).stem, roster=roster,
                               roll_policy=a.roll_policy, include_lines=a.include_lines, review_threshold=a.review_threshold)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}: {len(res['answers'])} answers, student_id={res['student_id']}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
