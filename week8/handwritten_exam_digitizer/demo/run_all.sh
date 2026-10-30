#!/usr/bin/env bash
# Start PromptGuard (8001) + Group 2 (8000) + Group 1 digitizer (8002), wait until healthy, run the demo, stop everything.
# Usage:  G1_DIR=~/Desktop/handwritten_exam_digitizer G2_DIR=~/path/AI-Exam-Answer-Grader ./run_all.sh
# Real PromptGuard: set PG_CMD="python -m promptguard serve --host 127.0.0.1 --port 8001 --profile lite" (and PG_DIR=its folder).
set -u
PY="${PY:-python3.12}"; G1_DIR="${G1_DIR:-$HOME/Desktop/handwritten_exam_digitizer}"; G2_DIR="${G2_DIR:?set G2_DIR to the AI-Exam-Answer-Grader folder}"
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG=/tmp/g1g2; mkdir -p "$LOG"; PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }; trap cleanup EXIT
wait_for() {  # url, name, tries
  for _ in $(seq 1 "$3"); do curl -s -o /dev/null "$1" && { echo "  ok: $2"; return 0; }; sleep 3; done
  echo "  FAILED: $2 (see $LOG)"; return 1; }

if [ -n "${PG_CMD:-}" ]; then (cd "${PG_DIR:-.}" && eval "$PG_CMD" >"$LOG/pg.log" 2>&1 &) ; PIDS+=($!)
else (cd "$HERE" && $PY -m uvicorn fake_promptguard:app --port 8001 >"$LOG/pg.log" 2>&1) & PIDS+=($!); echo "  (using the allow-all TEST guard)"; fi
(cd "$G2_DIR/backend" && PROMPTGUARD_URL=http://127.0.0.1:8001 PROMPTGUARD_TIMEOUT_SECONDS=15 uv run uvicorn api:app --app-dir src --port 8000 >"$LOG/g2.log" 2>&1) & PIDS+=($!)
(cd "$G1_DIR" && $PY -m exam_digitizer.cli serve --port 8002 >"$LOG/g1.log" 2>&1) & PIDS+=($!)
echo "waiting for services (first start downloads models; up to ~5 min)..."
wait_for http://127.0.0.1:8001/openapi.json "PromptGuard :8001" 40 || exit 1
wait_for http://127.0.0.1:8000/health "Group 2 :8000" 100 || exit 1
wait_for http://127.0.0.1:8002/health "Group 1 :8002" 100 || exit 1
cd "$G1_DIR" || exit 1
echo "--- scan -> digitize -> grade"; $PY demo/exam_pipeline.py demo/sample_booklet/T001_p1.png --answer-key demo/answer_key.json \
  --digitizer-url http://127.0.0.1:8002 --grader-url http://127.0.0.1:8000 --roster demo/roster.txt | tail -1
echo "--- after human review"; $PY demo/exam_pipeline.py demo/T001_reviewed.json --answer-key demo/answer_key.json --grader-url http://127.0.0.1:8000 | tail -1