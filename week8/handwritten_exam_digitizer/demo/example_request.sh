#!/bin/bash
# Start the service in one terminal:   python -m exam_digitizer.cli serve --port 8000
# Then call it (pages of ONE booklet, in order):
curl -s -F "pages=@demo/sample_booklet/T001_p1.png" \
     -F "booklet_id=T001" -F "roster=$(paste -sd, demo/roster.txt)" \
     http://127.0.0.1:8000/digitize | python -m json.tool
