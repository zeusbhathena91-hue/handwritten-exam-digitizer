"""API tests with FastAPI's TestClient (no server process needed)."""
import base64
import io

import pytest
from PIL import Image

from tests.conftest import slow


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from exam_digitizer.api.app import app
    with TestClient(app) as c:         # runs the lifespan → models load once
        yield c


@slow
def test_health_and_schema(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["schema_version"] == "1.0" and "trocr" in h["model"]
    s = client.get("/schema").json()
    assert "answers" in s["response"]["properties"] and s["example"]["schema_version"] == "1.0"


@slow
def test_digitize_multipart(client, page):
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, "PNG")
    r = client.post("/digitize", files=[("pages", ("p1.png", buf.getvalue(), "image/png"))],
                    data={"booklet_id": "API1", "roster": "23D007,23D008", "include_lines": "true"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["booklet_id"] == "API1" and j["pages"] == 1 and len(j["answers"]) >= 1
    assert "lines" in j["answers"][0]


@slow
def test_digitize_base64(client, page):
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, "PNG")
    body = {"booklet_id": "B64", "pages": [base64.b64encode(buf.getvalue()).decode()], "roster": ["23D007"]}
    r = client.post("/digitize-base64", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["booklet_id"] == "B64" and "lines" not in r.json()["answers"][0]


@slow
def test_bad_requests(client):
    r = client.post("/digitize", files=[("pages", ("p.png", b"not an image", "image/png"))])
    assert r.status_code == 400
    r = client.post("/digitize-base64", json={"pages": ["AAAA"], "roll_policy": "nearest"})
    assert r.status_code == 422
