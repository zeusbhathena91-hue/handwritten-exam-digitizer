"""TEST-ONLY stand-in for Group 12's PromptGuard: allows every answer. Proves the wiring, blocks nothing.
Do not use it for the prompt-injection demo or report guard results from it.
    python3.12 -m uvicorn fake_promptguard:app --port 8001"""
from fastapi import FastAPI
app = FastAPI()


@app.post("/v1/analyze")
def analyze(body: dict):
    return {"recommended_action": "ALLOW", "attack_detected": False, "risk_level": "LOW", "request_id": body.get("request_id")}
