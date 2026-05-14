import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "packages" / "runtime_android" / "fixtures" / "inference_result.valid.json"

REQUIRED_TOP_LEVEL = {
    "label",
    "distance",
    "confidence",
    "confidence_tier",
    "top_candidates",
    "flags",
}

REQUIRED_FLAGS = {
    "is_unknown",
    "is_background",
    "speech_pitch_required",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    missing = REQUIRED_TOP_LEVEL - payload.keys()
    if missing:
        fail(f"missing required fields: {sorted(missing)}")

    if payload["confidence_tier"] not in {"high", "medium", "low"}:
        fail("confidence_tier must be high|medium|low")

    flags = payload["flags"]
    missing_flags = REQUIRED_FLAGS - flags.keys()
    if missing_flags:
        fail(f"missing flag fields: {sorted(missing_flags)}")

    for key in ("label", "distance", "confidence"):
        if key not in payload["top_candidates"][0]:
            fail(f"top candidate is missing '{key}'")

    confidence = payload["confidence"]
    if not (0.0 <= confidence <= 1.0):
        fail("confidence must be between 0 and 1")

    print("PASS: inference_result fixture satisfies required contract fields")


if __name__ == "__main__":
    main()
