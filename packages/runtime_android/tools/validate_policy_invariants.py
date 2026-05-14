import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKGROUND_FIXTURE = ROOT / "packages" / "runtime_android" / "fixtures" / "inference_result.background.json"
UNKNOWN_FIXTURE = ROOT / "packages" / "runtime_android" / "fixtures" / "inference_result.unknown.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    payload = json.loads(BACKGROUND_FIXTURE.read_text(encoding="utf-8"))

    flags = payload.get("flags", {})
    if not flags.get("is_background", False):
        fail("background fixture must set flags.is_background=true")

    if payload.get("label") != "background":
        fail("background invariant violated: label must be background")

    # Notify intent is represented indirectly by low confidence tier in this scaffold.
    if payload.get("confidence_tier") != "low":
        fail("background invariant violated: confidence_tier must be low")

    if flags.get("is_unknown", False):
        fail("background invariant violated: is_unknown must be false when is_background is true")

    unknown_payload = json.loads(UNKNOWN_FIXTURE.read_text(encoding="utf-8"))
    unknown_flags = unknown_payload.get("flags", {})
    if not unknown_flags.get("is_unknown", False):
        fail("unknown fixture must set flags.is_unknown=true")

    if unknown_payload.get("label") != "unknown":
        fail("unknown invariant violated: label must be unknown")

    if unknown_flags.get("is_background", False):
        fail("unknown invariant violated: is_background must be false when is_unknown is true")

    if unknown_payload.get("confidence_tier") != "low":
        fail("unknown invariant violated: confidence_tier must be low")

    print("PASS: policy invariants for background and unknown handling are satisfied")


if __name__ == "__main__":
    main()
