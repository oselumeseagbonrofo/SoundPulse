from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = ROOT / "packages" / "shared_contracts" / "v1"


@dataclass(frozen=True)
class ContractBundle:
    class_labels: dict[str, Any]
    preprocessing: dict[str, Any]
    threshold_defaults: dict[str, Any]
    context_aware_alert_policy: dict[str, Any]

    @property
    def class_slugs(self) -> list[str]:
        return [item["slug"] for item in self.class_labels["classes"]]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_contract_bundle(contracts_dir: Path | None = None) -> ContractBundle:
    base_dir = contracts_dir or CONTRACTS_DIR
    bundle = ContractBundle(
        class_labels=_load_json(base_dir / "class_labels.json"),
        preprocessing=_load_json(base_dir / "preprocessing_config.json"),
        threshold_defaults=_load_json(base_dir / "threshold_defaults.json"),
        context_aware_alert_policy=_load_json(base_dir / "context_aware_alert_policy.json"),
    )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: ContractBundle) -> None:
    classes = bundle.class_labels.get("classes", [])
    if len(classes) != 10:
        raise ValueError(f"Expected 10 classes, found {len(classes)}")

    ids = [item["id"] for item in classes]
    if ids != list(range(len(classes))):
        raise ValueError(f"Class IDs must be contiguous and zero-based, found {ids}")

    slugs = [item["slug"] for item in classes]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Class slugs must be unique")

    audio = bundle.preprocessing.get("audio", {})
    if audio.get("window_duration_ms") != 1000:
        raise ValueError("Window duration must remain 1000 ms for the runtime contract")

    embedding_contract = bundle.preprocessing.get("embedding_contract", {})
    if embedding_contract.get("tiny_cnn_dim") != 1024:
        raise ValueError("Tiny CNN embedding dimension contract must be 1024")
    if embedding_contract.get("yamnet_dim") != 1024:
        raise ValueError("YAMNet embedding dimension contract must be 1024")

    modes = bundle.threshold_defaults.get("modes", {})
    required_modes = {"general", "focus", "sleep"}
    if set(modes.keys()) != required_modes:
        raise ValueError(f"Expected modes {required_modes}, found {set(modes.keys())}")


def summarize_contracts() -> str:
    bundle = load_contract_bundle()
    sample_rate = bundle.preprocessing["audio"]["sample_rate_hz"]
    return (
        "SoundPulse contracts loaded successfully\n"
        f"classes={','.join(bundle.class_slugs)}\n"
        f"sample_rate_hz={sample_rate}\n"
        f"tiny_cnn_dim={bundle.preprocessing['embedding_contract']['tiny_cnn_dim']}\n"
        f"yamnet_dim={bundle.preprocessing['embedding_contract']['yamnet_dim']}"
    )


if __name__ == "__main__":
    print(summarize_contracts())
