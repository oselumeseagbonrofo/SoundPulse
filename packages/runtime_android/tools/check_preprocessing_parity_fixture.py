import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "packages" / "shared_contracts" / "v1" / "preprocessing_config.json"
FIXTURE = ROOT / "packages" / "runtime_android" / "fixtures" / "preprocessing_parity_fixture.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def assert_equal(contract_value, fixture_value, name: str) -> None:
    if contract_value != fixture_value:
        fail(f"{name} mismatch: contract={contract_value} fixture={fixture_value}")


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert_equal(contract["audio"]["sample_rate_hz"], fixture["sample_rate_hz"], "sample_rate_hz")
    assert_equal(contract["audio"]["window_duration_ms"], fixture["window_duration_ms"], "window_duration_ms")
    assert_equal(contract["mel"]["n_mels"], fixture["n_mels"], "n_mels")
    assert_equal(contract["stft"]["n_fft"], fixture["n_fft"], "n_fft")
    assert_equal(contract["stft"]["hop_length"], fixture["hop_length"], "hop_length")
    assert_equal(contract["stft"]["win_length"], fixture["win_length"], "win_length")

    expected = fixture["expected_embedding_dims"]
    assert_equal(contract["embedding_contract"]["tiny_cnn_dim"], expected["tiny_cnn_dim"], "tiny_cnn_dim")
    assert_equal(contract["embedding_contract"]["yamnet_dim"], expected["yamnet_dim"], "yamnet_dim")

    print("PASS: preprocessing parity fixture matches shared contract")


if __name__ == "__main__":
    main()
