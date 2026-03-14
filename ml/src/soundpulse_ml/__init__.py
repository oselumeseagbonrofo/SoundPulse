"""SoundPulse ML workspace package."""

from .contracts import load_contract_bundle
from .datasets import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SOURCE_MAPPING_PATH,
    DEFAULT_SUMMARY_PATH,
    build_phase1_manifest,
    load_dataset_manifest,
    manifest_to_dataframe,
    resolve_audio_path,
    summarize_dataset_manifest,
    summarize_dataset_manifest_frame,
    validate_dataset_manifest,
    write_phase1_dataset_artifacts,
)

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_SOURCE_MAPPING_PATH",
    "DEFAULT_SUMMARY_PATH",
    "build_phase1_manifest",
    "load_contract_bundle",
    "load_dataset_manifest",
    "manifest_to_dataframe",
    "resolve_audio_path",
    "summarize_dataset_manifest",
    "summarize_dataset_manifest_frame",
    "validate_dataset_manifest",
    "write_phase1_dataset_artifacts",
]
