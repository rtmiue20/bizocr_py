"""Promote a passed candidate using an atomic active-model pointer.
The candidate directory is never deleted or overwritten.  Rollback is a
pointer update, so a failed copy or process restart cannot leave production
without a model.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

try:
    from training_pipeline.pipeline_common import read_json, sha256_file, utc_now, write_json_atomic
except ImportError:
    from pipeline_common import read_json, sha256_file, utc_now, write_json_atomic

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--saved-models", type=Path, default=Path("saved_models"))
    parser.add_argument("--rollback-to", type=Path, default=None)
    return parser.parse_args()
def _verify_candidate(candidate_dir: Path) -> tuple[dict, dict]:
    candidate = read_json(candidate_dir / "candidate_manifest.json")
    report = read_json(candidate_dir / "evaluation_report.json")
    if report.get("gate_status") != "PASSED":
        raise RuntimeError(
            f"Candidate gate status is {report.get('gate_status')}, not PASSED"
        )
    adapter_dir = Path(candidate["adapter_path"])
    if not adapter_dir.is_absolute():
        adapter_dir = candidate_dir / adapter_dir
    if not adapter_dir.is_dir():
        raise FileNotFoundError(adapter_dir)
    for relative, expected in candidate.get("adapter_files", {}).items():
        file_path = adapter_dir / relative
        if not file_path.exists() or sha256_file(file_path) != expected:
            raise RuntimeError(f"Adapter checksum mismatch: {file_path}")
    return candidate, report
def register(candidate_dir: Path, saved_models: Path) -> Path:
    candidate, report = _verify_candidate(candidate_dir)
    saved_models.mkdir(parents=True, exist_ok=True)
    active_dir = saved_models / "active_model"
    active_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = active_dir / "production_manifest.json"
    previous = read_json(pointer_path) if pointer_path.exists() else None
    try:
        relative_candidate = candidate_dir.resolve().relative_to(saved_models.resolve())
        candidate_reference = str(relative_candidate)
    except ValueError:
        candidate_reference = str(candidate_dir.resolve())
    production = {
        "active_version": candidate["candidate_version"],
        "candidate_path": candidate_reference,
        "adapter_path": candidate["adapter_path"],
        "base_model": candidate["base_model"],
        "dataset_version": candidate["dataset_version"],
        "field_accuracy": report.get("field_accuracy"),
        "schema_valid_rate": report.get("schema_valid_rate"),
        "activated_at": utc_now(),
        "previous_active": previous.get("active_version") if previous else None,
        "status": "ACTIVE",
    }
    write_json_atomic(pointer_path, production)
    write_json_atomic(
        active_dir / "deployment_history.json",
        {
            "last_activation": production,
            "previous_manifest": previous,
        },
    )
    print(f"Active model pointer updated atomically: {candidate['candidate_version']}")
    return pointer_path
def rollback(target: Path, saved_models: Path) -> Path:
    """Rollback to a previously verified candidate directory."""
    candidate, report = _verify_candidate(target)
    # Rollback is deliberately recorded as a new activation event.
    return register(target, saved_models)
def main() -> int:
    args = parse_args()
    try:
        target = args.rollback_to or args.candidate
        register(target, args.saved_models)
        return 0
    except Exception as exc:
        print(f"REGISTRATION_FAILED: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())