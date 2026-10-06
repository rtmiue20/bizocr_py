"""Promote a passed candidate using an atomic active-model pointer.
The candidate directory is never deleted or overwritten. Rollback is a
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
    from training_pipeline.pipeline_common import (
        read_json,
        sha256_file,
        utc_now,
        write_json_atomic,
    )
except ImportError:
    from pipeline_common import (
        read_json,
        sha256_file,
        utc_now,
        write_json_atomic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--saved-models", type=Path, default=Path("saved_models"))
    parser.add_argument("--rollback-to", type=Path, default=None)
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Explicitly bypass regression check for rollback operations",
    )
    parser.add_argument(
        "--allow-different-golden",
        action="store_true",
        help="Allow registering candidate evaluated on a different golden set than the active model",
    )
    args = parser.parse_args()
    if not args.candidate and not args.rollback_to:
        parser.error("Either --candidate or --rollback-to must be specified.")
    return args


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


def register(
    candidate_dir: Path,
    saved_models: Path,
    is_rollback: bool = False,
    allow_different_golden: bool = False,
) -> Path:
    candidate, report = _verify_candidate(candidate_dir)
    saved_models.mkdir(parents=True, exist_ok=True)
    active_dir = saved_models / "active_model"
    active_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = active_dir / "production_manifest.json"
    previous = read_json(pointer_path) if pointer_path.exists() else None

    # Regression check: Reject candidate if worse than currently running model on golden
    # Skip regression check entirely if this is a rollback operation
    if previous and not is_rollback:
        prev_acc = previous.get("field_accuracy")
        cand_acc = report.get("field_accuracy")
        prev_schema = previous.get("schema_valid_rate")
        cand_schema = report.get("schema_valid_rate")

        prev_sha = previous.get("golden_sha256")
        cand_sha = report.get("golden_sha256")
        same_golden = bool(prev_sha and cand_sha and prev_sha == cand_sha)

        if not same_golden and not allow_different_golden:
            raise RuntimeError(
                f"Candidate rejected: golden_sha256 mismatch between candidate ({str(cand_sha)[:10]}) "
                f"and active model ({str(prev_sha)[:10]}). "
                f"Cannot compare metrics across different golden sets. "
                f"Use --allow-different-golden to override."
            )

        if prev_acc is not None and cand_acc is not None:
            if cand_acc < prev_acc:
                msg = (
                    f"Candidate rejected: candidate field_accuracy ({cand_acc:.4f}) is worse than "
                    f"currently active model ({prev_acc:.4f})"
                )
                if same_golden:
                    msg += f" on identical golden set ({cand_sha[:10]})."
                raise RuntimeError(msg)

        if prev_schema is not None and cand_schema is not None:
            if cand_schema < prev_schema:
                raise RuntimeError(
                    f"Candidate rejected: candidate schema_valid_rate ({cand_schema:.4f}) is worse than "
                    f"currently active model ({prev_schema:.4f})."
                )

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
        "golden_sha256": report.get("golden_sha256"),
        "field_accuracy": report.get("field_accuracy"),
        "schema_valid_rate": report.get("schema_valid_rate"),
        "activated_at": utc_now(),
        "previous_active": previous.get("active_version") if previous else None,
        "status": "ACTIVE",
        "is_rollback": is_rollback,
    }
    write_json_atomic(pointer_path, production)

    # Append-only deployment history
    history_path = active_dir / "deployment_history.json"
    history_list: list[dict] = []
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                history_list = data
            elif isinstance(data, dict):
                history_list = [data]
        except Exception:
            history_list = []
    history_list.append(production)
    write_json_atomic(history_path, history_list)

    # Also append to deployment_history.jsonl
    jsonl_path = active_dir / "deployment_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(production, ensure_ascii=False) + "\n")

    print(f"Active model pointer updated atomically: {candidate['candidate_version']}")
    return pointer_path


def rollback(
    target: Path,
    saved_models: Path,
    allow_different_golden: bool = False,
) -> Path:
    """Rollback to a previously verified candidate directory."""
    candidate, report = _verify_candidate(target)
    # Rollback is deliberately recorded as a new activation event with is_rollback=True
    return register(
        target,
        saved_models,
        is_rollback=True,
        allow_different_golden=allow_different_golden,
    )


def main() -> int:
    args = parse_args()
    try:
        if args.rollback_to:
            rollback(
                args.rollback_to,
                args.saved_models,
                allow_different_golden=args.allow_different_golden,
            )
        else:
            register(
                args.candidate,
                args.saved_models,
                is_rollback=args.rollback,
                allow_different_golden=args.allow_different_golden,
            )
        return 0
    except Exception as exc:
        print(f"REGISTRATION_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())