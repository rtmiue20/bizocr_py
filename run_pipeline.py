"""Run the complete Model B MLOps pipeline in a safe, explicit order.
This orchestrator runs:
  export -> validate/split -> train -> evaluate gate -> register
It does not start the long-running OCR API.  That service should be started
separately after the active model pointer has been updated.
Examples:
  python run_pipeline.py --schema-dir schemas --batch-id 12
  python run_pipeline.py --schema-dir schemas --no-deploy
  python run_pipeline.py --dataset datasets/ds-existing --from-stage validate
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

STAGES = ("export", "validate", "train", "evaluate", "register")
STAGE_INDEX = {name: index for index, name in enumerate(STAGES)}


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Model B from trusted audit export to safe model registration."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--saved-models", type=Path, default=Path("saved_models"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--batch-id", type=int, default=None, help="Specific learning batch ID to export")
    parser.add_argument("--golden-ids", type=Path, default=None, help="Path to frozen golden dataset IDs JSON file")
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument("--sql-file", type=Path, default=None)
    parser.add_argument("--baseline-report", type=Path, default=None, help="Explicit baseline evaluation_report.json")
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"))
    # Cập nhật mặc định min-golden từ 10 thành 50
    parser.add_argument("--min-golden", type=int, default=50)
    parser.add_argument("--min-train-docs", type=int, default=300, help="Minimum required training documents to allow LoRA training")
    parser.add_argument("--min-field-accuracy", type=float, default=0.90)
    parser.add_argument("--min-schema-rate", type=float, default=0.99)
    parser.add_argument("--from-stage", choices=STAGES, default="export")
    parser.add_argument("--to-stage", choices=STAGES, default="register")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--no-deploy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def ensure_pipeline_files(root: Path) -> None:
    required = [
        root / "training_pipeline" / "__init__.py",
        root / "training_pipeline" / "pipeline_common.py",
        root / "training_pipeline" / "export_sanitized_dataset.py",
        root / "training_pipeline" / "validate_and_split.py",
        root / "training_pipeline" / "train_lora.py",
        root / "training_pipeline" / "evaluate_gate.py",
        root / "training_pipeline" / "register_model.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline files:\n  - " + "\n  - ".join(missing)
        )


def command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", f"training_pipeline.{module}", *arguments]


def print_command(label: str, args: Sequence[str]) -> None:
    print(f"\n[{label}]")
    print("$ " + " ".join(repr(str(arg)) for arg in args))


def run_command(
    label: str,
    args: list[str],
    root: Path,
    dry_run: bool,
) -> None:
    print_command(label, args)
    if dry_run:
        return
    subprocess.run(args, cwd=str(root), check=True)


def acquire_lock(root: Path, run_id: str):
    lock_path = root / ".model_b_pipeline.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another pipeline appears to be running: {lock_path}. "
            "Remove it only after confirming no run is active."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "run_id": run_id}, handle)
    return lock_path


def discover_candidate(saved_models: Path, candidate_version: str) -> Path:
    candidate = saved_models / candidate_version
    manifest = candidate / "candidate_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Training completed but candidate manifest is missing: {manifest}"
        )
    return candidate


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    ensure_pipeline_files(root)

    if STAGE_INDEX[args.from_stage] > STAGE_INDEX[args.to_stage]:
        raise ValueError("--from-stage must not come after --to-stage")
    if args.min_golden < 1:
        raise ValueError("--min-golden must be at least 1")
    if args.min_train_docs < 1:
        raise ValueError("--min-train-docs must be at least 1")
    if not 0 < args.min_field_accuracy <= 1:
        raise ValueError("--min-field-accuracy must be between 0 and 1")
    if not 0 < args.min_schema_rate <= 1:
        raise ValueError("--min-schema-rate must be between 0 and 1")

    # Thông báo đường dẫn HF_HOME cache nếu được cấu hình
    hf_cache = os.getenv("HF_HOME")
    if hf_cache:
        print(f"INFO: Using HuggingFace cache directory HF_HOME={hf_cache}")

    run_id = args.run_id or utc_run_id()
    dataset_root = resolve(root, args.dataset_root)
    saved_models = resolve(root, args.saved_models)
    schema_dir = resolve(root, args.schema_dir) if args.schema_dir else None
    sql_file = resolve(root, args.sql_file) if args.sql_file else None

    # Tự động tìm file golden_ids.json nếu người dùng không truyền trực tiếp
    golden_ids_file = resolve(root, args.golden_ids) if args.golden_ids else dataset_root / "golden_ids.json"

    dataset_dir = resolve(root, args.dataset) if args.dataset else dataset_root / f"ds-{run_id}"
    candidate_version = f"lora-{dataset_dir.name}-{run_id}"
    candidate_dir = saved_models / candidate_version

    if args.from_stage == "export" and dataset_dir.exists() and not args.dry_run:
        raise FileExistsError(
            f"Dataset directory already exists: {dataset_dir}. "
            "Use a new --run-id or start from an existing dataset explicitly."
        )
    if args.from_stage != "export" and not dataset_dir.exists():
        raise FileNotFoundError(
            f"Existing dataset is required when starting at {args.from_stage}: {dataset_dir}"
        )
    if args.to_stage == "register" and args.no_deploy:
        print("NOTE: --no-deploy set; register stage will be skipped.")

    lock_path = None
    try:
        if not args.dry_run:
            lock_path = acquire_lock(root, run_id)
        print(f"=== MODEL B PIPELINE: {run_id} ===")
        print(f"Project:   {root}")
        print(f"Dataset:   {dataset_dir}")
        print(f"Candidate: {candidate_dir}")
        start = STAGE_INDEX[args.from_stage]
        end = STAGE_INDEX[args.to_stage]

        # 1. EXPORT
        if start <= STAGE_INDEX["export"] <= end:
            export_args = [
                "--output-root", str(dataset_root),
                "--dataset-version", dataset_dir.name,
            ]
            if args.batch_id is not None:
                export_args.extend(["--batch-id", str(args.batch_id)])
            if sql_file:
                export_args.extend(["--sql-file", str(sql_file)])
            run_command(
                "1/5 EXPORT TRUSTED DATASET",
                command("export_sanitized_dataset", *export_args),
                root,
                args.dry_run,
            )

        # 2. VALIDATE AND SPLIT
        if start <= STAGE_INDEX["validate"] <= end:
            # Đã loại bỏ --min-train-docs để tránh lỗi argparse của validate_and_split
            validate_args = [
                "--dataset", str(dataset_dir),
                "--min-golden", str(args.min_golden),
            ]
            if golden_ids_file.exists():
                validate_args.extend(["--golden-ids", str(golden_ids_file)])
            if schema_dir:
                validate_args.extend(["--schema-dir", str(schema_dir)])
            if args.allow_small:
                validate_args.append("--allow-small")
            run_command(
                "2/5 VALIDATE AND SPLIT",
                command("validate_and_split", *validate_args),
                root,
                args.dry_run,
            )

        # 3. TRAIN LORA
        if start <= STAGE_INDEX["train"] <= end:
            # Kiểm tra ngưỡng mẫu huấn luyện tối thiểu trước khi chạy LoRA
            train_file = dataset_dir / "train.jsonl"
            if train_file.exists() and not args.dry_run:
                with train_file.open("r", encoding="utf-8") as f:
                    train_count = sum(1 for line in f if line.strip())
                if train_count < args.min_train_docs and not args.allow_small:
                    raise ValueError(
                        f"Insufficient training samples ({train_count} < {args.min_train_docs}). "
                        "LoRA training blocked to prevent overfitting. Use --allow-small to override."
                    )

            train_args = [
                "--dataset", str(dataset_dir),
                "--output-root", str(saved_models),
                "--candidate-version", candidate_version,
                "--base-model", args.base_model,
            ]
            if args.load_in_4bit:
                train_args.append("--load-in-4bit")
            run_command(
                "3/5 TRAIN LORA CANDIDATE",
                command("train_lora", *train_args),
                root,
                args.dry_run,
            )

        # 4. EVALUATION GATE
        if start <= STAGE_INDEX["evaluate"] <= end:
            baseline_report_path = args.baseline_report
            if not baseline_report_path:
                active_manifest_path = saved_models / "active_model" / "production_manifest.json"
                if active_manifest_path.exists():
                    try:
                        active_data = json.loads(active_manifest_path.read_text(encoding="utf-8"))
                        cand_ref = active_data.get("candidate_path")
                        act_ver = active_data.get("active_version")
                        candidates_to_check = []
                        if cand_ref:
                            p = Path(cand_ref)
                            candidates_to_check.append(p if p.is_absolute() else saved_models / p)
                        if act_ver:
                            candidates_to_check.append(saved_models / act_ver)
                        for c in candidates_to_check:
                            rep = c / "evaluation_report.json"
                            if rep.is_file():
                                baseline_report_path = rep
                                print(f"Auto-detected active baseline report: {rep}")
                                break
                    except Exception as e:
                        print(f"Could not auto-detect baseline report: {e}")

            evaluate_args = [
                "--candidate", str(candidate_dir),
                "--golden", str(dataset_dir / "golden.jsonl"),
                "--min-golden", str(args.min_golden),
                "--min-field-accuracy", str(args.min_field_accuracy),
                "--min-schema-rate", str(args.min_schema_rate),
            ]
            if baseline_report_path:
                evaluate_args.extend(["--baseline-report", str(baseline_report_path)])
            if schema_dir:
                evaluate_args.extend(["--schema-dir", str(schema_dir)])
            run_command(
                "4/5 EVALUATION GATE",
                command("evaluate_gate", *evaluate_args),
                root,
                args.dry_run,
            )

        # 5. REGISTER
        if start <= STAGE_INDEX["register"] <= end:
            if args.no_deploy:
                print("\n[5/5 REGISTER] skipped because --no-deploy was set.")
            else:
                run_command(
                    "5/5 ATOMIC REGISTER",
                    command(
                        "register_model",
                        "--candidate", str(candidate_dir),
                        "--saved-models", str(saved_models),
                    ),
                    root,
                    args.dry_run,
                )

        print("\n=== MODEL B PIPELINE COMPLETED ===")
        if args.no_deploy or args.to_stage != "register":
            print("No production activation was requested.")
        else:
            print(f"Active pointer: {saved_models / 'active_model' / 'production_manifest.json'}")
        return 0

    except subprocess.CalledProcessError as exc:
        print(
            f"\nPIPELINE_STOPPED: stage command failed with exit code {exc.returncode}. "
            "No later stage was executed.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:
        print(f"\nPIPELINE_STOPPED: {exc}", file=sys.stderr)
        return 1
    finally:
        if lock_path and lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())