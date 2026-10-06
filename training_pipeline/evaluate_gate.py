"""Run a real candidate-model evaluation against an isolated golden set."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from collections import defaultdict

try:
    from training_pipeline.pipeline_common import (
        extract_json_object,
        iter_jsonl,
        load_schema,
        sha256_jsonl,
        utc_now,
        validate_schema,
        values_equal,
        write_json_atomic,
    )
except ImportError:
    from pipeline_common import (
        extract_json_object,
        iter_jsonl,
        load_schema,
        sha256_jsonl,
        utc_now,
        validate_schema,
        values_equal,
        write_json_atomic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--allow-baseline-mismatch", action="store_true")
    parser.add_argument("--min-field-accuracy", type=float, default=0.90)
    parser.add_argument("--min-schema-rate", type=float, default=0.99)
    parser.add_argument("--min-golden", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_runtime_dependencies():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires torch, transformers and peft."
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def format_prompt(tokenizer, record: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": str(record["instruction"])},
        {"role": "user", "content": str(record["input"])},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"System:\n{messages[0]['content']}\nUser:\n{messages[1]['content']}\nAssistant:\n"


def load_candidate(candidate_dir: Path):
    manifest_path = candidate_dir / "candidate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("base_model", "adapter_path", "candidate_version", "dataset_version"):
        if not manifest.get(key):
            raise ValueError(f"Candidate manifest missing {key}")
    adapter_path = Path(manifest["adapter_path"])
    if not adapter_path.is_absolute():
        adapter_path = candidate_dir / adapter_path
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = load_runtime_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    base = AutoModelForCausalLM.from_pretrained(manifest["base_model"], **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return manifest, tokenizer, model, torch


def generate_prediction(tokenizer, model, torch, record, max_new_tokens: int) -> Any:
    prompt = format_prompt(tokenizer, record)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # temperature=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_json_object(raw)


def evaluate_record(predicted: Any, target: dict[str, Any], schema: dict | None):
    if not isinstance(predicted, dict):
        return {
            "json_valid": False,
            "schema_valid": False,
            "matched": 0,
            "total": len(target),
            "matched_field_names": [],
            "all_field_names": list(target.keys()),
            "target_non_null": sum(1 for v in target.values() if v is not None),
            "matched_non_null": 0,
            "target_null": sum(1 for v in target.values() if v is None),
            "hallucinated": 0,
            "wrong_value": 0,
        }

    schema_errors = validate_schema(predicted, schema)

    matched_names = []
    target_non_null = 0
    matched_non_null = 0
    target_null = 0
    hallucinated = 0
    wrong_value = 0

    for key, value in target.items():
        if value is None:
            target_null += 1
            if key in predicted:
                if predicted[key] is None:
                    matched_names.append(key)
                else:
                    hallucinated += 1
        else:
            target_non_null += 1
            if key in predicted:
                if values_equal(predicted[key], value):
                    matched_names.append(key)
                    matched_non_null += 1
                else:
                    wrong_value += 1

    matched = len(matched_names)

    return {
        "json_valid": True,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors[:5],
        "matched": matched,
        "total": len(target),
        "matched_field_names": matched_names,
        "all_field_names": list(target.keys()),
        "exact_document": matched == len(target) and all(
            key in target for key in predicted
        ),
        "extra_fields": sorted(set(predicted) - set(target)),
        "target_non_null": target_non_null,
        "matched_non_null": matched_non_null,
        "target_null": target_null,
        "hallucinated": hallucinated,
        "wrong_value": wrong_value,
    }


def run_gate(args: argparse.Namespace) -> bool:
    manifest, tokenizer, model, torch = load_candidate(args.candidate)
    golden_path = args.golden or (
        Path("datasets") / manifest["dataset_version"] / "golden.jsonl"
    )
    if not golden_path.exists():
        raise FileNotFoundError(golden_path)
    records = [record for _, record in iter_jsonl(golden_path)]
    if len(records) < args.min_golden:
        raise RuntimeError(
            f"Golden set has {len(records)} records; minimum is {args.min_golden}"
        )

    golden_sha256 = sha256_jsonl(golden_path)

    total_fields = matched_fields = 0
    valid_json = schema_valid = exact_documents = 0
    extra_fields = 0

    total_target_non_null = 0
    total_matched_non_null = 0
    total_target_null = 0
    total_hallucinated = 0
    total_wrong_value = 0

    errors: list[dict[str, Any]] = []

    per_field_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "matched": 0})
    per_type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total_docs": 0, "exact_docs": 0, "total_fields": 0, "matched_fields": 0})

    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata", {})
        document_type = metadata.get("document_type") if isinstance(metadata, dict) else None
        doc_type_str = str(document_type or "UNKNOWN")
        schema = load_schema(args.schema_dir, str(document_type) if document_type else None)
        try:
            predicted = generate_prediction(
                tokenizer, model, torch, record, args.max_new_tokens
            )
            result = evaluate_record(predicted, record["output"], schema)
        except Exception as exc:
            target_data = record.get("output", {})
            result = {
                "json_valid": False,
                "schema_valid": False,
                "matched": 0,
                "total": len(target_data),
                "matched_field_names": [],
                "all_field_names": list(target_data.keys()),
                "error": str(exc),
                "target_non_null": sum(1 for v in target_data.values() if v is not None),
                "matched_non_null": 0,
                "target_null": sum(1 for v in target_data.values() if v is None),
                "hallucinated": 0,
                "wrong_value": 0,
            }

        total_fields += result["total"]
        matched_fields += result["matched"]
        valid_json += int(result["json_valid"])
        schema_valid += int(result["schema_valid"])
        exact_documents += int(result.get("exact_document", False))
        extra_fields += int(bool(result.get("extra_fields")))

        total_target_non_null += result.get("target_non_null", 0)
        total_matched_non_null += result.get("matched_non_null", 0)
        total_target_null += result.get("target_null", 0)
        total_hallucinated += result.get("hallucinated", 0)
        total_wrong_value += result.get("wrong_value", 0)

        # Track per field
        for f in result.get("all_field_names", []):
            per_field_stats[f]["expected"] += 1
        for f in result.get("matched_field_names", []):
            per_field_stats[f]["matched"] += 1

        # Track per document type
        per_type_stats[doc_type_str]["total_docs"] += 1
        per_type_stats[doc_type_str]["exact_docs"] += int(result.get("exact_document", False))
        per_type_stats[doc_type_str]["total_fields"] += result["total"]
        per_type_stats[doc_type_str]["matched_fields"] += result["matched"]

        if result.get("error") or not result["schema_valid"]:
            errors.append({"index": index, **result})

    field_accuracy = matched_fields / total_fields if total_fields else 0.0
    json_rate = valid_json / len(records)
    schema_rate = schema_valid / len(records)

    non_null_recall = total_matched_non_null / total_target_non_null if total_target_non_null > 0 else 0.0
    wrong_value_rate = total_wrong_value / total_target_non_null if total_target_non_null > 0 else 0.0
    hallucination_rate = total_hallucinated / total_target_null if total_target_null > 0 else 0.0

    per_field_accuracy = {
        field: {
            "expected": stats["expected"],
            "matched": stats["matched"],
            "accuracy": round(stats["matched"] / stats["expected"], 6) if stats["expected"] > 0 else 0.0,
        }
        for field, stats in sorted(per_field_stats.items())
    }

    per_document_type = {
        dtype: {
            "doc_count": stats["total_docs"],
            "exact_document_rate": round(stats["exact_docs"] / stats["total_docs"], 6) if stats["total_docs"] > 0 else 0.0,
            "field_accuracy": round(stats["matched_fields"] / stats["total_fields"], 6) if stats["total_fields"] > 0 else 0.0,
        }
        for dtype, stats in sorted(per_type_stats.items())
    }

    report = {
        "candidate_version": manifest["candidate_version"],
        "dataset_version": manifest["dataset_version"],
        "golden_sha256": golden_sha256,
        "evaluated_docs": len(records),
        "field_accuracy": round(field_accuracy, 6),
        "non_null_recall": round(non_null_recall, 6),
        "wrong_value_rate": round(wrong_value_rate, 6),
        "hallucination_rate": round(hallucination_rate, 6),
        "json_valid_rate": round(json_rate, 6),
        "schema_valid_rate": round(schema_rate, 6),
        "exact_document_rate": round(exact_documents / len(records), 6),
        "documents_with_extra_fields": extra_fields,
        "per_field_accuracy": per_field_accuracy,
        "per_document_type": per_document_type,
        "thresholds": {
            "min_field_accuracy": args.min_field_accuracy,
            "min_schema_rate": args.min_schema_rate,
        },
        "errors_sample": errors[:50],
        "evaluated_at": utc_now(),
    }

    baseline_ok = True
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        comparable = baseline.get("golden_sha256") == golden_sha256

        if not comparable and not args.allow_baseline_mismatch:
            baseline_ok = False
        else:
            baseline_ok = (
                field_accuracy >= float(baseline.get("field_accuracy", 0.0))
                and schema_rate >= float(baseline.get("schema_valid_rate", 0.0))
            )

        report["baseline"] = {
            "report": str(args.baseline_report),
            "field_accuracy": baseline.get("field_accuracy"),
            "schema_valid_rate": baseline.get("schema_valid_rate"),
            "golden_sha256_match": comparable,
            "not_regressed": baseline_ok,
        }

    passed = (
        field_accuracy >= args.min_field_accuracy
        and schema_rate >= args.min_schema_rate
        and baseline_ok
    )
    report["gate_status"] = "PASSED" if passed else "REJECTED"
    write_json_atomic(args.candidate / "evaluation_report.json", report)
    print(
        f"Gate {report['gate_status']}: field_accuracy={field_accuracy:.2%}, "
        f"schema_rate={schema_rate:.2%}, docs={len(records)}, golden_sha256={golden_sha256[:10]}..."
    )
    return passed


def main() -> int:
    try:
        return 0 if run_gate(parse_args()) else 2
    except Exception as exc:
        print(f"EVALUATION_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())