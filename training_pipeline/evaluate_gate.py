"""Run a real candidate-model evaluation against an isolated golden set."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from training_pipeline.pipeline_common import (
        extract_json_object,
        iter_jsonl,
        load_schema,
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
        return {"json_valid": False, "schema_valid": False, "matched": 0, "total": len(target)}
    schema_errors = validate_schema(predicted, schema)
    matched = sum(
        1 for key, value in target.items()
        if key in predicted and values_equal(predicted[key], value)
    )
    return {
        "json_valid": True,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors[:5],
        "matched": matched,
        "total": len(target),
        "exact_document": matched == len(target) and all(
            key in target for key in predicted
        ),
        "extra_fields": sorted(set(predicted) - set(target)),
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
    total_fields = matched_fields = 0
    valid_json = schema_valid = exact_documents = 0
    extra_fields = 0
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        metadata = record.get("metadata", {})
        document_type = metadata.get("document_type") if isinstance(metadata, dict) else None
        schema = load_schema(args.schema_dir, str(document_type) if document_type else None)
        try:
            predicted = generate_prediction(
                tokenizer, model, torch, record, args.max_new_tokens
            )
            result = evaluate_record(predicted, record["output"], schema)
        except Exception as exc:
            result = {
                "json_valid": False,
                "schema_valid": False,
                "matched": 0,
                "total": len(record.get("output", {})),
                "error": str(exc),
            }
        total_fields += result["total"]
        matched_fields += result["matched"]
        valid_json += int(result["json_valid"])
        schema_valid += int(result["schema_valid"])
        exact_documents += int(result.get("exact_document", False))
        extra_fields += int(bool(result.get("extra_fields")))
        if result.get("error") or not result["schema_valid"]:
            errors.append({"index": index, **result})
    field_accuracy = matched_fields / total_fields if total_fields else 0.0
    json_rate = valid_json / len(records)
    schema_rate = schema_valid / len(records)
    report = {
        "candidate_version": manifest["candidate_version"],
        "dataset_version": manifest["dataset_version"],
        "evaluated_docs": len(records),
        "field_accuracy": round(field_accuracy, 6),
        "json_valid_rate": round(json_rate, 6),
        "schema_valid_rate": round(schema_rate, 6),
        "exact_document_rate": round(exact_documents / len(records), 6),
        "documents_with_extra_fields": extra_fields,
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
        baseline_ok = (
            field_accuracy >= float(baseline.get("field_accuracy", 0.0))
            and schema_rate >= float(baseline.get("schema_valid_rate", 0.0))
        )
        report["baseline"] = {
            "report": str(args.baseline_report),
            "field_accuracy": baseline.get("field_accuracy"),
            "schema_valid_rate": baseline.get("schema_valid_rate"),
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
        f"schema_rate={schema_rate:.2%}, docs={len(records)}"
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