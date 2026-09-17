"""Train a LoRA adapter from an explicit, validated dataset version.
This script uses Transformers + PEFT.  It saves an adapter candidate and a
manifest; it does not change the active production model.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

try:
    from training_pipeline.pipeline_common import sha256_file, sha256_jsonl, utc_now, write_json_atomic
except ImportError:
    from pipeline_common import sha256_file, sha256_jsonl, utc_now, write_json_atomic

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("saved_models"))
    parser.add_argument(
        "--candidate-version",
        default=None,
        help="Explicit candidate directory name. Must be unique for each run.",
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()
def load_dependencies():
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install transformers, datasets, "
            "peft, accelerate and torch."
        ) from exc
    return locals()
def format_record(tokenizer, record: dict) -> str:
    messages = [
        {"role": "system", "content": str(record["instruction"])},
        {"role": "user", "content": str(record["input"])},
        {
            "role": "assistant",
            "content": json.dumps(record["output"], ensure_ascii=False, sort_keys=True),
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return (
        f"System:\n{messages[0]['content']}\n"
        f"User:\n{messages[1]['content']}\n"
        f"Assistant:\n{messages[2]['content']}"
    )
def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record.get("output"), dict):
                    raise ValueError(f"Output must be an object in {path}")
                records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records
def train(args: argparse.Namespace) -> Path:
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    train_file = args.dataset / "train.jsonl"
    val_file = args.dataset / "val.jsonl"
    split_manifest = args.dataset / "split_manifest.json"
    for required in (train_file, val_file, split_manifest):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")
    deps = load_dependencies()
    torch = deps["torch"]
    Dataset = deps["Dataset"]
    LoraConfig = deps["LoraConfig"]
    TaskType = deps["TaskType"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    EarlyStoppingCallback = deps["EarlyStoppingCallback"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    train_records = load_records(train_file)
    val_records = load_records(val_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_texts = [format_record(tokenizer, record) for record in train_records]
    val_texts = [format_record(tokenizer, record) for record in val_records]
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
    train_tokens = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokens = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    use_4bit = args.load_in_4bit and torch.cuda.is_available()
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError as exc:
            raise RuntimeError("--load-in-4bit requires bitsandbytes") from exc
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if use_4bit and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    version = args.candidate_version or (
        f"lora-{args.dataset.name}-{utc_now().replace(':', '').replace('+00:00', 'Z')}"
    )
    output_dir = args.output_root / version
    output_dir.mkdir(parents=True, exist_ok=False)
    training_args_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    # Transformers renamed this argument in newer releases.
    import inspect
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_args_kwargs)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=val_tokens,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_files = sorted(
        str(path.relative_to(adapter_dir))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    file_hashes = {
        name: sha256_file(adapter_dir / name) for name in adapter_files
    }
    manifest = {
        "candidate_version": version,
        "base_model": args.base_model,
        "dataset_version": args.dataset.name,
        "dataset_source_sha256": sha256_jsonl(train_file),
        "trained_at": utc_now(),
        # Relative to candidate_dir so the candidate is portable between
        # machines and can be verified by evaluate/register without path
        # concatenation bugs.
        "adapter_path": "adapter",
        "adapter_files": file_hashes,
        "hyperparameters": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
            "load_in_4bit": use_4bit,
        },
        "status": "TRAINED",
    }
    write_json_atomic(output_dir / "candidate_manifest.json", manifest)
    print(f"Candidate adapter saved to {output_dir}")
    return output_dir
def main() -> int:
    try:
        train(parse_args())
        return 0
    except Exception as exc:
        print(f"TRAINING_FAILED: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())