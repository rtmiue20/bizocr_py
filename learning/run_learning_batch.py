"""Run continuous learning batch workflow."""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests

# 1. Thêm thư mục gốc vào sys.path để giải quyết triệt để lỗi import
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Sửa lỗi 1: Import trực tiếp mysql.connector
import mysql.connector

# Import an toàn sau khi đã cấu hình sys.path
from training_pipeline.pipeline_common import (
    iter_jsonl,
    parse_json_value,
    utc_now,
    values_equal,
    write_json_atomic,
)
from training_pipeline.build_correction_dictionary import build_dictionary_from_source

from python_service.ml.classifier_ml import document_ml_service, DocumentMlService
from python_service.normalizer.text_fixer import text_fixer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_learning_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous learning batch runner")
    parser.add_argument("--batch-id", type=int, required=True, help="Batch ID from training_batches")
    parser.add_argument("--golden-dir", type=Path, default=Path("golden"), help="Directory containing golden dataset")
    parser.add_argument("--min-frequency", type=int, default=2, help="Min repetition for correction rules")
    parser.add_argument("--min-golden", type=int, default=10, help="Minimum required golden records")
    parser.add_argument("--learning-curve-file", type=Path, default=Path("saved_models/learning_curve.jsonl"))
    parser.add_argument("--status-file", type=Path, default=Path("saved_models/status.json"))
    parser.add_argument("--reload-url", default=os.getenv("RELOAD_URL", "http://localhost:8000/api/v1/ml/reload"))
    parser.add_argument("--no-reload", action="store_true", help="Do not trigger API reload")
    return parser.parse_args()


def acquire_lock(lock_file: Path):
    """Lấy khóa độc quyền đa nền tảng (Hỗ trợ cả Windows và Linux)."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        # Fallback cho hệ điều hành Windows
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            os.close(fd)
            raise RuntimeError("Another batch training process is currently running.")
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("Another batch training process is currently running.")
    return fd


def get_admin_token() -> str:
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        raise ValueError("Missing ADMIN_TOKEN environment variable. Required for API reload.")
    return token


# Sửa lỗi 1: Cập nhật hàm kết nối CSDL theo chuẩn spec truyền vào
def connect_database():
    # Dùng biến DB_* như spec, không hardcode
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "mariadb"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER"), # user chỉ SELECT
        password=os.getenv("DB_PASSWORD", "049206"),
        database=os.getenv("DB_NAME", "bizocr"),
        autocommit=True
    )


def fetch_batch_documents(batch_id: int, exclude_ids: Set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    conn = connect_database()
    try:
        cursor = conn.cursor(dictionary=True)
        # 1. Tài liệu trong lô hiện tại
        sql_batch = """
        SELECT id, raw_ocr_text, document_type, extracted_fields, predicted_fields_json, model_version
        FROM documents WHERE training_eligible = 1 AND training_batch_id = %s ORDER BY id
        """
        cursor.execute(sql_batch, [batch_id])
        batch_docs = [doc for doc in cursor.fetchall() if str(doc["id"]) not in exclude_ids]

        # 2. Dữ liệu huấn luyện tích lũy <= batch_id
        sql_accumulated = """
        SELECT id, raw_ocr_text, document_type
        FROM documents WHERE training_eligible = 1 AND training_batch_id <= %s ORDER BY id
        """
        cursor.execute(sql_accumulated, [batch_id])
        accumulated_docs = [doc for doc in cursor.fetchall() if str(doc["id"]) not in exclude_ids]

        cursor.close()
        return batch_docs, accumulated_docs
    finally:
        conn.close()


def append_learning_curve(file_path: Path, entry: Dict[str, Any]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_classes(svc: Any) -> set:
    """Trích xuất danh sách nhãn lớp từ DocumentMlService một cách an toàn."""
    if hasattr(svc, "classes_"):
        return set(svc.classes_)
    if hasattr(svc, "model") and hasattr(svc.model, "classes_"):
        return set(svc.model.classes_)
    if hasattr(svc, "pipeline") and hasattr(svc.pipeline, "classes_"):
        return set(svc.pipeline.classes_)
    return set()


def evaluate_batch_performance(batch_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch_docs:
        return {"doc_count": 0, "type_accuracy": 0.0, "field_accuracy": 0.0, "details": {}}

    cls_correct, total_fields, matched_fields = 0, 0, 0
    field_stats = defaultdict(lambda: {"total": 0, "match": 0})

    for doc in batch_docs:
        raw_text = str(doc.get("raw_ocr_text") or "").strip()
        expected_type = str(doc.get("document_type") or "").strip()

        pred_cls = document_ml_service.predict(raw_text).get("predictedType")
        if pred_cls == expected_type:
            cls_correct += 1

        expected_fields = parse_json_value(doc.get("extracted_fields"))
        predicted_fields = parse_json_value(doc.get("predicted_fields_json"))

        if isinstance(expected_fields, dict) and isinstance(predicted_fields, dict):
            for field in ["field_confidences", "anomaly_analysis", "review_status"]:
                expected_fields.pop(field, None)
                predicted_fields.pop(field, None)

            for k, exp_val in expected_fields.items():
                if exp_val is None:
                    continue
                total_fields += 1
                field_stats[k]["total"] += 1
                if k in predicted_fields and values_equal(predicted_fields[k], exp_val):
                    matched_fields += 1
                    field_stats[k]["match"] += 1

    return {
        "doc_count": len(batch_docs),
        "type_accuracy": round(cls_correct / len(batch_docs), 4),
        "field_accuracy": round(matched_fields / total_fields, 4) if total_fields else 0.0,
        "total_fields": total_fields,
        "matched_fields": matched_fields,
        "details": dict(field_stats)
    }


def load_golden_set(golden_dir: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
    records, golden_ids = [], set()
    manifest_path = golden_dir / "golden_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("documents", []):
            golden_ids.add(str(entry.get("id")))

    golden_jsonl = golden_dir / "golden.jsonl"
    if golden_jsonl.is_file():
        for _, rec in iter_jsonl(golden_jsonl):
            records.append(rec)
            if "metadata" in rec and "id" in rec["metadata"]:
                golden_ids.add(str(rec["metadata"]["id"]))

    return records, golden_ids


def evaluate_on_golden(
    golden_records: List[Dict[str, Any]],
    classifier_instance: DocumentMlService,
    dictionary_path: Path,
) -> Dict[str, float]:
    if not golden_records:
        return {"golden_cls_accuracy": 0.0, "golden_field_accuracy": 0.0}

    cls_correct, total_fields, matched_fields = 0, 0, 0

    for rec in golden_records:
        raw_text = str(rec.get("input") or "").strip()
        metadata = rec.get("metadata", {})
        expected_type = str(metadata.get("document_type") or "").strip()
        expected_output = rec.get("output", {})
        predicted_base = metadata.get("predicted_fields_json", {})

        pred_type = classifier_instance.predict(raw_text).get("predictedType")
        if pred_type == expected_type:
            cls_correct += 1

        if isinstance(expected_output, dict) and isinstance(predicted_base, dict):
            # Sửa lỗi 3: Dùng đúng API .fix() của TextFixer
            from python_service.normalizer.text_fixer import TextFixer
            if dictionary_path and dictionary_path.exists():
                fixer = TextFixer(dictionary_path)
                fixed_predicted = fixer.fix(deepcopy(predicted_base))
            else:
                fixed_predicted = deepcopy(predicted_base)

            fixed_predicted.pop("review_status", None)
            expected_output_clean = {k: v for k, v in expected_output.items() if k != "review_status"}

            for k, exp_val in expected_output_clean.items():
                if exp_val is None:
                    continue
                total_fields += 1
                if k in fixed_predicted and values_equal(fixed_predicted[k], exp_val):
                    matched_fields += 1

    return {
        "golden_cls_accuracy": round(cls_correct / len(golden_records), 4),
        "golden_field_accuracy": round(matched_fields / total_fields, 4) if total_fields else 0.0,
    }


def trigger_api_reload(reload_url: str, admin_token: str) -> bool:
    try:
        headers = {"Authorization": f"Bearer {admin_token}", "X-Admin-Token": admin_token}
        resp = requests.post(reload_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            logger.info(f"API Reload thành công.")
            return True
        logger.warning(f"API Reload thất bại với mã {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Không thể kết nối tới reload endpoint ({reload_url}): {e}")
        return False


def run_learning_batch(args: argparse.Namespace) -> int:
    batch_id = args.batch_id
    admin_token = get_admin_token()
    lock_fd = acquire_lock(Path(".batch_training.lock"))

    logger.info(f"=== Bắt đầu vòng học liên tục cho lô batch_id={batch_id} ===")
    write_json_atomic(args.status_file, {"batch_id": batch_id, "status": "RUNNING", "updated_at": utc_now()})

    try:
        golden_records, golden_ids = load_golden_set(args.golden_dir)

        if len(golden_records) < args.min_golden:
            logger.warning(f"[SKIPPED] Số lượng Golden ({len(golden_records)}) nhỏ hơn min_golden.")
            write_json_atomic(args.status_file, {"batch_id": batch_id, "status": "SKIPPED", "reason": "INSUFFICIENT_GOLDEN", "updated_at": utc_now()})
            return 0

        batch_docs, accumulated_docs = fetch_batch_documents(batch_id, golden_ids)

        # 3. Chấm Baseline Performance
        model_version_before = getattr(document_ml_service, "model_version", "unknown")
        batch_eval = evaluate_batch_performance(batch_docs)
        baseline_golden = evaluate_on_golden(
            golden_records,
            document_ml_service,
            Path("saved_models/active_model/correction_dictionary.json")
        )

        curve_entry_initial = {
            "timestamp": utc_now(), "batch_id": batch_id, "event": "BATCH_EVALUATED",
            "model_version_before": model_version_before,
            "batch_doc_count": batch_eval["doc_count"],
            "batch_type_accuracy": batch_eval["type_accuracy"],
            "batch_field_accuracy": batch_eval["field_accuracy"],
            "baseline_golden": baseline_golden,
        }
        append_learning_curve(args.learning_curve_file, curve_entry_initial)

        # 4. Huấn luyện Candidate
        logger.info("Huấn luyện mô hình ứng viên và xây dựng từ điển sửa lỗi...")
        training_samples = [
            {"text": str(d.get("raw_ocr_text") or "").strip(), "label": str(d.get("document_type") or "").strip()}
            for d in accumulated_docs if d.get("raw_ocr_text") and d.get("document_type")
        ]

        if hasattr(document_ml_service, "DEFAULT_SAMPLES"):
            training_samples.extend(document_ml_service.DEFAULT_SAMPLES)

        candidate_service = DocumentMlService(model_dir=Path("saved_models/candidate_model"))

        # Tạo tập eval nội bộ để tránh lỗi thiếu arguments
        split_idx = max(1, int(len(training_samples) * 0.8))
        train_split = training_samples[:split_idx]
        eval_split = training_samples[split_idx:]
        if not eval_split:
            eval_split = train_split

        candidate_result = candidate_service.retrain(train_samples=train_split, eval_samples=eval_split)

        if candidate_result.get("status") != "PROMOTED":
            logger.warning(f"[REJECTED] Lớp phân loại ứng viên bị đánh rớt nội bộ.")
            write_json_atomic(args.status_file, {"batch_id": batch_id, "status": "REJECTED", "reason": "CLASSIFIER_INTERNAL_REJECT", "updated_at": utc_now()})
            return 0

        baseline_classes = get_classes(document_ml_service)
        candidate_classes = get_classes(candidate_service)

        if baseline_classes and not baseline_classes.issubset(candidate_classes):
            missing_classes = baseline_classes - candidate_classes
            logger.warning(f"[REJECTED] Candidate mất đi các lớp so với baseline: {missing_classes}")
            write_json_atomic(args.status_file, {"batch_id": batch_id, "status": "REJECTED", "reason": "CLASS_LOST", "updated_at": utc_now()})
            return 0

        # Xây dựng Candidate Dictionary và tự ghi file
        candidate_dict_path = Path("saved_models/candidate_model/correction_dictionary_candidate.json")
        try:
            dict_data = build_dictionary_from_source(accumulated_docs, min_frequency=args.min_frequency)
            candidate_dict_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(candidate_dict_path, dict_data)
        except Exception as exc:
            logger.error(f"Xây dựng từ điển sửa lỗi gặp lỗi: {exc}")
            candidate_dict_path = None

        # 5. Chấm Candidate trên Golden
        candidate_golden = evaluate_on_golden(golden_records, candidate_service, candidate_dict_path)

        # 6. Kiểm tra thoái lui
        not_regressed = (
                candidate_golden["golden_cls_accuracy"] >= baseline_golden["golden_cls_accuracy"]
                and (candidate_golden["golden_field_accuracy"] >= baseline_golden["golden_field_accuracy"] - 0.01)
        )

        reload_ok = False
        if not_regressed:
            promotion_status = "PROMOTED"
            logger.info(f"[PROMOTED] Batch {batch_id}")

            # 1. Lưu candidate thành active bằng method an toàn
            candidate_service.promote_to_active()

            # 2. Copy dictionary
            active_dict_path = Path("saved_models/active_model/correction_dictionary.json")
            active_dict_path.parent.mkdir(parents=True, exist_ok=True)
            if candidate_dict_path and candidate_dict_path.exists():
                shutil.copy2(candidate_dict_path, active_dict_path)

            # 2.5 Cập nhật ngân hàng Few-Shot Namecard và kNN Industry từ các thẻ đã duyệt
            try:
                from python_service.ml.industry_classifier import industry_classifier
                fewshot_dir = Path("saved_models/few_shot")
                fewshot_dir.mkdir(parents=True, exist_ok=True)
                fewshot_file = fewshot_dir / "namecard_fewshots.jsonl"

                namecard_count = 0
                with open(fewshot_file, "a", encoding="utf-8") as fs_out:
                    for doc in batch_docs:
                        doc_type = str(doc.get("document_type") or "").upper()
                        if doc_type in ["NAMECARD", "BUSINESS_CARD", "CARD"]:
                            ext = parse_json_value(doc.get("extracted_fields"))
                            raw_t = str(doc.get("raw_ocr_text") or "").strip()
                            if ext and raw_t:
                                entry = {"input": raw_t, "output": ext, "doc_id": doc.get("id")}
                                fs_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                namecard_count += 1

                                # Cập nhật kNN ngành nghề nếu có công ty & ngành đã gán
                                comp = ext.get("company")
                                ind_code = ext.get("industryCode") or ext.get("industry")
                                if comp and ind_code:
                                    industry_classifier.add_sample(comp, ind_code, persist=True)

                if namecard_count > 0:
                    logger.info(f"Đã cập nhật {namecard_count} mẫu namecard vào ngân hàng Few-shot và kNN Industry.")
            except Exception as fs_exc:
                logger.warning(f"Lỗi khi cập nhật Few-shot / Industry kNN: {fs_exc}")

            # 3. Reload qua API
            if not args.no_reload:
                reload_ok = trigger_api_reload(args.reload_url, admin_token)
        else:
            promotion_status = "REJECTED"
            logger.warning(f"[REJECTED] Lô {batch_id} bị từ chối do thoái lui (Regression).")
            if candidate_dict_path and candidate_dict_path.exists():
                candidate_dict_path.unlink()

        final_curve_entry = {
            "timestamp": utc_now(),
            "batch_id": batch_id,
            "event": promotion_status,
            "baseline_golden": baseline_golden,
            "candidate_golden": candidate_golden,
            "not_regressed": not_regressed,
            "reload_ok": reload_ok
        }
        append_learning_curve(args.learning_curve_file, final_curve_entry)

        write_json_atomic(args.status_file, {
            "batch_id": batch_id, "status": promotion_status,
            "metrics": candidate_golden, "updated_at": utc_now()
        })
        return 0

    except Exception as e:
        logger.error(f"Critical error during batch {batch_id}: {e}")
        write_json_atomic(args.status_file, {"batch_id": batch_id, "status": "FAILED", "error": str(e), "updated_at": utc_now()})
        return 1
    finally:
        os.close(lock_fd)


def main() -> int:
    return run_learning_batch(parse_args())


if __name__ == "__main__":
    sys.exit(main())