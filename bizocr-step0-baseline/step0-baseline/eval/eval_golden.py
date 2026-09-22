#!/usr/bin/env python3
"""
Bước 0 - Đo baseline trên golden set.

Cấu trúc golden set (mỗi tài liệu 1 cặp file, đã ẩn danh):
    golden/<id>.jpg           ảnh (chỉ cần khi dùng --api-url)
    golden/<id>.json          đáp án đúng:
        {"documentType": "NAMECARD",
         "fields": {"companyName": "...", "taxCode": "...", "phone": "..."}}
        (trường không có trong tài liệu: để chuỗi rỗng hoặc bỏ qua)

Hai cách lấy kết quả dự đoán của hệ thống:
  A) --predictions-dir DIR : mỗi <id>.json cùng schema đáp án, thêm tùy chọn:
        "latencyMs": 4200, "needsReview": false, "fieldSources": {"taxCode": "RULE"}
     (xuất từ DB/log của pipeline - cách này không phụ thuộc API nên nên dùng trước)
  B) --api-url URL [--token JWT] : gửi ảnh tới 1 endpoint ĐỒNG BỘ trả JSON kết quả.
     Sửa hàm pred_from_response() cho khớp DTO thật của bạn.

Ví dụ:
    python eval_golden.py --golden golden --predictions-dir preds --save baseline.json
    python eval_golden.py --golden golden --predictions-dir preds_new --compare baseline.json
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ADAPT: các trường so khớp theo chữ số (bỏ mọi ký tự không phải số)
NUMERIC_FIELDS = {"taxCode", "mst", "phone", "invoiceNumber"}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm(field: str, value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if field in NUMERIC_FIELDS:
        return re.sub(r"\D", "", s)
    s = strip_accents(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pred_from_response(resp: dict) -> dict:
    """ADAPT: đổi cho khớp JSON thật của endpoint. Trả về dict cùng schema đáp án."""
    return {
        "documentType": resp.get("documentType"),
        "fields": resp.get("fields", {}),
        "needsReview": resp.get("needsReview", False),
    }


def call_api(image_path: Path, url: str, token: str | None) -> tuple[dict, float]:
    import requests  # cài khi dùng chế độ B: pip install requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    t0 = time.perf_counter()
    with open(image_path, "rb") as f:
        r = requests.post(url, files={"file": (image_path.name, f)}, headers=headers, timeout=300)
    r.raise_for_status()
    return pred_from_response(r.json()), (time.perf_counter() - t0) * 1000.0


def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[int(round(p * (len(v) - 1)))]


def evaluate(golden: dict, preds: dict) -> dict:
    conf = Counter()
    cls_ok = cls_n = missing = review = 0
    latencies = []
    fs = defaultdict(lambda: {"expected": 0, "predicted": 0, "correct": 0, "wrong_nonempty": 0})

    for doc_id, exp in golden.items():
        pred = preds.get(doc_id)
        if pred is None:
            missing += 1
            continue
        cls_n += 1
        e_type, p_type = exp.get("documentType"), pred.get("documentType")
        conf[(e_type, p_type)] += 1
        cls_ok += int(e_type == p_type)
        review += int(bool(pred.get("needsReview")))
        if pred.get("latencyMs") is not None:
            latencies.append(float(pred["latencyMs"]))

        e_fields, p_fields = exp.get("fields", {}) or {}, pred.get("fields", {}) or {}
        for name in set(e_fields) | set(p_fields):
            e, p = norm(name, e_fields.get(name)), norm(name, p_fields.get(name))
            st = fs[name]
            if e:
                st["expected"] += 1
            if p:
                st["predicted"] += 1
                if p == e:
                    st["correct"] += 1
                else:
                    st["wrong_nonempty"] += 1  # đọc sai hoặc bịa ra trường không có

    exp_total = sum(s["expected"] for s in fs.values())
    pred_total = sum(s["predicted"] for s in fs.values())
    correct = sum(s["correct"] for s in fs.values())
    wrong = sum(s["wrong_nonempty"] for s in fs.values())
    summary = {
        "documents_evaluated": cls_n,
        "documents_missing_prediction": missing,
        "classification_accuracy": cls_ok / cls_n if cls_n else None,
        "field_recall": correct / exp_total if exp_total else None,       # đúng / có trong đáp án
        "field_precision": correct / pred_total if pred_total else None,  # đúng / hệ thống điền
        "wrong_nonempty_rate": wrong / pred_total if pred_total else None,
        "review_rate": review / cls_n if cls_n else None,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
    }
    per_field = {
        n: {**s, "recall": s["correct"] / s["expected"] if s["expected"] else None}
        for n, s in sorted(fs.items())
    }
    confusion = {f"{a} -> {b}": c for (a, b), c in sorted(conf.items(), key=lambda x: -x[1]) if a != b}
    return {"summary": summary, "per_field": per_field, "misclassified": confusion}


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:.0f}" if isinstance(v, float) else str(v))


def print_report(result: dict, baseline: dict | None):
    print("| metric | value | baseline | delta |\n|---|---|---|---|")
    for k, v in result["summary"].items():
        b = baseline["summary"].get(k) if baseline else None
        d = (v - b) if isinstance(v, (int, float)) and isinstance(b, (int, float)) else None
        print(f"| {k} | {fmt(v)} | {fmt(b)} | {fmt(d) if d is not None else '-'} |")
    print("\n| field | expected | predicted | correct | wrong_nonempty | recall |\n|---|---|---|---|---|---|")
    for n, s in result["per_field"].items():
        print(f"| {n} | {s['expected']} | {s['predicted']} | {s['correct']} | {s['wrong_nonempty']} | {fmt(s['recall'])} |")
    if result["misclassified"]:
        print("\nPhân loại nhầm:")
        for k, c in result["misclassified"].items():
            print(f"  {c} x {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--predictions-dir", type=Path)
    ap.add_argument("--api-url")
    ap.add_argument("--token")
    ap.add_argument("--save", type=Path, help="lưu kết quả làm baseline")
    ap.add_argument("--compare", type=Path, help="so sánh với baseline đã lưu")
    args = ap.parse_args()
    if not args.predictions_dir and not args.api_url:
        ap.error("cần --predictions-dir hoặc --api-url")

    golden = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.golden.glob("*.json"))}
    if not golden:
        sys.exit(f"Không thấy file .json đáp án trong {args.golden}")

    preds = {}
    for doc_id in golden:
        if args.predictions_dir:
            f = args.predictions_dir / f"{doc_id}.json"
            if f.exists():
                preds[doc_id] = json.loads(f.read_text(encoding="utf-8"))
        else:
            img = next((p for ext in (".jpg", ".jpeg", ".png") if (p := args.golden / f"{doc_id}{ext}").exists()), None)
            if img is None:
                continue
            pred, ms = call_api(img, args.api_url, args.token)
            pred["latencyMs"] = ms
            preds[doc_id] = pred

    result = evaluate(golden, preds)
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    print_report(result, baseline)
    if args.save:
        args.save.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã lưu baseline: {args.save}")


if __name__ == "__main__":
    main()
