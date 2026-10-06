import hmac
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from python_service.ml.classifier_ml import document_ml_service
from python_service.normalizer.text_fixer import text_fixer
from python_service.ml.namecard_field_tagger import namecard_tagger
from python_service.ml.industry_classifier import industry_classifier
from python_service.services.dedup import dedup_service

logger = logging.getLogger(__name__)

# Router chính /api/v1/ml và Router alias /ml
router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])
alias_router = APIRouter(prefix="/ml", tags=["Machine Learning Short URL"])


def require_admin_token(
    authorization: Optional[str] = Header(None),
    x_admin_token: Optional[str] = Header(None),
) -> str:
    """
    Hàm phụ thuộc xác thực Token quản trị.
    Từ chối tuyệt đối (Fail Closed) nếu ADMIN_TOKEN chưa được cài đặt trong môi trường.
    """
    admin_token = os.getenv("ADMIN_TOKEN", "dev-admin-token")
    if not admin_token:
        logger.error("ADMIN_TOKEN chưa được cấu hình trong biến môi trường (Fail Closed).")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: ADMIN_TOKEN environment variable is missing",
        )

    token = x_admin_token
    if not token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        else:
            token = authorization

    if not token or not hmac.compare_digest(token, admin_token):
        logger.warning("Truy cập bị từ chối: Token quản trị không hợp lệ hoặc thiếu")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Valid Admin token required",
        )
    return token


# --- Trạng thái tiến trình học (Learning Process Tracker với File Lock & Persistent Status) ---

class LearningJobTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.status: str = "IDLE"  # IDLE | RUNNING | COMPLETED | PROMOTED | REJECTED | FAILED
        self.batch_id: Optional[int] = None
        self.job_id: Optional[str] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.message: Optional[str] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.lock_file = Path("saved_models/learning.lock")
        self.status_file = Path("learning/status.json")
        self.fallback_status_file = Path("saved_models/learning_status.json")

    def _is_job_running(self) -> bool:
        """Kiểm tra tiến trình trong bộ nhớ hoặc kiểm tra an toàn qua psutil trên đĩa."""
        if self.process is not None and self.process.poll() is None:
            return True

        if self.lock_file.exists():
            try:
                content = self.lock_file.read_text(encoding="utf-8").strip()
                pid: Optional[int] = None
                saved_create_time: Optional[float] = None

                if content.startswith("{"):
                    try:
                        data = json.loads(content)
                        pid = data.get("pid")
                        saved_create_time = data.get("create_time")
                    except json.JSONDecodeError:
                        pass
                elif content.isdigit():
                    pid = int(content)

                if pid is not None and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        # Kiểm tra xem tiến trình còn sống và không phải zombie
                        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                            # Nếu có thông tin create_time, đối soát để tránh dùng nhầm PID bị tái sử dụng
                            if saved_create_time is None or abs(proc.create_time() - saved_create_time) < 2.0:
                                return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Nếu tiến trình không tồn tại hoặc PID đã bị gán cho tiến trình khác -> Xóa stale lock file
                try:
                    self.lock_file.unlink(missing_ok=True)
                except OSError:
                    pass
            except Exception as e:
                logger.warning(f"Lỗi kiểm tra lock file: {e}")

        return False

    def start_job(self, batch_id: int) -> Dict[str, Any]:
        with self._lock:
            if self._is_job_running():
                return {"running": True, "batch_id": self.batch_id, "job_id": self.job_id}

            now_str = datetime.now(timezone.utc).isoformat()
            self.job_id = f"job-{batch_id}-{int(datetime.now().timestamp())}"
            self.batch_id = batch_id
            self.status = "RUNNING"
            self.started_at = now_str
            self.completed_at = None
            self.exit_code = None
            self.message = f"Batch {batch_id} learning process started"
            self.metrics = None

            cmd = [
                sys.executable,
                "-m",
                "learning.run_learning_batch",
                "--batch-id",
                str(batch_id),
            ]

            log_dir = Path("saved_models/learning_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{self.job_id}.log"

            try:
                self.lock_file.parent.mkdir(parents=True, exist_ok=True)

                with open(log_file, "w", encoding="utf-8") as log_handle:
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        cwd=str(Path.cwd()),
                    )

                if self.process:
                    try:
                        p_info = psutil.Process(self.process.pid)
                        lock_data = {
                            "pid": self.process.pid,
                            "create_time": p_info.create_time(),
                            "job_id": self.job_id,
                            "batch_id": batch_id
                        }
                        self.lock_file.write_text(json.dumps(lock_data, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        self.lock_file.write_text(str(self.process.pid), encoding="utf-8")

                logger.info(f"Đã khởi chạy background learning job {self.job_id} (PID={self.process.pid})")
                return {"running": False, "job_id": self.job_id, "batch_id": batch_id, "pid": self.process.pid}
            except Exception as e:
                self.status = "FAILED"
                self.message = f"Failed to spawn process: {e}"
                if self.lock_file.exists():
                    try:
                        self.lock_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                logger.error(self.message)
                raise HTTPException(status_code=500, detail=self.message)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            # 1. Kiểm tra tiến trình đang chạy
            if self.process is not None:
                ret = self.process.poll()
                if ret is not None and self.status == "RUNNING":
                    self.completed_at = datetime.now(timezone.utc).isoformat()
                    self.exit_code = ret
                    if self.lock_file.exists():
                        try:
                            self.lock_file.unlink(missing_ok=True)
                        except OSError:
                            pass
                    if ret == 0:
                        self.status = "COMPLETED"
                        self.message = f"Job {self.job_id} completed successfully"
                    else:
                        self.status = "FAILED"
                        self.message = f"Job {self.job_id} failed with exit code {ret}"

            # 2. Đọc thông tin chi tiết từ status.json persistent trên đĩa
            status_path = self.status_file if self.status_file.is_file() else self.fallback_status_file
            if status_path.is_file():
                try:
                    with open(status_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    if isinstance(file_data, dict):
                        if file_data.get("status"):
                            self.status = file_data.get("status")
                        if "metrics" in file_data:
                            self.metrics = file_data.get("metrics")
                        if "message" in file_data:
                            self.message = file_data.get("message")
                        if "completed_at" in file_data:
                            self.completed_at = file_data.get("completed_at")
                        if "batch_id" in file_data:
                            self.batch_id = file_data.get("batch_id")
                        if "job_id" in file_data:
                            self.job_id = file_data.get("job_id")
                except Exception as e:
                    logger.error(f"Lỗi khi đọc file status {status_path}: {e}")

            return {
                "status": self.status,
                "batch_id": self.batch_id,
                "job_id": self.job_id,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "exit_code": self.exit_code,
                "message": self.message,
                "metrics": self.metrics,
            }


job_tracker = LearningJobTracker()


# --- Payload Models ---

class PredictRequest(BaseModel):
    rawText: str


class CollectRequest(BaseModel):
    rawText: str
    label: str


class ExtractRequest(BaseModel):
    rawText: str
    documentType: Optional[str] = "NAMECARD"
    ownerCompanyId: Optional[str] = None


class IndustryPredictPayload(BaseModel):
    companyName: str
    textSnippet: Optional[str] = None


class DedupCheckPayload(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    name: Optional[str] = None
    imageHash: Optional[str] = None


class ClusterRequest(BaseModel):
    n_clusters: int


class AnomalyRequest(BaseModel):
    current_amount: float
    historical_amounts: List[float]


class LearnBatchRequest(BaseModel):
    batch_id: int


# --- API Handlers ---

def _learn_batch_handler(payload: LearnBatchRequest, token: str = Depends(require_admin_token)):
    logger.info(f"Nhận yêu cầu huấn luyện lô batch_id={payload.batch_id}")
    res = job_tracker.start_job(payload.batch_id)
    if res.get("running"):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "status": "BUSY",
                "message": f"Another batch {res.get('batch_id')} is currently running (Job {res.get('job_id')})",
                "batch_id": res.get("batch_id"),
            },
        )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "ACCEPTED",
            "message": f"Learning batch {payload.batch_id} triggered successfully",
            "batch_id": payload.batch_id,
            "job_id": res.get("job_id"),
            "pid": res.get("pid"),
        },
    )


def _learning_status_handler(token: str = Depends(require_admin_token)):
    return job_tracker.get_status()


def _reload_handler(token: str = Depends(require_admin_token)):
    cls_reloaded = document_ml_service.reload()
    fix_reloaded = text_fixer.reload()

    return {
        "status": "SUCCESS",
        "message": "Models and correction dictionaries reloaded successfully",
        "classifier": {
            "reloaded": cls_reloaded,
            "active_version": document_ml_service.model_version,
            "is_trained": document_ml_service.is_trained,
        },
        "text_fixer": {
            "reloaded": fix_reloaded,
            "version": text_fixer.version,
            "rule_count": sum(len(m) for m in text_fixer.corrections.values()),
        },
        "reloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def _collect_handler(payload: CollectRequest, token: str = Depends(require_admin_token)):
    try:
        document_ml_service.collect(payload.rawText, payload.label)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Lỗi endpoint collect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Đăng ký các endpoints yêu cầu token quản trị cho cả 2 routers
for r in (router, alias_router):
    r.add_api_route("/learn-batch", _learn_batch_handler, methods=["POST"], status_code=202)
    r.add_api_route("/learning-status", _learning_status_handler, methods=["GET"])
    r.add_api_route("/reload", _reload_handler, methods=["POST"])
    r.add_api_route("/collect", _collect_handler, methods=["POST"])


# Đăng ký các endpoints công khai không yêu cầu token cho cả 2 routers
for r in (router, alias_router):
    @r.post("/predict")
    async def predict_document(payload: PredictRequest):
        try:
            return document_ml_service.predict(payload.rawText)
        except Exception as e:
            logger.error(f"Lỗi endpoint predict: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/extract")
    async def extract_fields(payload: ExtractRequest):
        try:
            res = await namecard_tagger.tag_fields(
                raw_text=payload.rawText,
                owner_company_id=payload.ownerCompanyId,
            )
            extracted = res.get("extractedFields", {})
            if extracted.get("company"):
                ind = industry_classifier.predict_industry(extracted["company"], text_snippet=payload.rawText)
                extracted["industry"] = ind.get("industry")
                extracted["industryCode"] = ind.get("industryCode")
                extracted["industryConfidence"] = ind.get("confidence")

            return {
                "status": res.get("status", "ACCEPTED"),
                "extractedFields": extracted,
                "completenessScore": res.get("completenessScore", 1.0),
                "llmInvoked": res.get("llmInvoked", False),
                "extractionMethod": res.get("extractionMethod", "REGEX_HEURISTICS"),
            }
        except Exception as e:
            logger.error(f"Lỗi endpoint extract: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/predict-industry")
    async def predict_company_industry(payload: IndustryPredictPayload):
        try:
            return industry_classifier.predict_industry(payload.companyName, payload.textSnippet)
        except Exception as e:
            logger.error(f"Lỗi endpoint predict-industry: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/check-duplicate")
    async def check_card_duplicate(payload: DedupCheckPayload):
        try:
            return dedup_service.check_duplicate(
                image_hash=payload.imageHash,
                phones=[payload.phone] if payload.phone else None,
                emails=[payload.email] if payload.email else None,
                company=payload.company,
                name=payload.name,
            )
        except Exception as e:
            logger.error(f"Lỗi endpoint check-duplicate: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/cluster-unknowns")
    async def cluster_unknowns(payload: ClusterRequest):
        try:
            return document_ml_service.cluster_unknowns(payload.n_clusters)
        except Exception as e:
            logger.error(f"Lỗi endpoint cluster-unknowns: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/analyze-anomaly")
    async def analyze_anomaly(payload: AnomalyRequest):
        try:
            return document_ml_service.analyze_anomaly(payload.current_amount, payload.historical_amounts)
        except Exception as e:
            logger.error(f"Lỗi endpoint analyze-anomaly: {e}")
            raise HTTPException(status_code=500, detail=str(e))