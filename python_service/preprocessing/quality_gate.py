"""
Cổng chất lượng ảnh sau crop/deskew (Bước 3).

Mặc định SHADOW: luôn cho pipeline chạy tiếp, chỉ gắn cờ + log/metric.
ENFORCE: từ chối request trước OCR khi không đạt ngưỡng.

Không ghi vào MariaDB — dùng logger riêng `bizocr.quality_gate` (file append)
và bộ đếm in-process để quan sát tỉ lệ rớt.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bizocr.quality_gate")


@dataclass(frozen=True)
class QualityGateConfig:
    mode: str = "shadow"  # shadow | enforce
    min_contour_area_ratio: float = 0.08
    min_width: int = 200
    min_height: int = 200
    min_laplacian_variance: float = 40.0
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 3.50
    require_crop: bool = False


@dataclass
class QualityResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    mode: str = "shadow"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "qualityPassed": self.passed,
            "qualityReasons": list(self.reasons),
            "qualityMode": self.mode.upper(),
        }


class QualityGateMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.evaluated = 0
        self.failed = 0
        self.reason_counts: Counter[str] = Counter()

    def record(self, result: QualityResult) -> None:
        with self._lock:
            self.evaluated += 1
            if not result.passed:
                self.failed += 1
                self.reason_counts.update(result.reasons)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = self.evaluated
            return {
                "evaluated": n,
                "failed": self.failed,
                "fail_rate": (self.failed / n) if n else None,
                "reasons": dict(self.reason_counts),
                "mode": None,
            }


_metrics = QualityGateMetrics()


def metrics_snapshot(config: Optional[QualityGateConfig] = None) -> Dict[str, Any]:
    snap = _metrics.snapshot()
    if config is not None:
        snap["mode"] = config.mode
    return snap


def evaluate(metadata: Dict[str, Any], config: QualityGateConfig) -> QualityResult:
    reasons: List[str] = []
    contour = float(metadata.get("contour_area_ratio") or 0.0)
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    aspect = float(
        metadata.get("aspect_ratio_after_crop")
        if metadata.get("aspect_ratio_after_crop") is not None
        else (metadata.get("aspect_ratio") or 0.0)
    )
    laplacian = float(metadata.get("laplacian_variance") or 0.0)
    cropped = bool(metadata.get("is_cropped"))

    if contour < config.min_contour_area_ratio:
        reasons.append("CONTOUR_AREA_TOO_SMALL")
    if width < config.min_width or height < config.min_height:
        reasons.append("RESOLUTION_TOO_LOW")
    if laplacian < config.min_laplacian_variance:
        reasons.append("IMAGE_BLURRY")
    if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
        reasons.append("ASPECT_RATIO_OUT_OF_RANGE")
    if config.require_crop and not cropped:
        reasons.append("CROP_NOT_APPLIED")

    result = QualityResult(passed=not reasons, reasons=reasons, mode=config.mode)
    _metrics.record(result)

    if reasons:
        logger.warning(
            "quality_gate mode=%s passed=%s reasons=%s contour_area_ratio=%s "
            "aspect_ratio_after_crop=%s size=%sx%s laplacian=%s cropped=%s",
            config.mode,
            result.passed,
            ",".join(reasons),
            contour,
            aspect,
            width,
            height,
            laplacian,
            cropped,
        )
    else:
        logger.info(
            "quality_gate mode=%s passed=true contour_area_ratio=%s aspect_ratio_after_crop=%s",
            config.mode,
            contour,
            aspect,
        )
    return result
