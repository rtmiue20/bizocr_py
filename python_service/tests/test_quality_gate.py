from python_service.preprocessing.quality_gate import QualityGateConfig, evaluate


def _ok_meta(**overrides):
    meta = {
        "contour_area_ratio": 0.62,
        "width": 900,
        "height": 640,
        "aspect_ratio_after_crop": 1.4062,
        "laplacian_variance": 180.0,
        "is_cropped": True,
    }
    meta.update(overrides)
    return meta


def test_shadow_mode_fails_soft_and_keeps_reasons():
    cfg = QualityGateConfig(mode="shadow", min_contour_area_ratio=0.5)
    result = evaluate(_ok_meta(contour_area_ratio=0.1), cfg)
    assert result.passed is False
    assert "CONTOUR_AREA_TOO_SMALL" in result.reasons
    assert result.mode == "shadow"


def test_enforce_mode_same_rules_different_mode_flag():
    cfg = QualityGateConfig(mode="enforce")
    result = evaluate(_ok_meta(width=50, height=50), cfg)
    assert result.passed is False
    assert "RESOLUTION_TOO_LOW" in result.reasons
    assert result.mode == "enforce"


def test_clean_document_passes():
    result = evaluate(_ok_meta(), QualityGateConfig(mode="shadow"))
    assert result.passed is True
    assert result.reasons == []


def test_aspect_and_blur_and_require_crop():
    cfg = QualityGateConfig(
        mode="shadow",
        min_laplacian_variance=100.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        require_crop=True,
    )
    result = evaluate(
        _ok_meta(
            aspect_ratio_after_crop=4.0,
            laplacian_variance=10.0,
            is_cropped=False,
        ),
        cfg,
    )
    assert set(result.reasons) >= {
        "ASPECT_RATIO_OUT_OF_RANGE",
        "IMAGE_BLURRY",
        "CROP_NOT_APPLIED",
    }
