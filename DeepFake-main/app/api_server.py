from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, cast

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from src.image_detector import ImageDetector
from src.fusion_engine import FusionEngine
from src.text_detector import TextDetector
from src.video_detector import VideoDetector


app = FastAPI(title="Deepfake Detection API", version="1.0.0")
logger = logging.getLogger("deepfake_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class HealthResponse(BaseModel):
    status: str


class SystemStatusResponse(BaseModel):
    status: str
    model_states: dict[str, str]
    request_limits: dict[str, int]


def _get_cors_origins() -> list[str]:
    raw = os.getenv("DEEPFAKE_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"]

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict[str, Any] = {}
_rate_limit_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_video_jobs: dict[str, dict[str, Any]] = {}


def _get_rate_limit_per_minute() -> int:
    raw = os.getenv("DEEPFAKE_RATE_LIMIT_PER_MINUTE", "60")
    try:
        value = int(raw)
        return value if value > 0 else 60
    except ValueError:
        return 60


def _get_max_upload_mb() -> int:
    raw = os.getenv("DEEPFAKE_MAX_UPLOAD_MB", "50")
    try:
        value = int(raw)
        return value if value > 0 else 50
    except ValueError:
        return 50


def _get_api_key() -> str:
    return os.getenv("DEEPFAKE_API_KEY", "").strip()


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured_key = _get_api_key()
    if not configured_key:
        return
    if x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


def _check_rate_limit(request: Request) -> None:
    limit = _get_rate_limit_per_minute()
    now = time.time()
    client_host = request.client.host if request.client else "unknown"
    bucket = _rate_limit_buckets[client_host]
    while bucket and (now - bucket[0]) > 60.0:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def _safe_log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    _safe_log(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=elapsed_ms,
    )
    return response


def _latest_file(pattern: str) -> Path | None:
    files = sorted((PROJECT_ROOT / "outputs" / "metrics").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _extract_confusion(payload: dict[str, Any], metrics_key: str) -> dict[str, int] | None:
    metrics = cast(dict[str, Any] | None, payload.get(metrics_key))
    if not isinstance(metrics, dict):
        return None

    try:
        tp = int(metrics.get("tp_fake", 0))
        tn = int(metrics.get("tn_real", 0))
        fp = int(metrics.get("fp_real_as_fake", 0))
        fn = int(metrics.get("fn_fake_as_real", 0))
    except (TypeError, ValueError):
        return None

    return {
        "tn_real_as_real": tn,
        "fp_real_as_fake": fp,
        "fn_fake_as_real": fn,
        "tp_fake_as_fake": tp,
    }


def _load_confusion_from_summary(file_path: Path | None, metrics_key: str) -> dict[str, Any] | None:
    if file_path is None or not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    matrix = _extract_confusion(payload, metrics_key)
    if matrix is None:
        return None

    return {
        "summary_file": str(file_path.relative_to(PROJECT_ROOT)),
        "matrix": matrix,
    }


def _get_threshold(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    except ValueError:
        pass
    return default


def _resolve_label(score: float, threshold: float, margin: float) -> str:
    return "FAKE" if score > (threshold + margin) else "REAL"


def _validate_upload_size(payload: bytes, kind: str) -> None:
    limit_mb = _get_max_upload_mb()
    if len(payload) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"{kind} payload exceeds {limit_mb}MB limit")


def _validate_content_type(upload: UploadFile, allowed_prefix: str, kind: str) -> None:
    content_type = (upload.content_type or "").lower()
    if not content_type.startswith(allowed_prefix):
        raise HTTPException(status_code=415, detail=f"Unsupported {kind} content-type: {content_type or 'unknown'}")


def _load_video_defaults() -> tuple[float, float]:
    threshold_default = 0.3055
    margin_default = 0.2
    meta_path = PROJECT_ROOT / "models" / "video_ensemble_meta.json"
    if not meta_path.exists():
        return threshold_default, margin_default

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return threshold_default, margin_default

    if not isinstance(payload, dict):
        return threshold_default, margin_default

    payload_dict = cast(dict[str, Any], payload)
    thresholds_any = payload_dict.get("thresholds", {})
    if not isinstance(thresholds_any, dict):
        return threshold_default, margin_default
    thresholds = cast(dict[str, Any], thresholds_any)

    threshold = thresholds.get("recommended_env_video_threshold", threshold_default)
    margin = thresholds.get("recommended_env_video_margin", margin_default)

    try:
        threshold_value = float(threshold)
        margin_value = float(margin)
    except (TypeError, ValueError):
        return threshold_default, margin_default

    if not (0.0 <= threshold_value <= 1.0):
        threshold_value = threshold_default
    if not (0.0 <= margin_value <= 1.0):
        margin_value = margin_default

    return threshold_value, margin_value


def _get_model(key: str):
    if key not in _models:
        if key == "text":
            _models[key] = TextDetector()
        elif key == "image":
            _models[key] = ImageDetector()
        elif key == "video":
            _models[key] = VideoDetector()
    return _models[key]


def _model_state(key: str) -> str:
    try:
        model = _get_model(key)
        if key == "video" and getattr(model, "model_loaded", True) is False:
            return "degraded"
        return "ready"
    except Exception:
        return "failed"


def _run_video_job(job_id: str, video_path: Path) -> None:
    _video_jobs[job_id]["status"] = "running"
    try:
        video_model = _get_model("video")
        if getattr(video_model, "model_loaded", True) is False:
            load_error = getattr(video_model, "load_error", "video model not loaded")
            raise RuntimeError(str(load_error))

        score, frames = video_model.predict(str(video_path))
        threshold = _get_threshold("DEEPFAKE_THRESHOLD_VIDEO", _load_video_defaults()[0])
        margin = _get_threshold("DEEPFAKE_MARGIN_VIDEO", _load_video_defaults()[1])
        _video_jobs[job_id]["status"] = "completed"
        _video_jobs[job_id]["result"] = {
            "score": float(score),
            "frames": int(frames),
            "threshold": threshold,
            "margin": margin,
            "label": _resolve_label(float(score), threshold, margin),
        }
    except Exception as exc:
        _video_jobs[job_id]["status"] = "failed"
        _video_jobs[job_id]["error"] = str(exc)
    finally:
        video_path.unlink(missing_ok=True)


def _analyze_text(text: str, thresholds: dict[str, float], std_limits: dict[str, float]) -> tuple[float | None, str, str | None, dict[str, float | list[float]] | None]:
    try:
        text_model = _get_model("text")
        consistency = text_model.predict_with_consistency(text)
        raw_score = float(consistency["score"])
        mean_score = float(consistency["mean_score"])
        std_score = float(consistency["std_score"])

        text_threshold = float(thresholds["text"])
        text_std_limit = float(std_limits["text"])
        if raw_score >= text_threshold and (mean_score < text_threshold or std_score > text_std_limit):
            score = mean_score
        else:
            score = raw_score
        return score, "Processed", None, consistency
    except Exception as exc:
        return None, "Failed", str(exc), None


async def _analyze_image(image: UploadFile, thresholds: dict[str, float], std_limits: dict[str, float]) -> tuple[float | None, str, str | None, dict[str, float | list[float]] | None, int | None]:
    temp_image_path: Path | None = None
    try:
        _validate_content_type(image, "image/", "image")
        payload = await image.read()
        _validate_upload_size(payload, "image")
        suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_image_path = Path(temp_file.name)
            temp_file.write(payload)

        image_model = _get_model("image")
        image_fake_class_index = int(getattr(image_model, "fake_class_index", 0))
        consistency = image_model.predict_with_consistency(str(temp_image_path))

        raw_score = float(consistency["score"])
        mean_score = float(consistency["mean_score"])
        std_score = float(consistency["std_score"])

        image_threshold = float(thresholds["image"])
        image_std_limit = float(std_limits["image"])
        if raw_score >= image_threshold and (mean_score < image_threshold or std_score > image_std_limit):
            score = mean_score
        else:
            score = raw_score
        return score, "Processed", None, consistency, image_fake_class_index
    except HTTPException:
        raise
    except Exception as exc:
        return None, "Failed", str(exc), None, None
    finally:
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink(missing_ok=True)


async def _analyze_video(video: UploadFile) -> tuple[float | None, str, str | None, dict[str, float | list[float]] | None, int | None]:
    temp_video_path: Path | None = None
    try:
        _validate_content_type(video, "video/", "video")
        payload = await video.read()
        _validate_upload_size(payload, "video")
        suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_video_path = Path(temp_file.name)
            temp_file.write(payload)

        video_model = _get_model("video")
        if getattr(video_model, "model_loaded", True) is False:
            load_error = getattr(video_model, "load_error", "video model not loaded")
            return None, "Degraded: model unavailable", load_error, None, None

        score, frames = video_model.predict(str(temp_video_path))
        video_score = float(score)
        verification = {
            "score": video_score,
            "mean_score": video_score,
            "std_score": 0.0,
            "scores": [video_score],
        }
        return video_score, "Processed", None, verification, int(frames)
    except HTTPException:
        raise
    except Exception as exc:
        return None, "Failed", str(exc), None, None
    finally:
        if temp_video_path and temp_video_path.exists():
            temp_video_path.unlink(missing_ok=True)


@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def health(_: None = Depends(_require_api_key)) -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics/confusion-matrices")
@app.get("/api/v1/metrics/confusion-matrices")
def confusion_matrices(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    text_summary = _latest_file("text_dataset_eval_summary_*.json")
    image_summary = _latest_file("image_dataset_eval_summary_*.json")
    video_summary = _latest_file("video_dataset_eval_summary_*.json")

    return {
        "text": _load_confusion_from_summary(text_summary, "metrics_clear_set"),
        "image": _load_confusion_from_summary(image_summary, "metrics"),
        "video": _load_confusion_from_summary(video_summary, "metrics"),
    }


@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
def system_status(_: None = Depends(_require_api_key)) -> SystemStatusResponse:
    states = {
        "text": _model_state("text"),
        "image": _model_state("image"),
        "video": _model_state("video"),
    }
    status = "ok" if all(state in {"ready", "degraded"} for state in states.values()) else "degraded"
    return SystemStatusResponse(
        status=status,
        model_states=states,
        request_limits={
            "rate_limit_per_minute": _get_rate_limit_per_minute(),
            "max_upload_mb": _get_max_upload_mb(),
        },
    )


@app.post("/api/v1/jobs/video")
async def submit_video_job(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    _check_rate_limit(request)
    _validate_content_type(video, "video/", "video")
    payload = await video.read()
    _validate_upload_size(payload, "video")

    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_video_path = Path(temp_file.name)
        temp_file.write(payload)

    job_id = str(uuid.uuid4())
    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "result": None,
        "error": None,
    }
    background_tasks.add_task(_run_video_job, job_id, temp_video_path)
    return {
        "job_id": job_id,
        "status": "queued",
        "request_id": getattr(request.state, "request_id", "n/a"),
    }


@app.get("/api/v1/jobs/{job_id}")
def get_video_job(job_id: str, _: None = Depends(_require_api_key)) -> dict[str, Any]:
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/analyze")
@app.post("/api/v1/analyze")
async def analyze(
    request: Request,
    text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
    _: None = Depends(_require_api_key),
) -> dict[str, Any]:
    _check_rate_limit(request)
    video_threshold_default, video_margin_default = _load_video_defaults()

    thresholds = {
        "text": _get_threshold("DEEPFAKE_THRESHOLD_TEXT", 0.954),
        "image": _get_threshold("DEEPFAKE_THRESHOLD_IMAGE", 0.568),
        "video": _get_threshold("DEEPFAKE_THRESHOLD_VIDEO", video_threshold_default),
    }

    margins = {
        "text": _get_threshold("DEEPFAKE_MARGIN_TEXT", 0.0275),
        "image": _get_threshold("DEEPFAKE_MARGIN_IMAGE", 0.0),
        "video": _get_threshold("DEEPFAKE_MARGIN_VIDEO", video_margin_default),
    }

    std_limits = {
        "text": _get_threshold("DEEPFAKE_TEXT_STD_LIMIT", 0.2),
        "image": _get_threshold("DEEPFAKE_IMAGE_STD_LIMIT", 0.12),
        "video": _get_threshold("DEEPFAKE_VIDEO_STD_LIMIT", 0.15),
    }

    selected_count = int(bool(text and text.strip())) + int(image is not None) + int(video is not None)
    if selected_count == 0:
        raise HTTPException(status_code=400, detail="Please provide at least one modality: text, image, or video.")

    scores: dict[str, float | None] = {"text": None, "image": None, "video": None}
    states: dict[str, str] = {"text": "Not selected", "image": "Not selected", "video": "Not selected"}
    errors: dict[str, str | None] = {"text": None, "image": None, "video": None}
    frames_analyzed: int | None = None
    image_fake_class_index: int | None = None
    text_verification: dict[str, float | list[float]] | None = None
    image_verification: dict[str, float | list[float]] | None = None
    video_verification: dict[str, float | list[float]] | None = None
    selected_modalities = [
        modality
        for modality, enabled in {
            "text": bool(text and text.strip()),
            "image": image is not None,
            "video": video is not None,
        }.items()
        if enabled
    ]
    selected_modality = selected_modalities[0] if len(selected_modalities) == 1 else "multimodal"

    if text and text.strip():
        score, state, error, verification = _analyze_text(text, thresholds, std_limits)
        scores["text"] = score
        states["text"] = state
        errors["text"] = error
        text_verification = verification

    if image is not None:
        (
            score,
            state,
            error,
            verification,
            fake_class_index,
        ) = await _analyze_image(image, thresholds, std_limits)
        scores["image"] = score
        states["image"] = state
        errors["image"] = error
        image_verification = verification
        image_fake_class_index = fake_class_index

    if video is not None:
        score, state, error, verification, frames = await _analyze_video(video)
        scores["video"] = score
        states["video"] = state
        errors["video"] = error
        video_verification = verification
        frames_analyzed = frames

    available_scores = {k: v for k, v in scores.items() if isinstance(v, float)}
    if not available_scores:
        raise HTTPException(status_code=503, detail="No modality score available; all selected models are unavailable or failed.")

    if selected_modality == "multimodal":
        fusion = FusionEngine()
        final_score = float(
            fusion.fuse_scores(
                available_scores.get("text", 0.5),
                available_scores.get("image", 0.5),
                available_scores.get("video", 0.5),
            )
        )
        selected_threshold = 0.5
        selected_margin = 0.0
        fusion_weights = [0.3, 0.3, 0.4]
    else:
        selected_score = scores[selected_modality]
        final_score = float(selected_score if selected_score is not None else 0.5)
        selected_threshold = float(thresholds[selected_modality])
        selected_margin = float(margins[selected_modality])
        fusion_weights = None

    degraded_modalities = [modality for modality, state in states.items() if state.startswith("Degraded")]
    failed_modalities = [modality for modality, state in states.items() if state == "Failed"]

    _safe_log(
        "analyze_completed",
        request_id=getattr(request.state, "request_id", "n/a"),
        selected_modalities=selected_modalities,
        final_score=final_score,
        label=_resolve_label(final_score, selected_threshold, selected_margin),
        degraded_modalities=degraded_modalities,
        failed_modalities=failed_modalities,
    )

    response_payload = {
        "scores": scores,
        "states": states,
        "errors": errors,
        "frames": frames_analyzed,
        "selected_modality": selected_modality,
        "selected_modalities": selected_modalities,
        "final_score": final_score,
        "label": _resolve_label(final_score, selected_threshold, selected_margin),
        "threshold": selected_threshold,
        "thresholds": thresholds,
        "margin": selected_margin,
        "margins": margins,
        "std_limits": std_limits,
        "image_fake_class_index": image_fake_class_index,
        "text_verification": text_verification,
        "image_verification": image_verification,
        "video_verification": video_verification,
        "degraded_mode": bool(degraded_modalities),
        "degraded_modalities": degraded_modalities,
        "failed_modalities": failed_modalities,
        "request_id": getattr(request.state, "request_id", "n/a"),
        "fusion": {
            "enabled": selected_modality == "multimodal",
            "weights": fusion_weights,
            "threshold": 0.5 if selected_modality == "multimodal" else None,
        },
    }
    return JSONResponse(content=response_payload)
