from __future__ import annotations

import mimetypes
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.job_manager import JobManager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_ROOT = Path(os.environ.get("APP_DATA_DIR", "/app/data/jobs"))
RAW_IMAGES_DIR = Path(os.environ.get("APP_RAW_IMAGES_DIR", "/raw_images"))
WEIGHTS_DIR = Path(os.environ.get("APP_WEIGHTS_DIR", "/weights"))

app = FastAPI(title="CSRNet Finetune Docker", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
job_manager = JobManager(DATA_ROOT)


def to_number(raw: str | None, default, cast):
    if raw in (None, ""):
        return default
    return cast(raw)


def iter_image_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield path


def mounted_resources_payload(limit: int = 12):
    raw_count = 0
    previews = []
    if RAW_IMAGES_DIR.exists() and RAW_IMAGES_DIR.is_dir():
        for image_path in iter_image_files(RAW_IMAGES_DIR):
            raw_count += 1
            if len(previews) < limit:
                rel = image_path.relative_to(RAW_IMAGES_DIR).as_posix()
                previews.append(
                    {
                        "name": image_path.name,
                        "relative_path": rel,
                        "preview_url": f"/api/mounted/raw-preview?rel={rel}",
                    }
                )

    weights = []
    if WEIGHTS_DIR.exists() and WEIGHTS_DIR.is_dir():
        for pattern in ("*.pth", "*.pth.tar", "*.pt"):
            for weight_path in sorted(WEIGHTS_DIR.glob(pattern)):
                weights.append({"name": weight_path.name, "path": str(weight_path)})

    return {
        "raw_images_dir": str(RAW_IMAGES_DIR),
        "raw_images_ready": RAW_IMAGES_DIR.exists() and RAW_IMAGES_DIR.is_dir(),
        "raw_images_count": raw_count,
        "raw_images_previews": previews,
        "weights_dir": str(WEIGHTS_DIR),
        "weights_ready": WEIGHTS_DIR.exists() and WEIGHTS_DIR.is_dir(),
        "weights": weights,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "jobs": job_manager.list_jobs(),
            "mounted": mounted_resources_payload(),
        },
    )


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": job_manager.list_jobs()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["log_tail"] = job_manager.read_log(job_id)
    return job


@app.get("/api/mounted")
def api_mounted():
    return mounted_resources_payload()


def gpu_status_payload():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "nvidia-smi failed", "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total_mb": parts[1],
                "memory_used_mb": parts[2],
                "utilization_gpu_percent": parts[3],
                "temperature_c": parts[4],
            }
        )
    return {"available": bool(gpus), "error": "", "gpus": gpus}


@app.get("/api/system")
def api_system():
    return {"gpu": gpu_status_payload(), "mounted": mounted_resources_payload(limit=8)}


@app.get("/api/mounted/raw-preview")
def api_raw_preview(rel: str = Query(...)):
    rel_path = Path(rel)
    target = (RAW_IMAGES_DIR / rel_path).resolve()
    base = RAW_IMAGES_DIR.resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="Invalid preview path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Preview image not found")
    mime = mimetypes.guess_type(str(target))[0] or "image/jpeg"
    return FileResponse(str(target), media_type=mime, filename=target.name)


@app.get("/api/jobs/{job_id}/download")
def api_job_download(job_id: str, kind: str = "best"):
    job = job_manager.get_job(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")

    result = job.get("result") or {}
    if kind == "production":
        path = result.get("production_checkpoint")
    else:
        path = result.get("best_checkpoint")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=Path(path).name)


@app.post("/api/jobs")
async def create_job(
    name: str = Form(""),
    annotations_zip: UploadFile = File(...),
    init_weights_name: str = Form(""),
    epochs: str = Form("40"),
    batch_size: str = Form("2"),
    train_h: str = Form("544"),
    train_w: str = Form("960"),
    lr: str = Form("3e-6"),
    freeze_frontend_epochs: str = Form("0"),
    perspective_top_scale: str = Form("0.9"),
    perspective_bottom_scale: str = Form("1.8"),
    perspective_gamma: str = Form("1.4"),
    loss_top_scale: str = Form("1.0"),
    loss_bottom_scale: str = Form("1.3"),
    loss_gamma: str = Form("1.2"),
    num_workers: str = Form("2"),
):
    if not annotations_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="annotations_zip must be a .zip file")
    if not RAW_IMAGES_DIR.exists() or not RAW_IMAGES_DIR.is_dir():
        raise HTTPException(status_code=400, detail=f"Mounted raw images dir is unavailable: {RAW_IMAGES_DIR}")
    init_weights_name = (init_weights_name or "").strip()
    init_weights_path = ""
    if init_weights_name:
        candidate = (WEIGHTS_DIR / Path(init_weights_name).name).resolve()
        weights_base = WEIGHTS_DIR.resolve()
        if weights_base not in candidate.parents and candidate != weights_base:
            raise HTTPException(status_code=400, detail="Invalid init weight selection")
        if not candidate.exists():
            raise HTTPException(status_code=400, detail=f"Selected init weight not found: {candidate.name}")
        init_weights_path = str(candidate)

    params = {
        "epochs": to_number(epochs, 40, int),
        "batch_size": to_number(batch_size, 2, int),
        "train_h": to_number(train_h, 544, int),
        "train_w": to_number(train_w, 960, int),
        "lr": to_number(lr, 3e-6, float),
        "freeze_frontend_epochs": to_number(freeze_frontend_epochs, 0, int),
        "perspective_top_scale": to_number(perspective_top_scale, 0.9, float),
        "perspective_bottom_scale": to_number(perspective_bottom_scale, 1.8, float),
        "perspective_gamma": to_number(perspective_gamma, 1.4, float),
        "loss_top_scale": to_number(loss_top_scale, 1.0, float),
        "loss_bottom_scale": to_number(loss_bottom_scale, 1.3, float),
        "loss_gamma": to_number(loss_gamma, 1.2, float),
        "num_workers": to_number(num_workers, 2, int),
    }

    job = job_manager.create_job(
        name=name.strip(),
        annotation_zip=annotations_zip,
        raw_images_dir=str(RAW_IMAGES_DIR),
        init_weights=None,
        init_weights_path=init_weights_path,
        params=params,
    )
    return JSONResponse({"success": True, "job": job})
