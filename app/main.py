from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.job_manager import JobManager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_ROOT = Path(os.environ.get("APP_DATA_DIR", "/app/data/jobs"))

app = FastAPI(title="CSRNet Finetune Docker", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
job_manager = JobManager(DATA_ROOT)


def to_number(raw: str | None, default, cast):
    if raw in (None, ""):
        return default
    return cast(raw)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "jobs": job_manager.list_jobs(),
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
    raw_images_dir: str = Form(...),
    init_weights: UploadFile | None = File(None),
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
    raw_images_dir = (raw_images_dir or "").strip()
    if not raw_images_dir:
        raise HTTPException(status_code=400, detail="raw_images_dir is required")

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
        raw_images_dir=raw_images_dir,
        init_weights=init_weights,
        params=params,
    )
    return JSONResponse({"success": True, "job": job})
