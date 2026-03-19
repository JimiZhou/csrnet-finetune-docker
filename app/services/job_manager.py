from __future__ import annotations

import json
import queue
import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from trainer.datumaro_builder import build_dataset
from trainer.train import TrainConfig, run_training


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobManager:
    def __init__(self, jobs_root: Path):
        self.jobs_root = jobs_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._load_existing_jobs()
        self._worker.start()

    def _load_existing_jobs(self) -> None:
        for state_path in self.jobs_root.glob("*/state.json"):
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                self._jobs[payload["id"]] = payload
            except Exception:
                continue

    def _save_state(self, job: Dict[str, Any]) -> None:
        state_path = self.jobs_root / job["id"] / "state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted((dict(job) for job in self._jobs.values()), key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def read_log(self, job_id: str, max_chars: int = 12000) -> str:
        log_path = self.jobs_root / job_id / "logs" / "train.log"
        if not log_path.exists():
            return ""
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        return text[-max_chars:]

    def create_job(
        self,
        *,
        name: str,
        annotation_zip,
        raw_images_dir: str,
        init_weights,
        init_weights_path: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        job_dir = self.jobs_root / job_id
        uploads_dir = job_dir / "uploads"
        logs_dir = job_dir / "logs"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        ann_path = uploads_dir / annotation_zip.filename
        ann_path.write_bytes(annotation_zip.file.read())

        init_path = None
        init_weights_path = (init_weights_path or "").strip()
        if init_weights_path:
            init_path = Path(init_weights_path)
        if init_weights and getattr(init_weights, "filename", ""):
            init_path = uploads_dir / init_weights.filename
            init_path.write_bytes(init_weights.file.read())

        job = {
            "id": job_id,
            "name": name or job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "annotation_zip": str(ann_path),
            "raw_images_dir": str(raw_images_dir),
            "init_weights": str(init_path) if init_path else None,
            "params": params,
            "dataset_root": str(job_dir / "dataset"),
            "model_path": str(job_dir / "models" / f"{job_id}.pth"),
            "error": "",
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save_state(job)
        self._queue.put(job_id)
        return dict(job)

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job["updated_at"] = now_iso()
            self._save_state(job)

    def _append_log(self, job_id: str, message: str) -> None:
        log_path = self.jobs_root / job_id / "logs" / "train.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception:
                self._append_log(job_id, traceback.format_exc())
                self._update_job(job_id, status="failed", phase="failed", error=traceback.format_exc())
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        job_dir = self.jobs_root / job_id
        dataset_root = Path(job["dataset_root"])
        models_dir = job_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        def log(message: str) -> None:
            self._append_log(job_id, message)

        raw_images_dir = Path(str(job["raw_images_dir"]))
        if not raw_images_dir.exists():
            raise FileNotFoundError(f"Mounted raw images dir not found: {raw_images_dir}")
        if not raw_images_dir.is_dir():
            raise NotADirectoryError(f"Mounted raw images path is not a directory: {raw_images_dir}")
        if job["init_weights"]:
            init_weights_path = Path(str(job["init_weights"]))
            if not init_weights_path.exists():
                raise FileNotFoundError(f"Init weights path not found: {init_weights_path}")

        self._update_job(job_id, status="running", phase="building_dataset", progress=15)
        log("Building dataset from CVAT Datumaro zip...")
        build_stats = build_dataset(Path(job["annotation_zip"]), raw_images_dir, dataset_root, log_fn=log)
        self._update_job(job_id, phase="training", progress=35)

        params = job["params"]
        config = TrainConfig(
            data_root=dataset_root,
            out=Path(job["model_path"]),
            pretrained=Path(job["init_weights"]) if job["init_weights"] else None,
            epochs=int(params.get("epochs", 40)),
            batch_size=int(params.get("batch_size", 2)),
            train_h=int(params.get("train_h", 544)),
            train_w=int(params.get("train_w", 960)),
            lr=float(params.get("lr", 3e-6)),
            freeze_frontend_epochs=int(params.get("freeze_frontend_epochs", 0)),
            perspective_top_scale=float(params.get("perspective_top_scale", 0.9)),
            perspective_bottom_scale=float(params.get("perspective_bottom_scale", 1.8)),
            perspective_gamma=float(params.get("perspective_gamma", 1.4)),
            loss_top_scale=float(params.get("loss_top_scale", 1.0)),
            loss_bottom_scale=float(params.get("loss_bottom_scale", 1.3)),
            loss_gamma=float(params.get("loss_gamma", 1.2)),
            num_workers=int(params.get("num_workers", 2)),
        )

        result = run_training(config, log_fn=log)
        production_path = models_dir / "csrnet_production.pth"
        shutil.copy2(config.out, production_path)

        self._update_job(
            job_id,
            status="completed",
            phase="completed",
            progress=100,
            result={
                **result,
                "dataset": build_stats,
                "production_checkpoint": str(production_path),
            },
        )
