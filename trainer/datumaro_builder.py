"""Build a CSRNet crowd-counting dataset from a CVAT Datumaro export."""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log_default(message: str) -> None:
    print(message)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = [s.lower() for s in archive_path.suffixes]
    lower_name = archive_path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
        return
    if lower_name.endswith(".tar") or lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_image_index(images_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for image_path in iter_image_files(images_root):
        rel = image_path.relative_to(images_root).as_posix()
        stem = image_path.stem
        name = image_path.name
        for key in {
            rel,
            rel.lower(),
            name,
            name.lower(),
            stem,
            stem.lower(),
            f"default/{name}",
            f"default/{stem}",
        }:
            index.setdefault(key, image_path)
    return index


def load_datumaro_payload(annotation_zip: Path) -> Dict:
    with zipfile.ZipFile(annotation_zip, "r") as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json")]
        preferred = sorted(json_names, key=lambda name: ("default.json" not in name.lower(), len(name)))
        for name in preferred:
            try:
                payload = json.loads(zf.read(name).decode("utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                return payload
    raise RuntimeError(f"Could not find a Datumaro items JSON inside {annotation_zip}")


def derive_split_group(source_key: str, stem: str) -> str:
    normalized = str(source_key or stem).replace("\\", "/")
    parent = Path(normalized).parent.as_posix()
    stem_value = Path(normalized).stem or stem

    if stem_value.startswith("vlcsnap-") and "m" in stem_value:
        marker = stem_value.find("m")
        if marker != -1:
            return f"{parent}|{stem_value[:marker + 1]}"

    if stem_value.startswith("frame_"):
        return f"{parent}|frame_sequence"

    digit_run = len(stem_value) - len(stem_value.rstrip("0123456789"))
    if digit_run >= 4 and len(stem_value) > digit_run and stem_value[-digit_run - 1] in {"_", "-"}:
        return f"{parent}|{stem_value[:-digit_run - 1]}"

    return f"{parent}|{stem_value}"


def resolve_item_image(item: Dict, image_index: Dict[str, Path]) -> Tuple[Path | None, str]:
    image_path = ""
    image_meta = item.get("image")
    media_meta = item.get("media")
    if isinstance(image_meta, dict):
        image_path = str(image_meta.get("path") or image_meta.get("name") or "")
    if not image_path and isinstance(media_meta, dict):
        image_path = str(media_meta.get("path") or media_meta.get("name") or "")
    if not image_path:
        image_path = str(item.get("path") or item.get("id") or "")

    lookup_keys = [
        image_path,
        image_path.lower(),
        Path(image_path).name,
        Path(image_path).name.lower(),
        Path(image_path).stem,
        Path(image_path).stem.lower(),
    ]
    for key in lookup_keys:
        if key in image_index:
            return image_index[key], image_path
    return None, image_path


def extract_points(item: Dict) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for ann in item.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        ann_type = str(ann.get("type") or "").lower()
        if ann_type != "points":
            continue
        coords = ann.get("points") or []
        if not isinstance(coords, list):
            continue
        for idx in range(0, len(coords), 2):
            if idx + 1 >= len(coords):
                break
            try:
                x = float(coords[idx])
                y = float(coords[idx + 1])
            except (TypeError, ValueError):
                continue
            points.append({"x": x, "y": y})
    return points


def build_dataset(
    annotation_zip: Path,
    images_root: Path,
    out_root: Path,
    log_fn=log_default,
) -> Dict[str, int]:
    if out_root.exists():
        shutil.rmtree(out_root)
    images_dir = out_root / "images"
    annotations_dir = out_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(images_root)
    payload = load_datumaro_payload(annotation_zip)
    items = payload.get("items", [])

    copied = 0
    skipped_missing_images = 0
    total_points = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        src_image, source_key = resolve_item_image(item, image_index)
        if src_image is None:
            skipped_missing_images += 1
            continue

        stem = Path(source_key or src_image.name).stem or src_image.stem
        dst_image = images_dir / src_image.name
        suffix_idx = 1
        while dst_image.exists() and dst_image.resolve() != src_image.resolve():
            dst_image = images_dir / f"{src_image.stem}_{suffix_idx}{src_image.suffix}"
            suffix_idx += 1
        final_stem = dst_image.stem

        shutil.copy2(src_image, dst_image)

        points = extract_points(item)
        total_points += len(points)
        ann_payload = {
            "points": points,
            "count": len(points),
            "source_image_path": source_key or src_image.name,
            "split_group": derive_split_group(source_key or src_image.name, final_stem),
        }
        (annotations_dir / f"{final_stem}.json").write_text(
            json.dumps(ann_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        copied += 1

    if copied == 0:
        raise RuntimeError("No matched image/annotation pairs were built. Check raw images archive structure.")

    stats = {
        "samples": copied,
        "skipped_missing_images": skipped_missing_images,
        "total_points": total_points,
    }
    log_fn(
        f"Built dataset at {out_root} with {copied} samples; "
        f"skipped_missing_images={skipped_missing_images}; total_points={total_points}"
    )
    return stats

