"""Standalone museum CSRNet fine-tuning entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from trainer.csrnet_model import CSRNet

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def discover_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None


def normalize_img(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def gaussian2d(shape: Tuple[int, int], sigma: float, center: Tuple[float, float]) -> np.ndarray:
    h, w = shape
    x0, y0 = center
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def compute_adaptive_sigmas(
    scaled_points: np.ndarray,
    base_sigma: float,
    min_sigma: float,
    max_sigma: float,
    knn: int,
) -> np.ndarray:
    if len(scaled_points) <= 1:
        return np.full((len(scaled_points),), float(base_sigma), dtype=np.float32)

    deltas = scaled_points[:, None, :] - scaled_points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    np.fill_diagonal(distances, np.inf)

    k = max(1, min(int(knn), len(scaled_points) - 1))
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    sigmas = np.mean(nearest, axis=1) * 0.3
    sigmas = np.where(np.isfinite(sigmas), sigmas, float(base_sigma))
    return np.clip(sigmas, float(min_sigma), float(max_sigma)).astype(np.float32)


def compute_perspective_scales(
    y_coords: np.ndarray,
    out_h: int,
    top_scale: float,
    bottom_scale: float,
    gamma: float,
) -> np.ndarray:
    if len(y_coords) == 0:
        return np.zeros((0,), dtype=np.float32)

    if out_h <= 1:
        y_norm = np.ones((len(y_coords),), dtype=np.float32)
    else:
        y_norm = np.clip(y_coords.astype(np.float32) / float(out_h - 1), 0.0, 1.0)

    safe_gamma = max(1e-3, float(gamma))
    curve = np.power(y_norm, safe_gamma, dtype=np.float32)
    scales = float(top_scale) + (float(bottom_scale) - float(top_scale)) * curve
    return scales.astype(np.float32, copy=False)


def build_row_weight_curve(height: int, top_scale: float, bottom_scale: float, gamma: float) -> torch.Tensor:
    rows = np.arange(height, dtype=np.float32)
    scales = compute_perspective_scales(rows, out_h=height, top_scale=top_scale, bottom_scale=bottom_scale, gamma=gamma)
    return torch.from_numpy(scales.reshape(1, 1, height, 1)).float()


def points_to_density(
    points: List[Dict],
    out_h: int,
    out_w: int,
    src_h: int,
    src_w: int,
    sigma: float,
    sigma_mode: str,
    sigma_min: float,
    sigma_max: float,
    sigma_knn: int,
    perspective_top_scale: float,
    perspective_bottom_scale: float,
    perspective_gamma: float,
) -> np.ndarray:
    if not points:
        return np.zeros((out_h, out_w), dtype=np.float32)

    sx = out_w / max(src_w, 1)
    sy = out_h / max(src_h, 1)
    scaled_points = []
    for point in points:
        x = max(0.0, min(out_w - 1.0, float(point['x']) * sx))
        y = max(0.0, min(out_h - 1.0, float(point['y']) * sy))
        scaled_points.append((x, y))
    coords = np.asarray(scaled_points, dtype=np.float32)

    if sigma_mode == 'adaptive':
        sigmas = compute_adaptive_sigmas(coords, base_sigma=sigma, min_sigma=sigma_min, max_sigma=sigma_max, knn=sigma_knn)
    else:
        sigmas = np.full((len(coords),), float(sigma), dtype=np.float32)

    perspective_scales = compute_perspective_scales(
        coords[:, 1],
        out_h=out_h,
        top_scale=perspective_top_scale,
        bottom_scale=perspective_bottom_scale,
        gamma=perspective_gamma,
    )
    sigma_floor_scale = min(1.0, float(perspective_top_scale), float(perspective_bottom_scale))
    sigma_ceil_scale = max(1.0, float(perspective_top_scale), float(perspective_bottom_scale))
    sigmas = np.clip(
        sigmas * perspective_scales,
        float(sigma_min) * sigma_floor_scale,
        float(sigma_max) * sigma_ceil_scale,
    ).astype(np.float32, copy=False)

    density = np.zeros((out_h, out_w), dtype=np.float32)
    for (x, y), point_sigma in zip(coords, sigmas):
        density += gaussian2d((out_h, out_w), sigma=float(point_sigma), center=(float(x), float(y))).astype(np.float32)

    total_mass = float(density.sum())
    if total_mass > 0:
        density *= (len(points) / total_mass)
    return density


def apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), float(gamma))
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def grouped_split(stems_with_groups: Sequence[Tuple[str, str]], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    groups: Dict[str, List[str]] = {}
    for stem, group in stems_with_groups:
        groups.setdefault(group, []).append(stem)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    total = len(stems_with_groups)
    target_val = max(1, int(round(total * val_ratio)))
    val: List[str] = []
    train: List[str] = []

    for group_key in group_keys:
        bucket = sorted(groups[group_key])
        if len(val) < target_val:
            val.extend(bucket)
        else:
            train.extend(bucket)

    if not train and val:
        train.append(val.pop())

    return sorted(train), sorted(val)


@dataclass
class TrainConfig:
    data_root: Path
    out: Path
    pretrained: Path | None = None
    epochs: int = 40
    batch_size: int = 2
    split_mode: str = "grouped"
    val_ratio: float = 0.2
    seed: int = 42
    train_h: int = 544
    train_w: int = 960
    sigma: float = 1.8
    sigma_mode: str = "adaptive"
    sigma_min: float = 1.2
    sigma_max: float = 3.0
    sigma_knn: int = 3
    perspective_top_scale: float = 0.9
    perspective_bottom_scale: float = 1.8
    perspective_gamma: float = 1.4
    loss_top_scale: float = 1.0
    loss_bottom_scale: float = 1.3
    loss_gamma: float = 1.2
    lr: float = 3e-6
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    lr_factor: float = 0.5
    lr_patience: int = 3
    early_stop_patience: int = 8
    density_loss: str = "smoothl1"
    smoothl1_beta: float = 0.5
    count_loss_weight: float = 0.1
    augment: bool = True
    noise_std: float = 4.0
    freeze_frontend_epochs: int = 0
    num_workers: int = 2


@dataclass
class TrainSample:
    img_path: Path
    ann_path: Path


class OverheadCrowdDataset(Dataset):
    def __init__(self, root: Path, split: str, config: TrainConfig):
        self.root = root
        self.split = split
        self.cfg = config
        self.train_h, self.train_w = config.train_h, config.train_w

        split_file = root / f'split_{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Missing {split_file}.')

        ids = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.samples: List[TrainSample] = []
        for stem in ids:
            img_path = discover_image_path(root / 'images', stem)
            ann_path = root / 'annotations' / f'{stem}.json'
            if img_path is None:
                raise FileNotFoundError(f'Missing image for stem: {stem}')
            if not ann_path.exists():
                raise FileNotFoundError(f'Missing annotation: {ann_path}')
            self.samples.append(TrainSample(img_path=img_path, ann_path=ann_path))

    def __len__(self):
        return len(self.samples)

    def _augment_image_and_points(self, img: Image.Image, points: List[Dict[str, float]]) -> Tuple[Image.Image, List[Dict[str, float]]]:
        width, height = img.size
        aug_points = [{'x': float(p['x']), 'y': float(p['y'])} for p in points]

        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for point in aug_points:
                point['x'] = max(0.0, min(width - 1.0, (width - 1.0) - point['x']))

        if random.random() < 0.15:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            for point in aug_points:
                point['y'] = max(0.0, min(height - 1.0, (height - 1.0) - point['y']))

        if random.random() < 0.9:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.72, 1.28))
        if random.random() < 0.8:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.80, 1.20))
        if random.random() < 0.45:
            img = apply_gamma(img, random.uniform(0.75, 1.35))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        return img, aug_points

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.img_path).convert('RGB')
        src_w, src_h = img.size
        ann = load_json(sample.ann_path)
        points = ann.get('points')
        count_only = ann.get('count')

        if points is None and count_only is None:
            raise ValueError(f'Annotation must contain points or count: {sample.ann_path}')

        point_list = [] if points is None else [{'x': float(p['x']), 'y': float(p['y'])} for p in points]

        if self.cfg.augment and point_list:
            img, point_list = self._augment_image_and_points(img, point_list)
        elif self.cfg.augment:
            img, _ = self._augment_image_and_points(img, [])

        img_rs = img.resize((self.train_w, self.train_h), resample=Image.BILINEAR)
        img_np = np.array(img_rs, dtype=np.uint8)
        if self.cfg.augment and self.cfg.noise_std > 0 and random.random() < 0.35:
            noise = np.random.normal(loc=0.0, scale=self.cfg.noise_std, size=img_np.shape)
            img_np = np.clip(img_np.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        x = normalize_img(img_np)
        out_h = max(1, self.train_h // 8)
        out_w = max(1, self.train_w // 8)

        if point_list:
            density = points_to_density(
                point_list,
                out_h=out_h,
                out_w=out_w,
                src_h=src_h,
                src_w=src_w,
                sigma=self.cfg.sigma,
                sigma_mode=self.cfg.sigma_mode,
                sigma_min=self.cfg.sigma_min,
                sigma_max=self.cfg.sigma_max,
                sigma_knn=self.cfg.sigma_knn,
                perspective_top_scale=self.cfg.perspective_top_scale,
                perspective_bottom_scale=self.cfg.perspective_bottom_scale,
                perspective_gamma=self.cfg.perspective_gamma,
            )
            y = torch.from_numpy(density).unsqueeze(0).float()
            gt_count = float(len(point_list))
        else:
            gt_count = float(count_only) if count_only is not None else 0.0
            if gt_count <= 0:
                density = np.zeros((out_h, out_w), dtype=np.float32)
            else:
                density = np.ones((out_h, out_w), dtype=np.float32)
                density *= gt_count / float(density.sum())
            y = torch.from_numpy(density).unsqueeze(0).float()

        return x, y, torch.tensor(gt_count, dtype=torch.float32), sample.img_path.name


def make_split(root: Path, val_ratio: float = 0.2, seed: int = 42, split_mode: str = 'grouped'):
    annotation_paths = sorted((root / 'annotations').glob('*.json'))
    stems_with_groups = []
    for ann_path in annotation_paths:
        stem = ann_path.stem
        if discover_image_path(root / 'images', stem) is None:
            continue
        ann = load_json(ann_path)
        group = str(ann.get('split_group') or stem)
        stems_with_groups.append((stem, group))

    if not stems_with_groups:
        raise RuntimeError(f'No matched image/annotation samples found under {root}')

    if split_mode == 'random':
        stems = [stem for stem, _group in stems_with_groups]
        random.Random(seed).shuffle(stems)
        n_val = max(1, int(len(stems) * val_ratio))
        val = sorted(stems[:n_val])
        train = sorted(stems[n_val:])
    else:
        train, val = grouped_split(stems_with_groups, val_ratio=val_ratio, seed=seed)

    (root / 'split_train.txt').write_text('\n'.join(train) + '\n', encoding='utf-8')
    (root / 'split_val.txt').write_text('\n'.join(val) + '\n', encoding='utf-8')
    return {'train': len(train), 'val': len(val)}


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    mae = 0.0
    mse = 0.0
    n = 0
    with torch.no_grad():
        for x, _y, gt_count, _name in loader:
            x = x.to(device)
            gt_count = gt_count.to(device)
            pred = torch.relu(model(x))
            pred_count = pred.sum(dim=(1, 2, 3))
            err = (pred_count - gt_count).abs()
            mae += float(err.sum().item())
            mse += float(((pred_count - gt_count) ** 2).sum().item())
            n += int(x.shape[0])
    return {'mae': mae / max(n, 1), 'rmse': math.sqrt(mse / max(n, 1))}


def set_frontend_trainable(model: CSRNet, trainable: bool):
    for param in model.frontend.parameters():
        param.requires_grad = bool(trainable)


def build_density_loss(name: str, smooth_l1_beta: float) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == 'mse':
        return nn.MSELoss(reduction='none')
    if normalized == 'l1':
        return nn.L1Loss(reduction='none')
    if normalized == 'smoothl1':
        return nn.SmoothL1Loss(beta=float(smooth_l1_beta), reduction='none')
    raise ValueError(f'Unsupported density loss: {name}')


def reduce_density_loss(loss_map: torch.Tensor, row_weights: torch.Tensor | None) -> torch.Tensor:
    if row_weights is None:
        return loss_map.mean()
    normalized_weights = row_weights / row_weights.mean().clamp_min(1e-6)
    return (loss_map * normalized_weights).mean()


def load_checkpoint_weights(model: CSRNet, pretrained: Path | None, log_fn) -> None:
    if pretrained and pretrained.exists():
        checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state, strict=True)
        log_fn(f'Loaded pretrained: {pretrained}')
    else:
        log_fn('No pretrained checkpoint found; training from random/ImageNet-init weights.')


def run_training(config: TrainConfig, log_fn=print) -> Dict[str, object]:
    if config.train_h % 8 != 0 or config.train_w % 8 != 0:
        raise ValueError('train_h and train_w must be divisible by 8.')

    seed_everything(config.seed)
    split_stats = make_split(config.data_root, val_ratio=config.val_ratio, seed=config.seed, split_mode=config.split_mode)
    log_fn(f"Split files written. train={split_stats['train']} val={split_stats['val']} mode={config.split_mode}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CSRNet(load_weights=config.pretrained is None)
    load_checkpoint_weights(model, config.pretrained, log_fn=log_fn)
    model.to(device)

    train_ds = OverheadCrowdDataset(config.data_root, 'train', config)
    val_ds = OverheadCrowdDataset(config.data_root, 'val', TrainConfig(**{**asdict(config), 'augment': False, 'noise_std': 0.0}))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)

    density_loss = build_density_loss(config.density_loss, config.smoothl1_beta)
    row_weights = build_row_weight_curve(
        config.train_h // 8,
        top_scale=config.loss_top_scale,
        bottom_scale=config.loss_bottom_scale,
        gamma=config.loss_gamma,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.min_lr,
    )

    config.out.parent.mkdir(parents=True, exist_ok=True)
    best_mae = float('inf')
    stale_epochs = 0
    best_metrics: Dict[str, float] = {}

    log_fn(
        f"device={device} train={len(train_ds)} val={len(val_ds)} batch_size={config.batch_size} "
        f"perspective_sigma=({config.perspective_top_scale:.2f}->{config.perspective_bottom_scale:.2f}) "
        f"perspective_loss=({config.loss_top_scale:.2f}->{config.loss_bottom_scale:.2f})"
    )

    for epoch in range(1, config.epochs + 1):
        frontend_frozen = epoch <= config.freeze_frontend_epochs
        set_frontend_trainable(model, trainable=not frontend_frozen)
        model.train()
        running = 0.0

        for x, y, gt_count, _name in train_loader:
            x = x.to(device)
            y = y.to(device)
            gt_count = gt_count.to(device)
            pred = torch.relu(model(x))
            pred_count = pred.sum(dim=(1, 2, 3))
            loss_density_map = density_loss(pred, y)
            loss_density = reduce_density_loss(loss_density_map, row_weights)
            loss_count = torch.mean((pred_count - gt_count) ** 2)
            loss = loss_density + float(config.count_loss_weight) * loss_count

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())

        metrics = evaluate(model, val_loader, device=device)
        scheduler.step(metrics['mae'])
        current_lr = opt.param_groups[0]['lr']
        avg_loss = running / max(len(train_loader), 1)
        log_fn(
            f"epoch={epoch:03d}/{config.epochs:03d} loss={avg_loss:.6f} val_mae={metrics['mae']:.3f} "
            f"val_rmse={metrics['rmse']:.3f} lr={current_lr:.6g} frontend_frozen={frontend_frozen}"
        )

        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val': metrics,
                    'args': asdict(config),
                },
                config.out,
            )
            log_fn(f"saved_best={config.out} val_mae={metrics['mae']:.3f}")
        else:
            stale_epochs += 1
            if config.early_stop_patience > 0 and stale_epochs >= config.early_stop_patience:
                log_fn(f"Early stop triggered at epoch {epoch}.")
                break

    return {
        'best_checkpoint': str(config.out),
        'best_mae': float(best_mae),
        'best_metrics': best_metrics,
        'summary': {
            'best_checkpoint': str(config.out),
            'best_mae': float(best_mae),
            'best_rmse': float(best_metrics.get('rmse', float('nan'))) if best_metrics else float('nan'),
            'epochs': int(config.epochs),
            'batch_size': int(config.batch_size),
            'train_h': int(config.train_h),
            'train_w': int(config.train_w),
            'lr': float(config.lr),
            'perspective_top_scale': float(config.perspective_top_scale),
            'perspective_bottom_scale': float(config.perspective_bottom_scale),
            'loss_top_scale': float(config.loss_top_scale),
            'loss_bottom_scale': float(config.loss_bottom_scale),
        },
    }


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pretrained', type=Path, default=None)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--split-mode', choices=('grouped', 'random'), default='grouped')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-h', type=int, default=544)
    ap.add_argument('--train-w', type=int, default=960)
    ap.add_argument('--sigma', type=float, default=1.8)
    ap.add_argument('--sigma-mode', choices=('fixed', 'adaptive'), default='adaptive')
    ap.add_argument('--sigma-min', type=float, default=1.2)
    ap.add_argument('--sigma-max', type=float, default=3.0)
    ap.add_argument('--sigma-knn', type=int, default=3)
    ap.add_argument('--perspective-top-scale', type=float, default=0.9)
    ap.add_argument('--perspective-bottom-scale', type=float, default=1.8)
    ap.add_argument('--perspective-gamma', type=float, default=1.4)
    ap.add_argument('--loss-top-scale', type=float, default=1.0)
    ap.add_argument('--loss-bottom-scale', type=float, default=1.3)
    ap.add_argument('--loss-gamma', type=float, default=1.2)
    ap.add_argument('--lr', type=float, default=3e-6)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--min-lr', type=float, default=1e-6)
    ap.add_argument('--lr-factor', type=float, default=0.5)
    ap.add_argument('--lr-patience', type=int, default=3)
    ap.add_argument('--early-stop-patience', type=int, default=8)
    ap.add_argument('--density-loss', choices=('smoothl1', 'mse', 'l1'), default='smoothl1')
    ap.add_argument('--smoothl1-beta', type=float, default=0.5)
    ap.add_argument('--count-loss-weight', type=float, default=0.1)
    ap.add_argument('--augment', dest='augment', action='store_true')
    ap.add_argument('--no-augment', dest='augment', action='store_false')
    ap.set_defaults(augment=True)
    ap.add_argument('--noise-std', type=float, default=4.0)
    ap.add_argument('--freeze-frontend-epochs', type=int, default=0)
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    run_training(parse_args())
