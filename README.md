# CSRNet Finetune Docker

一个独立的 CSRNet 微调工具：

- 不依赖你现有项目代码
- 提供基础 WebUI
- 上传 `CVAT Datumaro 标注 zip`
- 原图通过 Docker volume 挂载到容器内目录
- 初始化权重通过 Docker volume 挂载到容器内目录
- 后台自动构建数据集并训练
- 适合放进有 NVIDIA GPU 的 Docker 环境中运行

## 当前能力

- 上传 `annotations zip`
- 自动展示已挂载原图目录的图片数量与预览
- 自动列出已挂载的初始化权重，直接选择
- 支持透视感知训练参数
- 页面可实时查看任务状态、训练日志、训练效果总结与权重下载
- 页面显示 GPU 名称、显存占用、利用率、温度

## 运行方式

```bash
cd csrnet-finetune-docker
docker build -t csrnet-finetune:latest .
docker run --gpus all --shm-size=8g \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v /your/raw/images:/raw_images:ro \
  -v /your/model/weights:/weights:ro \
  -e APP_RAW_IMAGES_DIR=/raw_images \
  -e APP_WEIGHTS_DIR=/weights \
  csrnet-finetune:latest
```

打开：

```text
http://localhost:7860
```

## 使用预构建镜像

如果你把项目推到 GitHub 并启用 Actions，默认会推送镜像到 GHCR：

```text
ghcr.io/<github-user-or-org>/csrnet-finetune-docker:latest
```

然后你可以直接：

```bash
cp .env.example .env
# 编辑 .env，至少填 RAW_IMAGES_HOST_DIR 和 WEIGHTS_HOST_DIR
docker compose up -d
```

## GitHub Actions

项目已附带：

```text
.github/workflows/docker-image.yml
```

推送到 `main` 后会自动构建并推送镜像到 `GHCR`。

## 数据要求

### 1. 标注

推荐上传 `CVAT -> Datumaro 1.0` 导出的 zip。

### 2. 原图

原图不走上传，直接通过 Docker volume 挂载。

例如宿主机目录：

```text
raw_images/
├── default.json
└── default/
    ├── a.jpg
    ├── b.jpg
    └── ...
```

启动容器时挂载为：

```bash
-v /your/raw/images:/raw_images:ro
```

训练时会递归搜索图片文件，所以 `default/` 这种结构是支持的。

### 3. 初始化权重

推荐把可选的基础权重也挂载进容器，例如：

```bash
-v /your/model/weights:/weights:ro
```

页面会自动列出 `/weights` 下的 `.pth/.pt/.pth.tar`，用户直接选择即可。

## 目录结构

运行后会在挂载目录中生成：

```text
data/jobs/<job_id>/
├── uploads/
├── dataset/
├── models/
├── logs/
└── state.json
```

## 说明

- 当前版本是基础 MVP，优先保证能上传、构建、训练、看日志。
- 训练默认带透视感知参数，适合博物馆斜视角近大远小场景。
- 如果通过 Cloudflare Tunnel 使用，建议只上传标注 zip，其余资源都走挂载。
