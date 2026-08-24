# Backend

The image entry point runs the current safety pipeline:

1. RF-DETR Medium for `hanging_object` and `hanging_rope`.
2. YOLO26m segmentation for `person` bbox, confidence, and mask.
3. Depth Anything V3 once for an image-sized relative-depth map.
4. Geometry and depth-independent risk assessment.
5. Annotation preview, non-safe evidence, and execution timeline.

Depth values are relative and must not be interpreted as metric distances.

## Install

From `backend/`:

```powershell
python -m pip install -e ".[dev]"
```

Copy `configs/models.example.yaml` to `configs/models.local.yaml` and update
the RF-DETR checkpoint path. YOLO defaults to `auto_download: true`: when
`data/models/yolo26_medium/yolo26m-seg.pt` is absent, Ultralytics downloads the
official weight once into that Git-ignored location and reuses it on later
runs. Set `auto_download: false` for an offline-only deployment. Model weights
and local configuration are ignored by Git.

## Run one image

From the repository root:

```powershell
python backend/scripts/run_image.py `
  --image "data/samples/example.png" `
  --models-config "configs/models.local.yaml" `
  --geometry-config "configs/geometry.local.yaml" `
  --risk-config "configs/risk-policy.example.yaml" `
  --output-root "outputs"
```

The command creates `outputs/<run-id>/` with:

```text
detections.json
relative_depth.npy
model_metadata.json
detection_preview.png
annotation_preview.png
pipeline_timeline.json
masks/person_XX.npy       # only when YOLO returns masks
```

For `WARNING` or `DANGER`, the directory also contains an evidence PNG and
assessment JSON. A `SAFE` image still produces `annotation_preview.png`, but no
alert evidence artifact.

`detections.json` contains JSON-safe bbox/confidence fields and a `mask_ref`
for every persisted person mask. `relative_depth.npy` preserves float32 depth
values for later geometry processing.

## YOLO-only diagnostic

```powershell
python backend/scripts/test_yolo_person_segmenter.py `
  --image "path/to/image.png" `
  --model "path/to/yolo26m-seg.pt" `
  --confidence 0.35 `
  --output "outputs/person-segmentation.jpg"
```

Confidence thresholds use the range `0` to `1`; `0.35` means 35%.

## Run the local image API

Install the backend, keep the local model and geometry configs described above,
then run from `backend/`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`. The v1 endpoints are:

- `GET /api/v1/health`
- `POST /api/v1/detection/image` with multipart field `file` (`jpg`, `jpeg`, or
  `png`)

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/detection/image" `
  -F "file=@../data/samples/Crane_fall_zone_01.jpg"
```

Generated annotation images are served below `/evidence/...`; internal paths
and depth/mask artifacts are not exposed by the API. Models are lazy-loaded on
the first analysis and reused for later requests. Set
`CRANE_PRELOAD_MODELS=true` to load them once during application startup.

Optional environment settings:

- `CRANE_MODELS_CONFIG`
- `CRANE_GEOMETRY_CONFIG`
- `CRANE_RISK_CONFIG`
- `CRANE_EVIDENCE_ROOT`
- `CRANE_MAX_UPLOAD_BYTES`
- `CRANE_PRELOAD_MODELS`
- `CRANE_CORS_ORIGINS` (comma-separated exact origins; no wildcard)
- `CRANE_LOG_LEVEL`

Frame-local `person_id` and `load_id` values in a response are explanatory
labels only. Tracking and stable cross-frame IDs are intentionally not part of
this image API.
