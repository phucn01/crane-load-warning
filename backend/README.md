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
- `POST /api/v1/detection/video` with multipart field `file` (`mp4`, `mov`,
  `avi`, `mkv`, or `webm`); returns an in-memory job after upload
- `GET /api/v1/jobs/{job_id}` for progress and frame-risk statistics
- `GET /api/v1/jobs/{job_id}/stream` for the latest annotated processing frame
  as `multipart/x-mixed-replace` MJPEG
- `GET /api/v1/jobs/{job_id}/result` for the completed annotated MP4
- `GET /api/v1/jobs/{job_id}/segments/{segment_id}` for a finalized padded
  WARNING/DANGER frame segment
- `GET /api/v1/jobs/{job_id}/segments/{segment_id}/evidence/{frame_number}/rgb`
  for a representative annotated frame PNG
- `GET /api/v1/jobs/{job_id}/segments/{segment_id}/evidence/{frame_number}/bev`
  for that frame's Pseudo-BEV PNG

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
- `CRANE_VIDEO_UPLOAD_ROOT`
- `CRANE_VIDEO_OUTPUT_ROOT`
- `CRANE_MAX_VIDEO_UPLOAD_BYTES`
- `CRANE_RISK_SEGMENT_PRE_ROLL_SECONDS` (default `2.0`)
- `CRANE_RISK_SEGMENT_POST_ROLL_SECONDS` (default `2.0`)
- `CRANE_FFMPEG_PATH` (optional explicit FFmpeg executable)
- `CRANE_PRELOAD_MODELS`
- `CRANE_CORS_ORIGINS` (comma-separated exact origins; no wildcard)
- `CRANE_LOG_LEVEL`

Frame-local `person_id` and `load_id` values in a response are explanatory
labels only. Tracking and stable cross-frame IDs are intentionally not part of
this image API.

## Video processing architecture

Uploaded videos use UUID filenames under `storage/uploads/videos`. The HTTP
handler creates an in-memory `queued` job and submits blocking OpenCV and
inference work to a single-worker `ThreadPoolExecutor`, keeping the FastAPI
event loop available. The worker reads each frame once and reuses the image
service's Vision -> Geometry -> Risk -> Annotation frame pipeline. The same
annotated frame is JPEG-encoded into the one latest-preview slot and written in
order to the output video; individual frame JPEGs are never stored.

Job states are `queued`, `processing`, `completed`, and `failed`. Counts are
frame statistics (`safe_frame_count`, `warning_frame_count`, and
`danger_frame_count`), not event counts. Runtime event association is deferred
until tracking is introduced. The MJPEG endpoint is a processing preview, not
source-FPS playback.

WARNING/DANGER clips are written directly during processing. A compressed
in-memory ring buffer supplies the configured pre-roll; the active segment
writer stays open through the configured SAFE post-roll. A later DANGER frame
upgrades the segment's maximum level, while counts retain the exact number of
WARNING and DANGER classifications. Up to three representative non-safe frames
are retained per segment: the first risk frame, the first frame at the segment's
highest risk level, and the final risk frame. Duplicate selections collapse to
one item. Each selection stores an annotated RGB PNG and Pseudo-BEV PNG produced
during the original frame inference. These are contiguous frame-level risk
segments, not tracked safety events.

Risk clips are stored as UUID-named MP4 files in a sibling directory beside
the full output video:

```text
storage/outputs/videos/
├── <output-uuid>.mp4
└── <output-uuid>_segments/
    └── <segment-uuid>.mp4
```

Output preserves source width, height, ordering, and a valid source FPS (or a
25 FPS fallback). OpenCV first writes MP4 with `mp4v` and always releases both
capture and writer. After inference, FFmpeg finalizes the full output and every
risk clip as H.264 (`libx264`), `yuv420p`, `avc1`, with `faststart` for browser
playback. The project dependency `imageio-ffmpeg` supplies a platform binary;
`CRANE_FFMPEG_PATH` can override it. If FFmpeg is unavailable or conversion
fails, the original `mp4v` remains available and the API/frontend expose an
explicit playback compatibility warning.

> Current video processing evaluates detections independently per frame. Cross-frame person/load identity is intentionally not tracked.
