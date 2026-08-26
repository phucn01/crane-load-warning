# Crane Under Load Alert

Computer-vision safety system for detecting people beneath hanging loads and raising actionable alerts.

## Initial repository layout

- `backend/`: backend implementation will be added from the API/CV phases.
- `frontend/`: React dashboard will be added from the dashboard phase.
- `notebooks/`: research and validation notebooks.
- `configs/`: configuration templates are added with the relevant feature.
- `docs/`: architecture, safety-rule, and deployment documentation.

## Install and run the backend

From the repository root, create a virtual environment and install the backend
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".\backend[dev]"
```

Create the local model configuration from the example file:

```powershell
Copy-Item configs\models.example.yaml configs\models.local.yaml
```

Start the FastAPI backend:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`, with Swagger UI at
`http://127.0.0.1:8000/docs`. The first startup may download the configured
RF-DETR checkpoint from Hugging Face when it is not present locally.

## Run the image safety pipeline

The backend runs Vision, Geometry, Risk, and Annotation for one local image.
The command produces vision artifacts, an annotated camera/Pseudo-BEV preview,
non-safe evidence when required, and an execution timeline. Configure the
RF-DETR checkpoint in an untracked copy of
`configs/models.example.yaml`; the official YOLO26m-seg weight downloads
automatically when absent. Then run:

```powershell
python backend/scripts/run_image.py `
  --image "path/to/input.png" `
  --models-config "configs/models.local.yaml" `
  --geometry-config "configs/geometry.local.yaml" `
  --risk-config "configs/risk-policy.example.yaml"
```

See `backend/README.md` for installation and artifact details. Relative depth
and Pseudo-BEV geometry are non-metric until camera calibration exists.

## Vision models

The pipeline uses two specialized detectors. Each model is responsible for a
different type of object:

### Hanging-load detection — RF-DETR

The system uses the [Crane Safety Zone Model](https://huggingface.co/phucn001/crane_safety_zone_model)
checkpoint to detect:

- `hanging_object`
- `hanging_rope`

### Person detection — YOLO26m-seg

People are detected with the Ultralytics YOLO26m-seg model, which is downloaded
from Ultralytics when the configured local weight is missing.

### Fine-tuned models

- [HANGCON RF-DETR Medium](https://huggingface.co/phucn001/hangcon-rfdetr-medium) — fine-tuned for `hanging_object` and `hanging_rope`.
- [HANGCON YOLO26 Base](https://huggingface.co/phucn001/hangcon-yolo26-base) — fine-tuned for `hanging_object` and `hanging_rope`.

The fine-tuned label scope is limited to these two hanging-load classes; it
does not represent a general-purpose object detector.

The RF-DETR checkpoint is downloaded automatically into
`data/models/rfdetr_medium/` when it is not available locally.

## Run the image assessment interface

Keep the backend running on port 8000. In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to select or drop a JPG/PNG image, preview it
locally, run the safety assessment, and review the risk result and evidence.
Set `VITE_API_BASE_URL` when the API is hosted at a different origin.
The backend preloads RF-DETR, YOLO, and Depth Anything before it becomes ready,
so the first assessment from the frontend does not reload model weights. Set
`CRANE_PRELOAD_MODELS=false` before starting the backend only when lazy loading
is preferred.
