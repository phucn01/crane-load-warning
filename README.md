# Crane Under Load Alert

Computer-vision safety system for detecting people beneath hanging loads and raising actionable alerts.

## Initial repository layout

- `backend/`: backend implementation will be added from the API/CV phases.
- `frontend/`: React dashboard will be added from the dashboard phase.
- `notebooks/`: research and validation notebooks.
- `configs/`: configuration templates are added with the relevant feature.
- `docs/`: architecture, safety-rule, and deployment documentation.

## Phase 1: offline single-image inference

The backend now runs RF-DETR load/rope detection, YOLO26m person
segmentation, and one Depth Anything V3 relative-depth inference for a local
image. Configure the RF-DETR checkpoint in an untracked copy of
`configs/models.example.yaml`; the official YOLO26m-seg weight downloads
automatically when absent. Then run:

```powershell
python backend/scripts/run_offline_image.py `
  --image "path/to/input.png" `
  --config "configs/models.local.yaml"
```

See `backend/README.md` for installation and artifact details. Relative depth
and later Pseudo-BEV geometry are non-metric until camera calibration exists.
