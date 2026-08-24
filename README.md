# Crane Under Load Alert

Computer-vision safety system for detecting people beneath hanging loads and raising actionable alerts.

## Initial repository layout

- `backend/`: backend implementation will be added from the API/CV phases.
- `frontend/`: React dashboard will be added from the dashboard phase.
- `notebooks/`: research and validation notebooks.
- `configs/`: configuration templates are added with the relevant feature.
- `docs/`: architecture, safety-rule, and deployment documentation.

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
