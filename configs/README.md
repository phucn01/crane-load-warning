# Configuration

Copy example files into an environment-specific, untracked configuration before adding real credentials.

For Phase 1, copy `models.example.yaml` to `models.local.yaml`, then set the
local RF-DETR checkpoint path. The official YOLO26m-seg weight is downloaded
automatically to its configured checkpoint target when absent. Depth Anything
V3 may use a Hugging Face model identifier or an already-cached local model.

