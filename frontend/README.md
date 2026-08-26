# Crane Load Warning frontend

React + TypeScript + Vite interface for the local image safety assessment API.

## Development

Start the FastAPI backend on port 8000, then from `frontend/`:

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. One smart uploader accepts images and videos,
detects the media from file headers/MIME/extension, and shows a media-specific
confirmation action before routing it to the existing image or video API.
Image results retain local preview,
assessment quality details, and RGB/Pseudo-BEV/combined evidence; video results
retain live processing preview, progress, full output, and risk clips.

## Environment

Copy `.env.example` to `.env.local` when the backend is not available at the
default URL:

```text
VITE_API_BASE_URL=http://localhost:8000
```

The backend CORS allowlist defaults to `http://localhost:5173` and
`http://127.0.0.1:5173`. Override it with a comma-separated
`CRANE_CORS_ORIGINS` value when required.

## Production build

```bash
npm run build
```

The generated `dist/` directory is ignored by Git.

## Video processing

After a video upload the page polls the in-memory job every
750 ms, shows the latest annotated MJPEG processing preview, stops polling at
`completed` or `failed`, and uses HTML5 `<video controls>` for the result.
Preview frames are not playback at source FPS, and SAFE/WARNING/DANGER counts
describe independently assessed frames rather than safety events.
Completed results fetch the paginated frame-risk feed and merge adjacent frames
into an interactive SAFE/WARNING/DANGER timeline synchronized with the main
video. Timeline ranges and evidence markers seek the player to their exact
timestamps. WARNING/DANGER clips retain 2-second pre-roll and post-roll context
by default, but are shown as lightweight review cards rather than separate
embedded players. Each card can jump the main player to its risk start, open the
saved clip, or show annotated/Pseudo-BEV evidence for up to three representative
risk frames (first, highest-risk, and last). These clips and images are
frame-level risk evidence, not tracked safety events.
The dedicated video report reuses the same frame-risk timeline with a compact
segment review queue, a three-view evidence panel, interpretation notes, and a
print layout that removes interactive controls.
The backend normally finalizes generated files as H.264/yuv420p/faststart for
HTML5 playback. When it must fall back to `mp4v`, the result view displays a
codec compatibility warning.

Run frontend tests with:

```bash
npm test
```

> Current video processing evaluates detections independently per frame. Cross-frame person/load identity is intentionally not tracked.

## History

Open `/?history=1` or use the **History** link in the header. The page reads
processing jobs and sampled WARNING/DANGER risk snapshots exclusively through
the FastAPI endpoints. It never connects to Supabase directly. Risk snapshots
are time/frame evidence and are intentionally not labelled as safety events.

