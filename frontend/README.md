# Crane Load Warning frontend

React + TypeScript + Vite interface for the local image safety assessment API.

## Development

Start the FastAPI backend on port 8000, then from `frontend/`:

```bash
npm install
npm run dev
```

Open `http://localhost:5173`. The interface supports click selection and drag
and drop for JPG/JPEG/PNG images, local preview, explicit analysis, assessment
quality details, and RGB/Pseudo-BEV/combined evidence.

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

