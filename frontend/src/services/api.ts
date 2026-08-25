import type {
  ImageDetectionResponse,
  VideoJob,
  VideoJobCreated,
  VideoFrameRiskResultsPage,
  VideoReport,
} from "../types/detection";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/$/, "");

interface ErrorPayload {
  detail?: string | Array<{ msg?: string }>;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function analyzeImage(
  file: File,
  signal?: AbortSignal,
): Promise<ImageDetectionResponse> {
  const form = new FormData();
  form.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/detection/image`, {
      method: "POST",
      body: form,
      signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      "Cannot reach the analysis service. Check that the backend is running.",
      0,
    );
  }

  if (!response.ok) {
    const payload = await readErrorPayload(response);
    throw new ApiError(errorMessage(payload, response.status), response.status);
  }

  return (await response.json()) as ImageDetectionResponse;
}

export function evidenceUrl(path: string | null): string | null {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export const apiUrl = evidenceUrl;

export async function uploadVideo(
  file: File,
  signal?: AbortSignal,
): Promise<VideoJobCreated> {
  const form = new FormData();
  form.append("file", file);
  return requestJson<VideoJobCreated>("/api/v1/detection/video", {
    method: "POST",
    body: form,
    signal,
  });
}

export async function getVideoJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<VideoJob> {
  return requestJson<VideoJob>(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    signal,
  });
}

export async function getVideoFrameResults(
  jobId: string,
  afterFrame: number,
  limit = 1000,
  signal?: AbortSignal,
): Promise<VideoFrameRiskResultsPage> {
  const query = new URLSearchParams({
    after_frame: String(afterFrame),
    limit: String(limit),
  });
  return requestJson<VideoFrameRiskResultsPage>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/frames?${query}`,
    { signal },
  );
}

export async function getAllVideoFrameResults(
  jobId: string,
  signal?: AbortSignal,
): Promise<VideoFrameRiskResultsPage["items"]> {
  const results: VideoFrameRiskResultsPage["items"] = [];
  let afterFrame = 0;
  let hasMore = true;
  while (hasMore) {
    const page = await getVideoFrameResults(jobId, afterFrame, 1000, signal);
    results.push(...page.items);
    afterFrame = page.next_after_frame;
    hasMore = page.has_more && page.items.length > 0;
  }
  return results;
}

export async function getVideoReport(jobId: string): Promise<VideoReport> {
  return requestJson<VideoReport>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/report`,
    {},
  );
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(
      "Cannot reach the analysis service. Check that the backend is running.",
      0,
    );
  }
  if (!response.ok) {
    const payload = await readErrorPayload(response);
    throw new ApiError(errorMessage(payload, response.status), response.status);
  }
  return (await response.json()) as T;
}

async function readErrorPayload(response: Response): Promise<ErrorPayload | null> {
  try {
    return (await response.json()) as ErrorPayload;
  } catch {
    return null;
  }
}

function errorMessage(payload: ErrorPayload | null, status: number): string {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    const messages = payload.detail
      .map((item) => item.msg)
      .filter((message): message is string => Boolean(message));
    if (messages.length) return messages.join("; ");
  }
  return `Analysis failed with HTTP ${status}.`;
}
