"""Persist a downloadable JSON summary for a completed video job."""

from __future__ import annotations

import json
from pathlib import Path

from app.models import JobStatus, VideoJob


def write_video_report(job: VideoJob) -> Path:
    """Write the public report atomically and return its final path."""
    if job.status is not JobStatus.COMPLETED:
        raise ValueError("video report requires a completed job")

    report_path = job.report_path
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    payload = _report_payload(job)
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report_path


def _report_payload(job: VideoJob) -> dict[str, object]:
    prefix = f"/api/v1/jobs/{job.job_id}"
    return {
        "schema_version": "1.0",
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "started_at": None if job.started_at is None else job.started_at.isoformat(),
        "completed_at": (
            None if job.completed_at is None else job.completed_at.isoformat()
        ),
        "input_filename": job.input_path.name,
        "summary": {
            "processed_frames": job.current_frame,
            "total_frames": job.total_frames,
            "safe_frames": job.safe_frame_count,
            "warning_frames": job.warning_frame_count,
            "danger_frames": job.danger_frame_count,
            "max_risk_level": job.max_risk_level,
            "average_processing_fps": job.processing_fps,
            "elapsed_seconds": job.elapsed_seconds,
            "risk_segment_count": len(job.risk_segments),
        },
        "video": {
            "filename": job.output_path.name,
            "url": f"{prefix}/result",
            "download_url": f"{prefix}/download",
            "codec": job.output_codec,
            "browser_playback_compatible": job.browser_playback_compatible,
            "playback_warning": job.playback_warning,
        },
        "risk_segments": [
            {
                "segment_id": segment.segment_id,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "risk_start_frame": segment.risk_start_frame,
                "risk_end_frame": segment.risk_end_frame,
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "max_risk_level": segment.max_risk_level,
                "warning_frame_count": segment.warning_frame_count,
                "danger_frame_count": segment.danger_frame_count,
                "result_url": f"{prefix}/segments/{segment.segment_id}",
                "codec": segment.output_codec,
                "browser_playback_compatible": (
                    segment.browser_playback_compatible
                ),
                "playback_warning": segment.playback_warning,
                "evidence": [
                    {
                        "frame_number": item.frame_number,
                        "timestamp_seconds": item.timestamp_seconds,
                        "risk_level": item.risk_level,
                        "original_url": (
                            f"{prefix}/segments/{segment.segment_id}/evidence/"
                            f"{item.frame_number}/original"
                        ),
                        "rgb_url": (
                            f"{prefix}/segments/{segment.segment_id}/evidence/"
                            f"{item.frame_number}/rgb"
                        ),
                        "pseudo_bev_url": (
                            f"{prefix}/segments/{segment.segment_id}/evidence/"
                            f"{item.frame_number}/bev"
                        ),
                    }
                    for item in segment.frame_evidence
                ],
            }
            for segment in job.risk_segments
        ],
    }


__all__ = ["write_video_report"]
