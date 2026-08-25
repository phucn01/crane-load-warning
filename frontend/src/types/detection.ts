export type RiskLevel = "SAFE" | "WARNING" | "DANGER";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectionItem {
  detection_id: string;
  source_model: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  has_mask: boolean;
}

export interface PseudoBEVPoint {
  lateral: number;
  longitudinal: number;
}

export interface PseudoBEVRectangle {
  minimum_lateral: number;
  maximum_lateral: number;
  minimum_longitudinal: number;
  maximum_longitudinal: number;
}

export interface SafetyZones {
  footprint: PseudoBEVRectangle;
  danger: PseudoBEVRectangle;
  warning: PseudoBEVRectangle;
}

export interface PersonGeometry {
  person_id: string;
  confidence: number;
  bbox: BoundingBox;
  pseudo_bev_point: PseudoBEVPoint | null;
  mask_reliable: boolean;
  quality_reasons: string[];
}

export interface LoadGeometry {
  load_id: string;
  confidence: number;
  bbox: BoundingBox;
  pseudo_bev_points: PseudoBEVPoint[];
  safety_zones: SafetyZones | null;
  quality_reasons: string[];
}

export interface GeometryResponse {
  coordinate_system: "relative_pseudo_bev_not_metric";
  depth_low: number;
  depth_high: number;
  quality_reasons: string[];
  persons: PersonGeometry[];
  loads: LoadGeometry[];
}

export interface PairAssessment {
  person_id: string;
  load_id: string;
  risk_level: RiskLevel;
  matched_zone: RiskLevel | null;
  confidence: number;
  assessment_reliable: boolean;
  quality_reasons: string[];
}

export interface AssessmentResponse {
  risk_level: RiskLevel;
  assessment_reliable: boolean;
  quality_reasons: string[];
  contributing_person_ids: string[];
  contributing_load_ids: string[];
  pairs: PairAssessment[];
}

export interface DetectionSummary {
  person_count: number;
  load_count: number;
  rope_count: number;
}

export interface EvidenceResponse {
  rgb_url: string | null;
  pseudo_bev_url: string | null;
  combined_url: string | null;
}

export interface DepthMetadata {
  height: number;
  width: number;
  dtype: string;
  finite_min: number;
  finite_max: number;
  finite_fraction: number;
  convention: string;
}

export interface ProcessingMetadata {
  pipeline_version: string;
  frame_id: string;
  image_width: number;
  image_height: number;
  depth: DepthMetadata;
  models_loaded: Record<string, boolean>;
  config_versions: Record<string, string>;
}

export interface ImageDetectionResponse {
  status: "completed";
  processing_time_ms: number;
  assessment: AssessmentResponse;
  summary: DetectionSummary;
  detections: DetectionItem[];
  geometry: GeometryResponse;
  evidence: EvidenceResponse;
  metadata: ProcessingMetadata;
}

export type VideoJobStatus = "queued" | "processing" | "completed" | "failed";

export interface VideoJobCreated {
  job_id: string;
  status: "queued";
  status_url: string;
  stream_url: string;
  frame_results_url: string;
  result_url: string;
}

export interface VideoFrameRiskResult {
  frame_number: number;
  timestamp_seconds: number;
  risk_level: RiskLevel;
}

export interface VideoFrameRiskResultsPage {
  job_id: string;
  job_status: VideoJobStatus;
  items: VideoFrameRiskResult[];
  next_after_frame: number;
  has_more: boolean;
}

export interface VideoSummary {
  processed_frames: number;
  safe_frames: number;
  warning_frames: number;
  danger_frames: number;
  max_risk_level: RiskLevel | null;
  average_processing_fps: number;
  risk_segment_count: number;
}

export interface RiskSegment {
  segment_id: string;
  start_frame: number;
  end_frame: number;
  risk_start_frame: number;
  risk_end_frame: number;
  start_seconds: number;
  end_seconds: number;
  max_risk_level: "WARNING" | "DANGER";
  warning_frame_count: number;
  danger_frame_count: number;
  frame_evidence: VideoFrameEvidence[];
  result_url: string;
  output_codec: string;
  browser_playback_compatible: boolean;
  playback_warning: string | null;
}

export interface VideoFrameEvidence {
  frame_number: number;
  timestamp_seconds: number;
  risk_level: "WARNING" | "DANGER";
  original_url: string;
  rgb_url: string;
  pseudo_bev_url: string;
}

export interface VideoJob {
  job_id: string;
  status: VideoJobStatus;
  input_path: string;
  output_path: string;
  current_frame: number;
  total_frames: number;
  progress: number;
  processing_fps: number;
  elapsed_seconds: number;
  current_risk_level: RiskLevel | null;
  max_risk_level: RiskLevel | null;
  safe_frame_count: number;
  warning_frame_count: number;
  danger_frame_count: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  stream_url: string;
  frame_results_url: string;
  result_url: string | null;
  download_url: string | null;
  report_url: string | null;
  summary: VideoSummary | null;
  risk_segments: RiskSegment[];
  output_codec: string;
  browser_playback_compatible: boolean;
  playback_warning: string | null;
}

export interface VideoReportEvidence {
  frame_number: number;
  timestamp_seconds: number;
  risk_level: "WARNING" | "DANGER";
  original_url: string;
  rgb_url: string;
  pseudo_bev_url: string;
}

export interface VideoReportSegment {
  segment_id: string;
  start_frame: number;
  end_frame: number;
  risk_start_frame: number;
  risk_end_frame: number;
  start_seconds: number;
  end_seconds: number;
  max_risk_level: "WARNING" | "DANGER";
  warning_frame_count: number;
  danger_frame_count: number;
  result_url: string;
  codec: string;
  browser_playback_compatible: boolean;
  playback_warning: string | null;
  evidence: VideoReportEvidence[];
}

export interface VideoReport {
  schema_version: string;
  job_id: string;
  status: "completed";
  created_at: string;
  started_at: string | null;
  completed_at: string;
  input_filename: string;
  summary: {
    processed_frames: number;
    total_frames: number;
    safe_frames: number;
    warning_frames: number;
    danger_frames: number;
    max_risk_level: RiskLevel | null;
    average_processing_fps: number;
    elapsed_seconds: number;
    risk_segment_count: number;
  };
  video: {
    filename: string;
    url: string;
    download_url: string;
    codec: string;
    browser_playback_compatible: boolean;
    playback_warning: string | null;
  };
  risk_segments: VideoReportSegment[];
}
