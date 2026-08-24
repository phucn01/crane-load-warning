export type RiskLevel = "SAFE" | "WARNING" | "DANGER";

export type AnalysisState =
  | "idle"
  | "selected"
  | "processing"
  | "success"
  | "error";

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
