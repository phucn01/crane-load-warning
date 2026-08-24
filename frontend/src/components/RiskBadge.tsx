import type { RiskLevel } from "../types/detection";

interface RiskBadgeProps {
  level: RiskLevel;
  compact?: boolean;
}

const LABELS: Record<RiskLevel, string> = {
  SAFE: "Safe",
  WARNING: "Warning",
  DANGER: "Danger",
};

export default function RiskBadge({ level, compact = false }: RiskBadgeProps) {
  return (
    <span className={`risk-badge risk-${level.toLowerCase()} ${compact ? "risk-compact" : ""}`}>
      <span className="risk-dot" aria-hidden="true" />
      {LABELS[level]}
    </span>
  );
}
