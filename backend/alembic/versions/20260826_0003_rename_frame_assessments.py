"""Rename risk snapshots to frame assessments."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260826_0003"
down_revision: str | None = "20260826_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("risk_snapshots", "frame_assessments")
    for old, new in (
        ("ix_risk_snapshots_job_id", "ix_frame_assessments_job_id"),
        ("ix_risk_snapshots_created_at", "ix_frame_assessments_created_at"),
        ("ix_risk_snapshots_risk_level", "ix_frame_assessments_risk_level"),
    ):
        op.execute(f'ALTER INDEX "{old}" RENAME TO "{new}"')


def downgrade() -> None:
    for old, new in (
        ("ix_frame_assessments_job_id", "ix_risk_snapshots_job_id"),
        ("ix_frame_assessments_created_at", "ix_risk_snapshots_created_at"),
        ("ix_frame_assessments_risk_level", "ix_risk_snapshots_risk_level"),
    ):
        op.execute(f'ALTER INDEX "{old}" RENAME TO "{new}"')
    op.rename_table("frame_assessments", "risk_snapshots")
