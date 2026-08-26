"""Create processing jobs and sampled risk snapshots.

Revision ID: 20260825_0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260825_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("media_type", sa.String(length=16), nullable=False),
        sa.Column("input_name", sa.Text(), nullable=False),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=True),
        sa.Column("processed_frames", sa.Integer(), nullable=True),
        sa.Column("safe_frame_count", sa.Integer(), nullable=False),
        sa.Column("warning_frame_count", sa.Integer(), nullable=False),
        sa.Column("danger_frame_count", sa.Integer(), nullable=False),
        sa.Column("max_risk_level", sa.String(length=16), nullable=True),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("average_processing_fps", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "media_type in ('image', 'video')",
            name="ck_job_media_type",
        ),
        sa.CheckConstraint(
            "status in ('queued', 'processing', 'completed', 'failed')",
            name="ck_job_status",
        ),
        sa.CheckConstraint(
            "max_risk_level is null or "
            "max_risk_level in ('SAFE', 'WARNING', 'DANGER')",
            name="ck_job_max_risk_level",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_processing_jobs_created_at", "processing_jobs", ["created_at"])
    op.create_index("ix_processing_jobs_media_type", "processing_jobs", ["media_type"])
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"])
    op.create_table(
        "risk_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("timestamp_sec", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("assessment_reliable", sa.Boolean(), nullable=False),
        sa.Column("quality_reasons", sa.JSON(), nullable=False),
        sa.Column("evidence_path", sa.Text(), nullable=True),
        sa.Column("rgb_evidence_path", sa.Text(), nullable=True),
        sa.Column("pseudo_bev_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "risk_level in ('WARNING', 'DANGER')",
            name="ck_snapshot_risk_level",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["processing_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_snapshots_created_at", "risk_snapshots", ["created_at"])
    op.create_index("ix_risk_snapshots_job_id", "risk_snapshots", ["job_id"])
    op.create_index("ix_risk_snapshots_risk_level", "risk_snapshots", ["risk_level"])


def downgrade() -> None:
    op.drop_index("ix_risk_snapshots_risk_level", table_name="risk_snapshots")
    op.drop_index("ix_risk_snapshots_job_id", table_name="risk_snapshots")
    op.drop_index("ix_risk_snapshots_created_at", table_name="risk_snapshots")
    op.drop_table("risk_snapshots")
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_media_type", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_created_at", table_name="processing_jobs")
    op.drop_table("processing_jobs")
