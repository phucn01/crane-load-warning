"""Allow SAFE frame assessments in the existing history table."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_0002"
down_revision: str | None = "20260825_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "risk_snapshots",
        sa.Column(
            "assessment_status",
            sa.String(length=32),
            nullable=False,
            server_default="FULL_EVALUATION",
        ),
    )
    op.alter_column("risk_snapshots", "risk_level", nullable=True)
    op.alter_column("risk_snapshots", "assessment_reliable", nullable=True)
    op.drop_constraint("ck_snapshot_risk_level", "risk_snapshots", type_="check")
    op.create_check_constraint(
        "ck_snapshot_risk_level",
        "risk_snapshots",
        "risk_level is null or risk_level in ('SAFE', 'WARNING', 'DANGER')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_snapshot_risk_level", "risk_snapshots", type_="check")
    op.create_check_constraint(
        "ck_snapshot_risk_level",
        "risk_snapshots",
        "risk_level in ('WARNING', 'DANGER')",
    )
    op.alter_column("risk_snapshots", "assessment_reliable", nullable=False)
    op.alter_column("risk_snapshots", "risk_level", nullable=False)
    op.drop_column("risk_snapshots", "assessment_status")
