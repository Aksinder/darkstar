"""Add slot_device_energy table (per-device slot energy for savings v2)

Revision ID: c7a91b2f4d10
Revises: 65db61e6fcae
Create Date: 2026-07-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a91b2f4d10"
down_revision: str | None = "65db61e6fcae"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the per-device slot energy table (composite PK slot_start+device_id)."""
    op.create_table(
        "slot_device_energy",
        sa.Column("slot_start", sa.String(), nullable=False),
        sa.Column("device_id", sa.String(), nullable=False),
        sa.Column("kwh", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("slot_start", "device_id"),
    )


def downgrade() -> None:
    """Drop the per-device slot energy table."""
    op.drop_table("slot_device_energy")
