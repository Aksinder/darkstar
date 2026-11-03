"""add cyclic_kwh to slot_observations

Cyclic-load energy (pool pump, filter) that the recorder/repair subtracted from
load_kwh (base load). Money consumers (savings, hindsight benchmark) add it back
to reconstruct whole-house consumption. Nullable, no server default: NULL marks a
row written before this column existed.

Revision ID: f3c1d2e4a5b6
Revises: c7a91b2f4d10
Create Date: 2026-11-03 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3c1d2e4a5b6"
down_revision: str | Sequence[str] | None = "c7a91b2f4d10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "slot_observations" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("slot_observations")]
        if "cyclic_kwh" not in columns:
            op.add_column(
                "slot_observations",
                sa.Column("cyclic_kwh", sa.Float(), nullable=True),
            )


def downgrade() -> None:
    """Downgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "slot_observations" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("slot_observations")]
        if "cyclic_kwh" in columns:
            with op.batch_alter_table("slot_observations", schema=None) as batch_op:
                batch_op.drop_column("cyclic_kwh")
