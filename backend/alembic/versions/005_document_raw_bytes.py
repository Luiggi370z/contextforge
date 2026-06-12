"""add size_bytes + raw_bytes to documents

Revision ID: 005_document_raw_bytes
Revises: 004_eval_tables
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "005_document_raw_bytes"
down_revision = "004_eval_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    op.add_column("documents", sa.Column("raw_bytes", sa.LargeBinary(), nullable=False))


def downgrade() -> None:
    op.drop_column("documents", "raw_bytes")
    op.drop_column("documents", "size_bytes")
