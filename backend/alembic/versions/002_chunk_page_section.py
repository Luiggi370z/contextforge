"""add page/section columns to chunks

Revision ID: 002_chunk_page_section
Revises: 001_initial
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "002_chunk_page_section"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("section", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("page", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "page")
    op.drop_column("chunks", "section")
