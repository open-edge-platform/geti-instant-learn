# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Create DB tables

Revision ID: 8b19996bfac7
Revises: 
Create Date: 2025-09-02 10:50:18.129567+00:00

"""

# DO NOT EDIT MANUALLY EXISTING MIGRATIONS.

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '8b19996bfac7'
down_revision: str | None = None
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    op.create_table('Pipeline',
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('Processor',
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('type', sa.Enum('DUMMY', name='processortype'), nullable=False),
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('pipeline_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_id'], ['Pipeline.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_id')
    )
    op.create_table('Prompt',
    sa.Column('type', sa.Enum('TEXT', 'VISUAL', name='prompttype'), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('pipeline_id', sa.Uuid(), nullable=False),
    sa.Column('text', sa.String(), nullable=True),
    sa.Column('image_path', sa.String(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_id'], ['Pipeline.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('Sink',
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('pipeline_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_id'], ['Pipeline.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_id')
    )
    op.create_table('Source',
    sa.Column('type', sa.Enum('VIDEO_FILE', 'WEB_CAMERA', 'IMAGE_DIRECTORY', name='sourcetype'), nullable=False),
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('pipeline_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_id'], ['Pipeline.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('pipeline_id')
    )
    op.create_table('Annotation',
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('prompt_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['prompt_id'], ['Prompt.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('prompt_id')
    )
    op.create_table('Label',
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('color', sa.String(), nullable=False),
    sa.Column('pipeline_id', sa.Uuid(), nullable=True),
    sa.Column('prompt_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint('pipeline_id IS NOT NULL OR prompt_id IS NOT NULL', name='label_parent_check'),
    sa.ForeignKeyConstraint(['pipeline_id'], ['Pipeline.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['prompt_id'], ['Prompt.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('Label')
    op.drop_table('Annotation')
    op.drop_table('Source')
    op.drop_table('Sink')
    op.drop_table('Prompt')
    op.drop_table('Processor')
    op.drop_table('Pipeline')
