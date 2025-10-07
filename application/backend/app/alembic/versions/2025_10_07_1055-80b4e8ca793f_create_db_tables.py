# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Create DB tables

Revision ID: 80b4e8ca793f
Revises: 
Create Date: 2025-10-07 10:55:32.661994+00:00

"""

# DO NOT EDIT MANUALLY EXISTING MIGRATIONS.

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '80b4e8ca793f'
down_revision: str | None = None
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    op.create_table('Project',
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('single_active_project', 'Project', ['active'], unique=True, sqlite_where=sa.text('active IS 1'))

    op.create_table('Processor',
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('type', sa.Enum('DUMMY', name='processortype'), nullable=False),
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_id')
    )

    op.create_table('Prompt',
    sa.Column('type', sa.Enum('TEXT', 'VISUAL', name='prompttype'), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('text', sa.String(), nullable=True),
    sa.Column('image_path', sa.String(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('Sink',
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_id')
    )

    op.create_table('Source',
    sa.Column('connected', sa.Boolean(), nullable=False),
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_source_type_per_project "
        "ON Source (project_id, json_extract(config, '$.source_type'))"
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
    sa.Column('project_id', sa.Uuid(), nullable=True),
    sa.Column('prompt_id', sa.Uuid(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint('project_id IS NOT NULL OR prompt_id IS NOT NULL', name='label_parent_check'),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['prompt_id'], ['Prompt.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('Label')
    op.drop_table('Annotation')
    op.execute("DROP INDEX IF EXISTS uq_source_type_per_project")
    op.drop_table('Source')
    op.drop_table('Sink')
    op.drop_table('Prompt')
    op.drop_table('Processor')
    op.drop_index('single_active_project', table_name='Project', sqlite_where=sa.text('active IS 1'))
    op.drop_table('Project')
