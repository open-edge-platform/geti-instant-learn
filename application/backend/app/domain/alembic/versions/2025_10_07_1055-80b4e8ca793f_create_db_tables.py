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

from domain.db.constraints import CheckConstraintName, UniqueConstraintName

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
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', name=UniqueConstraintName.PROJECT_NAME)
    )
    op.create_index(
        UniqueConstraintName.SINGLE_ACTIVE_PROJECT,
        'Project',
        ['active'],
        unique=True,
        sqlite_where=sa.text('active IS 1')
    )

    op.create_table('Processor',
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT,
        'Processor',
        ['project_id', 'name'],
        unique=True,
        sqlite_where=sa.text('name IS NOT NULL')
    )

    op.create_table('Prompt',
    sa.Column('type', sa.Enum('TEXT', 'VISUAL', name='prompttype'), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('text', sa.Text(), nullable=True),
    sa.Column('frame_id', sa.Uuid(), nullable=True),
    sa.Column('thumbnail', sa.Text(), nullable=True),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.CheckConstraint(
        "(type = 'TEXT' AND text IS NOT NULL AND frame_id IS NULL) OR "
        "(type = 'VISUAL' AND frame_id IS NOT NULL AND text IS NULL)",
        name=CheckConstraintName.PROMPT_CONTENT)
    )
    op.create_index(
        UniqueConstraintName.SINGLE_TEXT_PROMPT_PER_PROJECT,
        'Prompt',
        ['project_id', 'type'],
        unique=True,
        sqlite_where=sa.text("type = 'TEXT'")
    )
    op.create_index(
        UniqueConstraintName.UNIQUE_FRAME_ID_PER_PROMPT,
        'Prompt',
        ['frame_id'],
        unique=True,
        sqlite_where=sa.text("frame_id IS NOT NULL")
    )

    op.create_table('Sink',
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
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
        sa.text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {UniqueConstraintName.SOURCE_TYPE_PER_PROJECT} "
            "ON Source (project_id, json_extract(config, '$.source_type'))"
        )
    )
    op.execute(
        sa.text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {UniqueConstraintName.SOURCE_NAME_PER_PROJECT} "
            "ON Source (project_id, json_extract(config, '$.name')) "
            "WHERE json_extract(config, '$.name') IS NOT NULL"
        )
    )
    op.create_index(
        UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT,
        'Source',
        ['project_id', 'connected'],
        unique=True,
        sqlite_where=sa.text('connected IS 1')
    )

    op.create_table('Annotation',
    sa.Column('config', sqlite.JSON(), nullable=False),
    sa.Column('label_id', sa.Uuid(), nullable=True),
    sa.Column('prompt_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['label_id'], ['Label.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['prompt_id'], ['Prompt.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('Label',
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('color', sa.String(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.UniqueConstraint('name', 'project_id', name=UniqueConstraintName.LABEL_NAME_PER_PROJECT),
    sa.ForeignKeyConstraint(['project_id'], ['Project.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_index(UniqueConstraintName.LABEL_NAME_PER_PROJECT, table_name='Label')
    op.drop_table('Label')
    op.drop_table('Annotation')
    op.execute(sa.text(f"DROP INDEX IF EXISTS {UniqueConstraintName.SOURCE_NAME_PER_PROJECT}"))
    op.execute(sa.text(f"DROP INDEX IF EXISTS {UniqueConstraintName.SOURCE_TYPE_PER_PROJECT}"))
    op.drop_index(UniqueConstraintName.SINGLE_CONNECTED_SOURCE_PER_PROJECT, table_name='Source')
    op.drop_table('Source')
    op.drop_table('Sink')
    op.drop_index(UniqueConstraintName.SINGLE_TEXT_PROMPT_PER_PROJECT, table_name='Prompt')
    op.drop_index(UniqueConstraintName.UNIQUE_FRAME_ID_PER_PROMPT, table_name='Prompt')
    op.drop_table('Prompt')
    op.drop_index(UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT, table_name='Processor')
    op.drop_table('Processor')
    op.drop_index(UniqueConstraintName.SINGLE_ACTIVE_PROJECT, table_name='Project')
    op.drop_table('Project')
