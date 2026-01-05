"""Add chat_memories table

Revision ID: 001
Revises:
Create Date: 2025-01-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create chat_memories table
    op.create_table(
        'chat_memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('query_hash', sa.String(64), nullable=False),
        sa.Column('sources_used', sa.Text(), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('relevance_score', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create indexes
    op.create_index('idx_chat_memories_query_hash', 'chat_memories', ['query_hash'])
    op.create_index('idx_chat_memories_user_id', 'chat_memories', ['user_id'])
    op.create_index('idx_chat_memories_created_at', 'chat_memories', ['created_at'])
    op.create_index('idx_query_hash_created', 'chat_memories', ['query_hash', 'created_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_query_hash_created', table_name='chat_memories')
    op.drop_index('idx_chat_memories_created_at', table_name='chat_memories')
    op.drop_index('idx_chat_memories_user_id', table_name='chat_memories')
    op.drop_index('idx_chat_memories_query_hash', table_name='chat_memories')

    # Drop table
    op.drop_table('chat_memories')
