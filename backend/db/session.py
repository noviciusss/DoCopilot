import os
import logging
import ssl
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

# Fallback default for local dev
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://docopilot:docopilot@localhost:5432/docopilot")

# Convert standard postgres:// or postgresql:// to asyncpg driver syntax
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Clean up query params: asyncpg uses 'ssl' instead of 'sslmode'
if "sslmode=require" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("sslmode=require", "ssl=require")

connect_args = {}
# If connecting to cloud PostgreSQL (Neon, Azure, AWS), configure SSL context
# Disable prepared statement caching for cloud PostgreSQL compatibility (Neon/pgBouncer)
connect_args["statement_cache_size"] = 0
connect_args["prepared_statement_cache_size"] = 0

if "neon.tech" in DATABASE_URL or "azure" in DATABASE_URL or "ssl=" in DATABASE_URL or "sslmode=" in DATABASE_URL:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connect_args["ssl"] = ssl_ctx

engine = create_async_engine(

    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass 

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()