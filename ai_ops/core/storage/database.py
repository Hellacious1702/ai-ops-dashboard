"""
Database connection and session management.
Provides async SQLite with connection pooling and Parquet export capabilities.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_ops.shared.config import get_settings
from ai_ops.shared.models import Base


class DatabaseManager:
    """
    Manages database connections and operations.
    
    WHY: Centralized database management ensures consistent connection handling,
    proper pooling, and enables features like Parquet export for analytics.
    Uses async SQLAlchemy for non-blocking I/O in the dashboard.
    """

    def __init__(self, settings: object | None = None) -> None:
        self.settings = settings or get_settings()
        self.db_path = self.settings.database.sqlite_path
        self.parquet_path = self.settings.database.parquet_export_path
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._export_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize database connections and create tables."""
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_path.mkdir(parents=True, exist_ok=True)

        # Async engine for application use
        self._async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # Sync engine for migrations and Parquet export
        self._sync_engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # Enable WAL mode for better concurrency
        @event.listens_for(self._sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

        @event.listens_for(self._async_engine.sync_engine, "connect")
        def set_async_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

        # Create session factories
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._sync_session_factory = sessionmaker(
            self._sync_engine,
            expire_on_commit=False,
        )

        # Create tables
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Start periodic Parquet export
        if self.settings.database.export_interval_seconds > 0:
            self._export_task = asyncio.create_task(self._periodic_export())

    async def close(self) -> None:
        """Close database connections."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not self._async_session_factory:
            await self.initialize()

        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def sync_session(self):
        """Get a sync database session (for Parquet export)."""
        if not self._sync_session_factory:
            raise RuntimeError("Database not initialized")
        return self._sync_session_factory()

    async def _periodic_export(self) -> None:
        """Periodically export data to Parquet format."""
        while True:
            try:
                await asyncio.sleep(self.settings.database.export_interval_seconds)
                await self.export_to_parquet()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Parquet export failed: {e}")

    async def export_to_parquet(self) -> dict[str, int]:
        """
        Export all tables to Parquet files for analytics.
        
        WHY: Parquet is columnar format optimized for analytical queries.
        This enables fast dashboard queries and external BI tool integration.
        Runs in sync context to avoid blocking the async event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._export_to_parquet_sync)

    def _export_to_parquet_sync(self) -> dict[str, int]:
        """Synchronous Parquet export implementation."""
        results = {}

        with self.sync_session() as session:
            # Export LLM calls
            calls_df = pd.read_sql(
                "SELECT * FROM llm_calls",
                self._sync_engine,
            )
            if not calls_df.empty:
                calls_path = self.parquet_path / "llm_calls.parquet"
                calls_df.to_parquet(calls_path, index=False)
                results["llm_calls"] = len(calls_df)

            # Export metric snapshots
            metrics_df = pd.read_sql(
                "SELECT * FROM metric_snapshots",
                self._sync_engine,
            )
            if not metrics_df.empty:
                metrics_path = self.parquet_path / "metric_snapshots.parquet"
                metrics_df.to_parquet(metrics_path, index=False)
                results["metric_snapshots"] = len(metrics_df)

            # Export recommendations
            recs_df = pd.read_sql(
                "SELECT * FROM recommendations",
                self._sync_engine,
            )
            if not recs_df.empty:
                recs_path = self.parquet_path / "recommendations.parquet"
                recs_df.to_parquet(recs_path, index=False)
                results["recommendations"] = len(recs_df)

            # Export alerts
            alerts_df = pd.read_sql(
                "SELECT * FROM alerts",
                self._sync_engine,
            )
            if not alerts_df.empty:
                alerts_path = self.parquet_path / "alerts.parquet"
                alerts_df.to_parquet(alerts_path, index=False)
                results["alerts"] = len(alerts_df)

        return results

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim space."""
        async with self._async_engine.begin() as conn:
            await conn.execute("VACUUM")

    async def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        async with self.session() as session:
            from sqlalchemy import func, select

            from ai_ops.shared.models import (
                AlertModel,
                LLMCallModel,
                MetricSnapshotModel,
                RecommendationModel,
            )

            calls = await session.scalar(select(func.count(LLMCallModel.id)))
            metrics = await session.scalar(select(func.count(MetricSnapshotModel.id)))
            recs = await session.scalar(select(func.count(RecommendationModel.id)))
            alerts = await session.scalar(select(func.count(AlertModel.id)))

            return {
                "llm_calls": calls or 0,
                "metric_snapshots": metrics or 0,
                "recommendations": recs or 0,
                "alerts": alerts or 0,
            }


# Global database manager instance
_db_manager: DatabaseManager | None = None


async def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


async def close_db_manager() -> None:
    """Close the global database manager."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session (dependency injection friendly)."""
    manager = await get_db_manager()
    async with manager.session() as session:
        yield session
