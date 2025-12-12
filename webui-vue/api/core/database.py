# -*- coding: utf-8 -*-
"""
SQLite Database Module for Job History

Provides database initialization and connection management for the job tracking system.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import threading

from .config import PROJECT_ROOT

# Database path
DB_PATH = PROJECT_ROOT / "jobs.db"

# Thread-local storage for connections
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row
        # Enable foreign keys
        _local.connection.execute("PRAGMA foreign_keys = ON")
    return _local.connection


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database operations with automatic commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Initialize the database schema."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_name TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                updated_at TEXT,
                started_at TEXT,
                stopped_at TEXT,
                completed_at TEXT,
                
                -- Training progress snapshot
                current_epoch INTEGER DEFAULT 0,
                total_epochs INTEGER,
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER,
                final_loss REAL,
                
                -- Paths
                output_dir TEXT,
                checkpoint_path TEXT,
                lora_path TEXT,
                
                -- Metadata
                config_snapshot TEXT,
                error_message TEXT
            )
        """)
        
        # Job logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                timestamp TEXT,
                level TEXT,
                message TEXT
            )
        """)
        
        # Job loss history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_loss_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
                step INTEGER,
                loss REAL,
                lr REAL
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_loss_history_job_id ON job_loss_history(job_id)
        """)
        
        print("[Database] Initialized jobs.db")


def close_db() -> None:
    """Close the thread-local database connection."""
    if hasattr(_local, 'connection') and _local.connection is not None:
        _local.connection.close()
        _local.connection = None


def reset_db() -> None:
    """Reset the database (drop all tables and recreate). USE WITH CAUTION."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS job_loss_history")
        cursor.execute("DROP TABLE IF EXISTS job_logs")
        cursor.execute("DROP TABLE IF EXISTS jobs")
    init_db()

