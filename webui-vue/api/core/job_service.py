# -*- coding: utf-8 -*-
"""
Job Service Module

Provides CRUD operations and business logic for job management.
"""

import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .database import get_db
from .config import PROJECT_ROOT, CONFIGS_DIR, OUTPUT_BASE_DIR


class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert sqlite3.Row to dictionary."""
    if row is None:
        return None
    return dict(row)


def create_job(
    name: str,
    config_name: str,
    config_data: Dict[str, Any],
    output_dir: Optional[str] = None
) -> str:
    """
    Create a new job entry.
    
    Args:
        name: User-friendly job name (typically from output_name)
        config_name: Reference to the config file name
        config_data: Full configuration dictionary to snapshot
        output_dir: Output directory path
        
    Returns:
        job_id (UUID string)
    """
    job_id = str(uuid.uuid4())
    now = _now()
    
    # Determine output directory
    if output_dir is None:
        output_dir = str(OUTPUT_BASE_DIR)
    
    # Determine total epochs from config
    total_epochs = config_data.get("advanced", {}).get("num_train_epochs", 10)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (
                id, name, config_name, status, created_at, updated_at,
                total_epochs, output_dir, config_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id,
            name,
            config_name,
            JobStatus.PENDING,
            now,
            now,
            total_epochs,
            output_dir,
            json.dumps(config_data, ensure_ascii=False)
        ))
    
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single job by ID.
    
    Args:
        job_id: Job UUID
        
    Returns:
        Job dictionary or None if not found
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        job = _row_to_dict(row)
        # Parse config_snapshot JSON
        if job.get("config_snapshot"):
            try:
                job["config_snapshot"] = json.loads(job["config_snapshot"])
            except json.JSONDecodeError:
                pass
        
        return job


def list_jobs(
    status_filter: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    """
    List jobs with optional filtering and pagination.
    
    Args:
        status_filter: Filter by status (None = all)
        search: Search by name (partial match)
        limit: Max results to return
        offset: Pagination offset
        
    Returns:
        Tuple of (list of jobs, total count)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Build query
        where_clauses = []
        params = []
        
        if status_filter and status_filter != "all":
            where_clauses.append("status = ?")
            params.append(status_filter)
        
        if search:
            where_clauses.append("name LIKE ?")
            params.append(f"%{search}%")
        
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM jobs {where_sql}", params)
        total = cursor.fetchone()[0]
        
        # Get paginated results
        cursor.execute(f"""
            SELECT id, name, config_name, status, created_at, updated_at,
                   started_at, stopped_at, completed_at,
                   current_epoch, total_epochs, current_step, total_steps,
                   final_loss, output_dir, checkpoint_path, lora_path, error_message
            FROM jobs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])
        
        jobs = [_row_to_dict(row) for row in cursor.fetchall()]
        
        return jobs, total


def update_job_status(
    job_id: str,
    status: str,
    **kwargs
) -> bool:
    """
    Update job status and optional fields.
    
    Args:
        job_id: Job UUID
        status: New status
        **kwargs: Additional fields to update (error_message, checkpoint_path, lora_path, etc.)
        
    Returns:
        True if updated, False if job not found
    """
    now = _now()
    
    # Build update fields
    fields = ["status = ?", "updated_at = ?"]
    values = [status, now]
    
    # Handle timestamp fields based on status
    if status == JobStatus.RUNNING:
        fields.append("started_at = ?")
        values.append(now)
    elif status == JobStatus.STOPPED:
        fields.append("stopped_at = ?")
        values.append(now)
    elif status == JobStatus.COMPLETED:
        fields.append("completed_at = ?")
        values.append(now)
    elif status == JobStatus.FAILED:
        fields.append("stopped_at = ?")
        values.append(now)
    
    # Add any additional fields
    allowed_fields = {
        "error_message", "checkpoint_path", "lora_path", "final_loss",
        "current_epoch", "total_epochs", "current_step", "total_steps", "output_dir"
    }
    for key, value in kwargs.items():
        if key in allowed_fields:
            fields.append(f"{key} = ?")
            values.append(value)
    
    values.append(job_id)
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE jobs SET {', '.join(fields)} WHERE id = ?
        """, values)
        
        return cursor.rowcount > 0


def update_job_progress(
    job_id: str,
    current_epoch: int,
    current_step: int,
    total_steps: int,
    loss: Optional[float] = None,
    lr: Optional[float] = None
) -> bool:
    """
    Update job training progress.
    
    Args:
        job_id: Job UUID
        current_epoch: Current epoch number
        current_step: Current step number
        total_steps: Total steps
        loss: Current loss value
        lr: Current learning rate
        
    Returns:
        True if updated
    """
    now = _now()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        fields = [
            "current_epoch = ?",
            "current_step = ?",
            "total_steps = ?",
            "updated_at = ?"
        ]
        values = [current_epoch, current_step, total_steps, now]
        
        if loss is not None:
            fields.append("final_loss = ?")
            values.append(loss)
        
        values.append(job_id)
        
        cursor.execute(f"""
            UPDATE jobs SET {', '.join(fields)} WHERE id = ?
        """, values)
        
        # Also save to loss history if we have loss/lr
        if loss is not None or lr is not None:
            save_loss_history(job_id, current_step, loss, lr)
        
        return cursor.rowcount > 0


def save_job_log(job_id: str, level: str, message: str) -> None:
    """
    Save a log entry for a job.
    
    Args:
        job_id: Job UUID
        level: Log level (info, warning, error)
        message: Log message
    """
    now = _now()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO job_logs (job_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        """, (job_id, now, level, message))


def save_loss_history(
    job_id: str,
    step: int,
    loss: Optional[float],
    lr: Optional[float]
) -> None:
    """
    Save a loss/lr data point for charts.
    
    Args:
        job_id: Job UUID
        step: Training step
        loss: Loss value
        lr: Learning rate
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO job_loss_history (job_id, step, loss, lr)
            VALUES (?, ?, ?, ?)
        """, (job_id, step, loss, lr))


def get_job_logs(job_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Get logs for a job.
    
    Args:
        job_id: Job UUID
        limit: Max logs to return
        
    Returns:
        List of log entries (newest first)
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, level, message
            FROM job_logs
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (job_id, limit))
        
        return [_row_to_dict(row) for row in cursor.fetchall()]


def get_job_loss_history(job_id: str) -> Dict[str, List]:
    """
    Get loss history for charts.
    
    Args:
        job_id: Job UUID
        
    Returns:
        Dictionary with 'steps', 'loss', and 'lr' arrays
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT step, loss, lr
            FROM job_loss_history
            WHERE job_id = ?
            ORDER BY step ASC
        """, (job_id,))
        
        steps = []
        losses = []
        lrs = []
        
        for row in cursor.fetchall():
            steps.append(row["step"])
            losses.append(row["loss"])
            lrs.append(row["lr"])
        
        return {"steps": steps, "loss": losses, "lr": lrs}


def delete_job(job_id: str, delete_outputs: bool = False) -> bool:
    """
    Delete a job and optionally its output files.
    
    Args:
        job_id: Job UUID
        delete_outputs: If True, delete associated output files
        
    Returns:
        True if deleted
    """
    # First get job info for paths
    job = get_job(job_id)
    if job is None:
        return False
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Delete from all tables (cascades handle logs and history)
        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        deleted = cursor.rowcount > 0
    
    # Optionally delete output files
    if deleted and delete_outputs:
        # Delete checkpoint file
        if job.get("checkpoint_path"):
            checkpoint = Path(job["checkpoint_path"])
            if checkpoint.exists():
                checkpoint.unlink()
        
        # Delete LoRA file
        if job.get("lora_path"):
            lora = Path(job["lora_path"])
            if lora.exists():
                lora.unlink()
    
    return deleted


def duplicate_job(job_id: str, new_name: Optional[str] = None) -> Optional[str]:
    """
    Duplicate a job's configuration to create a new job.
    
    Args:
        job_id: Source job UUID
        new_name: Name for the new job (default: original_name_copy)
        
    Returns:
        New job_id or None if source not found
    """
    source_job = get_job(job_id)
    if source_job is None:
        return None
    
    # Determine new name
    if new_name is None:
        new_name = f"{source_job['name']}_copy"
    
    # Get config snapshot
    config_data = source_job.get("config_snapshot", {})
    if isinstance(config_data, str):
        try:
            config_data = json.loads(config_data)
        except json.JSONDecodeError:
            config_data = {}
    
    # Update output_name in config
    if "training" in config_data:
        config_data["training"]["output_name"] = new_name
    
    # Create new job
    return create_job(
        name=new_name,
        config_name=source_job.get("config_name", ""),
        config_data=config_data,
        output_dir=source_job.get("output_dir")
    )


def update_job_config(job_id: str, config_data: Dict[str, Any]) -> bool:
    """
    Update a job's configuration (only if pending or stopped).
    
    Args:
        job_id: Job UUID
        config_data: New configuration dictionary
        
    Returns:
        True if updated
    """
    job = get_job(job_id)
    if job is None:
        return False
    
    # Only allow editing if not running or completed
    if job["status"] in [JobStatus.RUNNING]:
        return False
    
    now = _now()
    
    # Update name from config if changed
    new_name = config_data.get("training", {}).get("output_name", job["name"])
    total_epochs = config_data.get("advanced", {}).get("num_train_epochs", job.get("total_epochs", 10))
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE jobs SET
                name = ?,
                total_epochs = ?,
                config_snapshot = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            new_name,
            total_epochs,
            json.dumps(config_data, ensure_ascii=False),
            now,
            job_id
        ))
        
        return cursor.rowcount > 0


def get_running_job() -> Optional[Dict[str, Any]]:
    """
    Get the currently running job if any.
    
    Returns:
        Running job or None
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM jobs WHERE status = ? LIMIT 1
        """, (JobStatus.RUNNING,))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        job = _row_to_dict(row)
        if job.get("config_snapshot"):
            try:
                job["config_snapshot"] = json.loads(job["config_snapshot"])
            except json.JSONDecodeError:
                pass
        
        return job


def clear_stale_running_jobs() -> int:
    """
    Mark any 'running' jobs as 'stopped' on startup (server restart recovery).
    
    Returns:
        Number of jobs marked as stopped
    """
    now = _now()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE jobs SET
                status = ?,
                stopped_at = ?,
                updated_at = ?,
                error_message = 'Server restarted'
            WHERE status = ?
        """, (JobStatus.STOPPED, now, now, JobStatus.RUNNING))
        
        return cursor.rowcount

