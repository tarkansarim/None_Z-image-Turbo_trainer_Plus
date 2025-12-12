# -*- coding: utf-8 -*-
"""
Jobs API Router

Provides REST endpoints for job management (list, create, update, delete, resume, etc.)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from core import job_service
from core.job_service import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ============================================================================
# Pydantic Models
# ============================================================================

class CreateJobRequest(BaseModel):
    name: str
    config_name: str
    config: Dict[str, Any]
    output_dir: Optional[str] = None


class UpdateJobConfigRequest(BaseModel):
    config: Dict[str, Any]


class DuplicateJobRequest(BaseModel):
    new_name: Optional[str] = None


class DeleteJobRequest(BaseModel):
    delete_outputs: bool = False


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    List all jobs with optional filtering and pagination.
    
    Status can be: all, pending, running, stopped, completed, failed
    """
    try:
        jobs, total = job_service.list_jobs(
            status_filter=status,
            search=search,
            limit=limit,
            offset=offset
        )
        
        return {
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/running")
async def get_running_job():
    """Get the currently running job, if any."""
    try:
        job = job_service.get_running_job()
        return {"job": job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get running job: {str(e)}")


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Get details of a specific job."""
    try:
        job = job_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.post("")
async def create_job(request: CreateJobRequest):
    """
    Create a new job from a configuration.
    
    This creates a job entry in pending status. Call /resume to start it.
    """
    try:
        job_id = job_service.create_job(
            name=request.name,
            config_name=request.config_name,
            config_data=request.config,
            output_dir=request.output_dir
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": f"Job '{request.name}' created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.put("/{job_id}")
async def update_job_config(job_id: str, request: UpdateJobConfigRequest):
    """
    Update a job's configuration.
    
    Only allowed for pending or stopped jobs.
    """
    try:
        job = job_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] == JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Cannot edit a running job")
        
        success = job_service.update_job_config(job_id, request.config)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update job config")
        
        return {
            "success": True,
            "message": "Job configuration updated"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update job: {str(e)}")


@router.delete("/{job_id}")
async def delete_job(job_id: str, delete_outputs: bool = Query(False)):
    """
    Delete a job.
    
    Set delete_outputs=true to also delete checkpoint and LoRA files.
    """
    try:
        job = job_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] == JobStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Cannot delete a running job. Stop it first.")
        
        success = job_service.delete_job(job_id, delete_outputs=delete_outputs)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete job")
        
        return {
            "success": True,
            "message": f"Job deleted" + (" with outputs" if delete_outputs else "")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.post("/{job_id}/duplicate")
async def duplicate_job(job_id: str, request: DuplicateJobRequest):
    """
    Duplicate a job to create a new one with the same configuration.
    """
    try:
        new_job_id = job_service.duplicate_job(job_id, request.new_name)
        if new_job_id is None:
            raise HTTPException(status_code=404, detail="Source job not found")
        
        return {
            "success": True,
            "job_id": new_job_id,
            "message": "Job duplicated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to duplicate job: {str(e)}")


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = Query(500, ge=1, le=2000)):
    """Get logs for a specific job."""
    try:
        job = job_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        logs = job_service.get_job_logs(job_id, limit=limit)
        return {"logs": logs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/{job_id}/history")
async def get_job_loss_history(job_id: str):
    """Get loss/lr history for charts."""
    try:
        job = job_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        history = job_service.get_job_loss_history(job_id)
        return history
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# Note: Resume and Stop endpoints are handled in training.py since they
# interact with the training process. The jobs router just provides
# job_id lookup for the training router to use.

