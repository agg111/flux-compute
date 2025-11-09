"""
API Routes for ML Workload Optimization
"""
import asyncio
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from models import Job, JobCreate, JobUpdate, StatusCheck, StatusCheckCreate
from database import db
from utils.supabase_utils import (
    save_workload_to_supabase,
    save_optimization_plan_to_supabase,
    get_migrations_for_workload
)
import logging

logger = logging.getLogger(__name__)

# Create API router
api_router = APIRouter(prefix="/api")


@api_router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ML Workload Optimization API", "status": "running"}


@api_router.post("/status-check", response_model=StatusCheck)
async def create_status_check(status_check: StatusCheckCreate):
    """Create a new status check"""
    status_check_obj = StatusCheck(
        client_name=status_check.client_name,
        timestamp=datetime.now(timezone.utc)
    )
    
    status_check_dict = status_check_obj.model_dump()
    status_check_dict['timestamp'] = status_check_obj.timestamp.isoformat()
    
    await db.status_checks.insert_one(status_check_dict)
    return status_check_obj


@api_router.post("/jobs", response_model=Job)
async def create_job(job: JobCreate):
    """Create a new ML workload job"""
    # Import here to avoid circular imports
    from agents.scout_agent import scout_agent
    
    job_obj = Job(
        model_name=job.model_name,
        datasize=job.datasize,
        workload_type=job.workload_type,
        duration=job.duration,
        budget=job.budget,
        precision=job.precision,
        framework=job.framework
    )
    
    # Convert to dict for MongoDB
    job_dict = job_obj.model_dump()
    job_dict['created_at'] = job_obj.created_at.isoformat()
    job_dict['updated_at'] = job_obj.updated_at.isoformat()
    
    # Insert into MongoDB
    await db.jobs.insert_one(job_dict)
    
    # Save to Supabase
    workload_json = {
        'model_name': job_obj.model_name,
        'datasize': job_obj.datasize,
        'workload_type': job_obj.workload_type,
        'duration': job_obj.duration,
        'budget': job_obj.budget,
        'precision': job_obj.precision,
        'framework': job_obj.framework
    }
    
    # Save optimization plan
    plan_data = {
        "workload_id": job_obj.workload_id,
        "initial_budget": job_obj.budget,
        "optimization_version": 1
    }
    plan_id = save_optimization_plan_to_supabase(job_obj.workload_id, plan_data)
    save_workload_to_supabase(job_obj.workload_id, workload_json, "PENDING", plan_id)
    
    # Trigger scout agent in background
    asyncio.create_task(scout_agent(
        job_obj.workload_id,
        job_obj.model_name,
        job_obj.datasize,
        job_obj.workload_type,
        job_obj.budget
    ))
    
    return job_obj


@api_router.get("/jobs")
async def get_jobs():
    """Get all jobs"""
    jobs = []
    async for job in db.jobs.find({}, {"_id": 0}).sort("created_at", -1):
        # Convert ISO string timestamps back to datetime objects
        if isinstance(job['created_at'], str):
            job['created_at'] = datetime.fromisoformat(job['created_at'])
        if isinstance(job['updated_at'], str):
            job['updated_at'] = datetime.fromisoformat(job['updated_at'])
        jobs.append(job)
    
    return jobs


@api_router.get("/jobs/{workload_id}", response_model=Job)
async def get_job(workload_id: str):
    """Get a specific job by workload_id"""
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Convert ISO string timestamps back to datetime objects
    if isinstance(job['created_at'], str):
        job['created_at'] = datetime.fromisoformat(job['created_at'])
    if isinstance(job['updated_at'], str):
        job['updated_at'] = datetime.fromisoformat(job['updated_at'])
    
    return job


@api_router.get("/jobs/{workload_id}/migrations")
async def get_job_migrations(workload_id: str):
    """Get migration history for a specific job"""
    # Verify job exists
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get migrations from Supabase
    migrations = get_migrations_for_workload(workload_id)
    
    return {
        "workload_id": workload_id,
        "migration_count": len(migrations),
        "migrations": migrations
    }


@api_router.patch("/jobs/{workload_id}", response_model=Job)
async def update_job(workload_id: str, job_update: JobUpdate):
    """Update job status and details"""
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Prepare update data
    update_data = job_update.model_dump(exclude_unset=True)
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Update in database
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {"$set": update_data}
    )
    
    # Fetch updated job
    updated_job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    
    # Convert ISO string timestamps back to datetime objects
    if isinstance(updated_job['created_at'], str):
        updated_job['created_at'] = datetime.fromisoformat(updated_job['created_at'])
    if isinstance(updated_job['updated_at'], str):
        updated_job['updated_at'] = datetime.fromisoformat(updated_job['updated_at'])
    
    return updated_job


@api_router.delete("/jobs/{workload_id}")
async def delete_job(workload_id: str):
    """Delete a job"""
    result = await db.jobs.delete_one({"workload_id": workload_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job deleted successfully", "workload_id": workload_id}
