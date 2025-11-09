from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from enum import Enum
import asyncio
import random
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Supabase connection
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Enums
class WorkloadType(str, Enum):
    INFERENCE = "Inference"
    FINETUNING = "Fine-tuning"
    EMBEDDINGS = "Embeddings Generation"
    TRAINING = "Training"

class JobStatus(str, Enum):
    PENDING = "Pending"
    SCOUTING = "Scouting"
    ANALYZING = "Analyzing"
    FOUND_BETTER_DEAL = "Found Better Deal"
    MIGRATING = "Migrating"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class PrecisionType(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    MIXED = "Mixed Precision"

class FrameworkType(str, Enum):
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    JAX = "JAX"
    ONNX = "ONNX"


# Job Models
class JobCreate(BaseModel):
    model_name: str
    datasize: str
    workload_type: WorkloadType
    duration: str
    budget: float
    precision: Optional[PrecisionType] = None
    framework: Optional[FrameworkType] = None

class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    workload_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    datasize: str
    workload_type: WorkloadType
    duration: str
    budget: float
    precision: Optional[PrecisionType] = None
    framework: Optional[FrameworkType] = None
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_cost: Optional[float] = None
    recommended_gpu: Optional[str] = None
    recommended_memory: Optional[str] = None
    scout_results: Optional[dict] = None
    optimizer_results: Optional[dict] = None

class JobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    estimated_cost: Optional[float] = None
    recommended_gpu: Optional[str] = None
    recommended_memory: Optional[str] = None


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str


# Utility Functions
def fetch_huggingface_model_details(model_name: str) -> dict:
    """Fetch model details from Hugging Face"""
    try:
        # Try to find the model on Hugging Face
        search_url = f"https://huggingface.co/api/models?search={model_name}"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            if models and len(models) > 0:
                # Get the first matching model
                model_id = models[0].get('id', model_name)
                
                # Fetch detailed model info
                model_url = f"https://huggingface.co/{model_id}"
                model_response = requests.get(model_url, timeout=10)
                
                if model_response.status_code == 200:
                    soup = BeautifulSoup(model_response.text, 'html.parser')
                    
                    # Extract model card information
                    model_details = {
                        'model_id': model_id,
                        'model_name': model_name,
                        'found_on_hf': True,
                        'url': model_url
                    }
                    
                    # Try to extract parameters from the page
                    # Look for common patterns like "7B", "70B", "13B" parameters
                    text_content = soup.get_text()
                    
                    # Extract parameter count
                    import re
                    param_patterns = [
                        r'(\d+\.?\d*)\s*[Bb]illion',
                        r'(\d+\.?\d*)\s*B\s+parameters',
                        r'(\d+\.?\d*)\s*B\s+param',
                    ]
                    
                    for pattern in param_patterns:
                        match = re.search(pattern, text_content)
                        if match:
                            model_details['parameters'] = f"{match.group(1)}B"
                            break
                    
                    # Extract architecture info
                    if 'transformer' in text_content.lower():
                        model_details['architecture'] = 'Transformer'
                    elif 'diffusion' in text_content.lower():
                        model_details['architecture'] = 'Diffusion'
                    elif 'bert' in model_name.lower():
                        model_details['architecture'] = 'BERT'
                    elif 'gpt' in model_name.lower():
                        model_details['architecture'] = 'GPT'
                    elif 'llama' in model_name.lower():
                        model_details['architecture'] = 'LLaMA'
                    
                    # Extract model size/memory requirements from name
                    size_match = re.search(r'(\d+)[bB]', model_name)
                    if size_match:
                        model_details['size_billions'] = int(size_match.group(1))
                    
                    return model_details
        
        # If not found on HF, return basic info
        return {
            'model_name': model_name,
            'found_on_hf': False,
            'estimated_size': 'unknown'
        }
        
    except Exception as e:
        logger.error(f"Error fetching HuggingFace model details: {str(e)}")
        return {
            'model_name': model_name,
            'found_on_hf': False,
            'error': str(e)
        }


def save_optimization_plan_to_supabase(workload_id: str, plan_data: dict):
    """Save optimization plan to Supabase and return the plan ID"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return None
        
        data = {
            'workload_id': workload_id,
            'plan_data': plan_data,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Upsert the plan
        result = supabase.table('optimization_plans').upsert(data).execute()
        
        if result.data and len(result.data) > 0:
            plan_id = result.data[0].get('id')
            logger.info(f"Saved optimization plan to Supabase for workload {workload_id}, plan_id: {plan_id}")
            return plan_id
        
        return None
        
    except Exception as e:
        logger.error(f"Error saving to Supabase: {str(e)}")
        return None


def get_optimization_plan_from_supabase(workload_id: str) -> dict:
    """Get optimization plan from Supabase"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return None
        
        result = supabase.table('optimization_plans').select('*').eq('workload_id', workload_id).execute()
        
        if result.data and len(result.data) > 0:
            logger.info(f"Retrieved optimization plan from Supabase for workload {workload_id}")
            return result.data[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Error retrieving from Supabase: {str(e)}")
        return None


def save_workload_to_supabase(workload_id: str, workload_data: dict, status: str = "PENDING", plan_id: int = None):
    """Save workload to Supabase with simplified structure"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return
        
        data = {
            'workload_id': workload_id,
            'workload_data': workload_data,
            'status': status,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        if plan_id:
            data['plan_id'] = plan_id
        
        # Upsert the workload
        result = supabase.table('workloads').upsert(data).execute()
        logger.info(f"Saved workload to Supabase: {workload_id}, plan_id: {plan_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error saving workload to Supabase: {str(e)}")
        return None


def update_workload_in_supabase(workload_id: str, status: str = None, workload_data: dict = None):
    """Update workload in Supabase"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return
        
        updates = {'updated_at': datetime.now(timezone.utc).isoformat()}
        
        if status:
            updates['status'] = status
        
        if workload_data:
            updates['workload_data'] = workload_data
        
        result = supabase.table('workloads').update(updates).eq('workload_id', workload_id).execute()
        logger.info(f"Updated workload in Supabase: {workload_id} - status: {status}")
        return result
        
    except Exception as e:
        logger.error(f"Error updating workload in Supabase: {str(e)}")
        return None


# Agent Functions
async def scout_agent(workload_id: str, model_name: str, datasize: str, workload_type: str, budget: float):
    """Scout Agent - Searches for available GPU resources from AWS and GCP"""
    logger.info(f"Scout Agent: Starting resource search for workload {workload_id}")
    
    # Update status to Scouting
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {"$set": {"status": JobStatus.SCOUTING, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Update Supabase (Scouting maps to RUNNING)
    update_workload_in_supabase(workload_id, status="RUNNING")
    
    # Simulate scout agent searching cloud providers
    await asyncio.sleep(2)
    
    # Parse data size to estimate GPU requirements
    datasize_value = float(''.join(filter(str.isdigit, datasize)))
    datasize_unit = ''.join(filter(str.isalpha, datasize)).upper()
    
    # Convert to GB for comparison
    datasize_gb = datasize_value
    if datasize_unit == 'TB':
        datasize_gb = datasize_value * 1024
    elif datasize_unit == 'MB':
        datasize_gb = datasize_value / 1024
    
    # Determine GPU requirements based on workload
    # Always return 3 options from AWS and 3 from GCP
    aws_options = []
    gcp_options = []
    
    if workload_type in ["Training", "Fine-tuning"]:
        if datasize_gb > 100 or "70b" in model_name.lower() or "65b" in model_name.lower():
            # Large model training
            aws_options = [
                {"provider": "AWS", "instance": "p4d.24xlarge", "gpu": "8x A100 (40GB)", "memory": "320GB", "cost_per_hour": 32.77},
                {"provider": "AWS", "instance": "p4de.24xlarge", "gpu": "8x A100 (80GB)", "memory": "1152GB", "cost_per_hour": 40.96},
                {"provider": "AWS", "instance": "p3.16xlarge", "gpu": "8x V100 (16GB)", "memory": "488GB", "cost_per_hour": 24.48},
            ]
            gcp_options = [
                {"provider": "GCP", "instance": "a2-highgpu-8g", "gpu": "8x A100 (40GB)", "memory": "680GB", "cost_per_hour": 29.39},
                {"provider": "GCP", "instance": "a2-ultragpu-8g", "gpu": "8x A100 (80GB)", "memory": "680GB", "cost_per_hour": 35.73},
                {"provider": "GCP", "instance": "a2-highgpu-4g", "gpu": "4x A100 (40GB)", "memory": "340GB", "cost_per_hour": 14.69},
            ]
        elif datasize_gb > 20 or "7b" in model_name.lower() or "13b" in model_name.lower():
            # Medium model training
            aws_options = [
                {"provider": "AWS", "instance": "g5.12xlarge", "gpu": "4x A10G (24GB)", "memory": "192GB", "cost_per_hour": 5.67},
                {"provider": "AWS", "instance": "p3.8xlarge", "gpu": "4x V100 (16GB)", "memory": "244GB", "cost_per_hour": 12.24},
                {"provider": "AWS", "instance": "g5.24xlarge", "gpu": "4x A10G (24GB)", "memory": "384GB", "cost_per_hour": 8.14},
            ]
            gcp_options = [
                {"provider": "GCP", "instance": "a2-highgpu-4g", "gpu": "4x A100 (40GB)", "memory": "340GB", "cost_per_hour": 14.69},
                {"provider": "GCP", "instance": "a2-highgpu-2g", "gpu": "2x A100 (40GB)", "memory": "170GB", "cost_per_hour": 7.35},
                {"provider": "GCP", "instance": "n1-highmem-16-v100", "gpu": "4x V100 (16GB)", "memory": "104GB", "cost_per_hour": 9.76},
            ]
        else:
            # Small model training
            aws_options = [
                {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006},
                {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526},
                {"provider": "AWS", "instance": "g5.2xlarge", "gpu": "1x A10G (24GB)", "memory": "32GB", "cost_per_hour": 1.212},
            ]
            gcp_options = [
                {"provider": "GCP", "instance": "n1-standard-8-t4", "gpu": "1x T4 (16GB)", "memory": "30GB", "cost_per_hour": 0.71},
                {"provider": "GCP", "instance": "n1-standard-4-t4", "gpu": "1x T4 (16GB)", "memory": "15GB", "cost_per_hour": 0.58},
                {"provider": "GCP", "instance": "a2-highgpu-1g", "gpu": "1x A100 (40GB)", "memory": "85GB", "cost_per_hour": 3.67},
            ]
    else:  # Inference or Embeddings
        if "70b" in model_name.lower() or "65b" in model_name.lower():
            # Large model inference
            aws_options = [
                {"provider": "AWS", "instance": "g5.12xlarge", "gpu": "4x A10G (24GB)", "memory": "192GB", "cost_per_hour": 5.67},
                {"provider": "AWS", "instance": "g5.48xlarge", "gpu": "8x A10G (24GB)", "memory": "768GB", "cost_per_hour": 16.29},
                {"provider": "AWS", "instance": "p3.8xlarge", "gpu": "4x V100 (16GB)", "memory": "244GB", "cost_per_hour": 12.24},
            ]
            gcp_options = [
                {"provider": "GCP", "instance": "a2-highgpu-2g", "gpu": "2x A100 (40GB)", "memory": "170GB", "cost_per_hour": 7.35},
                {"provider": "GCP", "instance": "a2-highgpu-4g", "gpu": "4x A100 (40GB)", "memory": "340GB", "cost_per_hour": 14.69},
                {"provider": "GCP", "instance": "a2-highgpu-1g", "gpu": "1x A100 (40GB)", "memory": "85GB", "cost_per_hour": 3.67},
            ]
        else:
            # Small/medium model inference
            aws_options = [
                {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006},
                {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526},
                {"provider": "AWS", "instance": "g5.2xlarge", "gpu": "1x A10G (24GB)", "memory": "32GB", "cost_per_hour": 1.212},
            ]
            gcp_options = [
                {"provider": "GCP", "instance": "n1-standard-4-t4", "gpu": "1x T4 (16GB)", "memory": "15GB", "cost_per_hour": 0.58},
                {"provider": "GCP", "instance": "n1-standard-8-t4", "gpu": "1x T4 (16GB)", "memory": "30GB", "cost_per_hour": 0.71},
                {"provider": "GCP", "instance": "a2-highgpu-1g", "gpu": "1x A100 (40GB)", "memory": "85GB", "cost_per_hour": 3.67},
            ]
    
    # Combine AWS and GCP options (3 from each)
    gpu_options = aws_options + gcp_options
    
    scout_results = {
        "available_resources": gpu_options,
        "search_timestamp": datetime.now(timezone.utc).isoformat(),
        "providers_searched": ["AWS", "GCP"]
    }
    
    # Update job with scout results
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {
            "$set": {
                "scout_results": scout_results,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        }
    )
    
    # Update Supabase with scout results (add to workload_data)
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    workload_json = {
        'model_name': job['model_name'],
        'datasize': job['datasize'],
        'workload_type': job['workload_type'],
        'duration': job['duration'],
        'budget': job['budget'],
        'precision': job.get('precision'),
        'framework': job.get('framework'),
        'scout_results': scout_results
    }
    update_workload_in_supabase(workload_id, workload_data=workload_json)
    
    logger.info(f"Scout Agent: Found {len(gpu_options)} GPU options for workload {workload_id}")
    
    # Trigger optimizer agent
    asyncio.create_task(optimizer_agent(workload_id, scout_results, budget))


async def optimizer_agent(workload_id: str, scout_results: dict, budget: float):
    """Optimizer Agent - Selects the best GPU resource based on cost and performance"""
    logger.info(f"Optimizer Agent: Starting optimization for workload {workload_id}")
    
    # Update status to Analyzing
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {"$set": {"status": JobStatus.ANALYZING, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    # Analyzing still maps to RUNNING in Supabase
    update_workload_in_supabase(workload_id, status="RUNNING")
    
    # Get current plan from Supabase
    current_plan = get_optimization_plan_from_supabase(workload_id)
    logger.info(f"Optimizer Agent: Retrieved current plan from Supabase")
    
    # Get job details to fetch model information
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    model_name = job.get('model_name', '')
    
    # Fetch model details from Hugging Face
    logger.info(f"Optimizer Agent: Fetching model details from HuggingFace for {model_name}")
    model_details = fetch_huggingface_model_details(model_name)
    logger.info(f"Optimizer Agent: Model details - {model_details}")
    
    # Simulate optimizer analysis with model insights
    await asyncio.sleep(2)
    
    available_resources = scout_results.get("available_resources", [])
    
    # Filter options within budget (assuming duration for cost calculation)
    suitable_options = [
        opt for opt in available_resources 
        if opt["cost_per_hour"] * 2 <= budget  # Assume 2 hours minimum
    ]
    
    if not suitable_options:
        suitable_options = available_resources  # Use all if none fit budget
    
    # Use model details to make smarter selection
    best_option = None
    
    # Check if model is large (70B+) and needs high-end GPUs
    if model_details.get('size_billions', 0) >= 70 or '70b' in model_name.lower() or '65b' in model_name.lower():
        # Prefer A100 GPUs for large models
        a100_options = [opt for opt in suitable_options if 'A100' in opt['gpu']]
        if a100_options:
            best_option = min(a100_options, key=lambda x: x["cost_per_hour"])
            logger.info(f"Optimizer Agent: Selected A100 for large model ({model_name})")
    
    # Check if it's a diffusion model
    elif model_details.get('architecture') == 'Diffusion' or 'diffusion' in model_name.lower():
        # Diffusion models benefit from A10G or T4
        preferred_options = [opt for opt in suitable_options if 'A10G' in opt['gpu'] or 'T4' in opt['gpu']]
        if preferred_options:
            best_option = min(preferred_options, key=lambda x: x["cost_per_hour"])
            logger.info(f"Optimizer Agent: Selected GPU optimized for diffusion model")
    
    # Default: Select cheapest suitable option
    if not best_option:
        best_option = min(suitable_options, key=lambda x: x["cost_per_hour"])
    
    # Calculate estimated cost and optimization percentage
    estimated_cost = best_option["cost_per_hour"] * 2  # Estimate 2 hours
    
    # Calculate optimization percentage (savings vs most expensive option)
    max_cost = max([opt["cost_per_hour"] for opt in suitable_options])
    optimization_percentage = round(((max_cost - best_option["cost_per_hour"]) / max_cost) * 100, 1) if len(suitable_options) > 1 and max_cost > 0 else 0
    
    optimizer_results = {
        "recommended_resource": best_option,
        "estimated_cost": round(estimated_cost, 2),
        "alternatives": [opt for opt in suitable_options if opt != best_option][:2],
        "optimization_timestamp": datetime.now(timezone.utc).isoformat(),
        "savings": round(max([opt["cost_per_hour"] for opt in suitable_options]) - best_option["cost_per_hour"], 2) if len(suitable_options) > 1 else 0,
        "optimization_percentage": optimization_percentage,
        "model_insights": model_details
    }
    
    # Save improved plan to Supabase
    improved_plan = {
        'workload_id': workload_id,
        'model_name': model_name,
        'model_details': model_details,
        'scout_options_count': len(available_resources),
        'selected_resource': best_option,
        'optimization_percentage': optimization_percentage,
        'estimated_cost': estimated_cost,
        'status': 'optimized',
        'optimization_version': (current_plan.get('plan_data', {}).get('optimization_version', 1) + 1) if current_plan else 2
    }
    save_optimization_plan_to_supabase(workload_id, improved_plan)
    logger.info(f"Optimizer Agent: Saved improved plan to Supabase (v{improved_plan['optimization_version']})")
    
    # Update status to Found Better Deal first
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {
            "$set": {
                "optimizer_results": optimizer_results,
                "status": JobStatus.FOUND_BETTER_DEAL,
                "recommended_gpu": best_option["gpu"],
                "recommended_memory": best_option["memory"],
                "estimated_cost": estimated_cost,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        }
    )
    
    # Update Supabase with full workload data including optimizer results
    job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
    workload_json = {
        'model_name': job['model_name'],
        'datasize': job['datasize'],
        'workload_type': job['workload_type'],
        'duration': job['duration'],
        'budget': job['budget'],
        'precision': job.get('precision'),
        'framework': job.get('framework'),
        'scout_results': job.get('scout_results'),
        'optimizer_results': optimizer_results,
        'recommended_gpu': best_option["gpu"],
        'recommended_memory': best_option["memory"],
        'estimated_cost': float(estimated_cost)
    }
    update_workload_in_supabase(workload_id, status="RUNNING", workload_data=workload_json)
    
    logger.info(f"Optimizer Agent: Selected {best_option['instance']} for workload {workload_id}")
    
    # Simulate migration phase
    await asyncio.sleep(1)
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {"$set": {"status": JobStatus.MIGRATING, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    # Finally set to Running
    await asyncio.sleep(1)
    await db.jobs.update_one(
        {"workload_id": workload_id},
        {"$set": {"status": JobStatus.RUNNING, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    # Job is complete from optimization perspective - set to COMPLETE in Supabase
    update_workload_in_supabase(workload_id, status="COMPLETE")


# Routes
@api_router.get("/")
async def root():
    return {"message": "ML Job Submission Platform API"}


# Job Management Routes
@api_router.post("/jobs", response_model=Job)
async def create_job(job_input: JobCreate):
    """Submit a new ML job"""
    job_dict = job_input.model_dump()
    job_obj = Job(**job_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = job_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    await db.jobs.insert_one(doc)
    
    # Save workload to Supabase
    workload_json = {
        'model_name': job_obj.model_name,
        'datasize': job_obj.datasize,
        'workload_type': job_obj.workload_type,
        'duration': job_obj.duration,
        'budget': float(job_obj.budget),
        'precision': job_obj.precision,
        'framework': job_obj.framework
    }
    save_workload_to_supabase(job_obj.workload_id, workload_json, "PENDING")
    
    # Save initial plan to Supabase
    initial_plan = {
        'workload_id': job_obj.workload_id,
        'model_name': job_obj.model_name,
        'datasize': job_obj.datasize,
        'workload_type': job_obj.workload_type,
        'duration': job_obj.duration,
        'budget': job_obj.budget,
        'precision': job_obj.precision,
        'framework': job_obj.framework,
        'status': 'initial',
        'optimization_version': 1
    }
    save_optimization_plan_to_supabase(job_obj.workload_id, initial_plan)
    
    # Trigger scout agent in background
    asyncio.create_task(scout_agent(
        job_obj.workload_id,
        job_obj.model_name,
        job_obj.datasize,
        job_obj.workload_type,
        job_obj.budget
    ))
    
    return job_obj


@api_router.get("/jobs", response_model=List[Job])
async def get_jobs():
    """Get all jobs"""
    jobs = await db.jobs.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for job in jobs:
        if isinstance(job['created_at'], str):
            job['created_at'] = datetime.fromisoformat(job['created_at'])
        if isinstance(job['updated_at'], str):
            job['updated_at'] = datetime.fromisoformat(job['updated_at'])
    
    # Sort by created_at descending (newest first)
    jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
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
    """Cancel/delete a job"""
    result = await db.jobs.delete_one({"workload_id": workload_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": "Job deleted successfully"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()