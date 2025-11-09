"""
Optimizer Agent - Selects the best GPU resource based on cost and performance
Uses AI agent for intelligent migration decisions
"""
import asyncio
import logging
import requests
from datetime import datetime, timezone
from database import db
from models import JobStatus
from utils.supabase_utils import (
    update_workload_in_supabase, 
    get_optimization_plan_from_supabase,
    save_optimization_plan_to_supabase
)
from utils.helpers import fetch_huggingface_model_details

logger = logging.getLogger(__name__)

# Customer Agent ID for optimization decisions
OPTIMIZER_AGENT_ID = "d32e7ebe-bedf-4887-85cb-1c740e1e3831"


async def optimizer_agent(workload_id: str, scout_results: dict, budget: float):
    """Optimizer Agent - Selects the best GPU resource based on cost and performance"""
    from agents.migration_agent import migration_agent
    
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
    new_plan_id = save_optimization_plan_to_supabase(workload_id, improved_plan)
    logger.info(f"Optimizer Agent: Saved improved plan to Supabase (v{improved_plan['optimization_version']}, plan_id: {new_plan_id})")
    
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
    update_workload_in_supabase(workload_id, status="RUNNING", workload_data=workload_json, plan_id=new_plan_id)
    
    logger.info(f"Optimizer Agent: Selected {best_option['instance']} for workload {workload_id}")
    
    # Trigger Migration Agent
    logger.info(f"Optimizer Agent: Triggering Migration Agent for workload {workload_id}")
    asyncio.create_task(migration_agent(workload_id, best_option, optimizer_results))
