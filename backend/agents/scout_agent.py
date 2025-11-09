"""
Scout Agent - Searches for available GPU resources and monitors for better deals
"""
import asyncio
import logging
import random
import requests
from datetime import datetime, timezone
from database import db
from models import JobStatus

logger = logging.getLogger(__name__)

# Customer Agent IDs
OPTIMIZER_AGENT_ID = "d32e7ebe-bedf-4887-85cb-1c740e1e3831"  # For migration decisions
SCOUT_AGENT_ID = "88708788-ed0e-4306-a3eb-6de5878f3512"  # For spot instance pricing
AGENT_API_URL = "https://api.emergentagent.com/v1/agent/invoke"


def fetch_spot_instances_with_ai(model_name: str, datasize: str, workload_type: str, budget: float) -> list:
    """
    Use AI agent to fetch latest spot instance prices from AWS and GCP
    
    Args:
        model_name: ML model name
        datasize: Dataset size (e.g., "50GB", "2TB")
        workload_type: Type of workload (Training, Inference, etc.)
        budget: Budget in dollars
    
    Returns:
        list: List of 6 instances (3 AWS + 3 GCP) with current spot prices
    """
    try:
        prompt = f"""You are an expert at finding the best GPU spot instances for ML workloads. 

TASK: Find the latest generation GPU spot instances with CURRENT PRICING from AWS and GCP.

WORKLOAD REQUIREMENTS:
- Model: {model_name}
- Dataset Size: {datasize}
- Workload Type: {workload_type}
- Budget: ${budget}

INSTRUCTIONS:
1. Check AWS EC2 Spot Pricing: https://aws.amazon.com/ec2/spot/pricing/
2. Check GCP Spot VM GPU Pricing: https://cloud.google.com/spot-vms/pricing?hl=en#gpu-pricing
3. Focus on LATEST GENERATION instances (G5, G6, P4, P5 for AWS; A2, A3 for GCP)
4. Return 3 instances from AWS and 3 from GCP
5. Prioritize instances that match the workload requirements

For Training/Fine-tuning workloads: Look for high compute GPUs (A100, A10G, V100)
For Inference/Embeddings: Look for cost-effective GPUs (T4, A10G)

Return ONLY a JSON array with EXACTLY 6 instances (3 AWS + 3 GCP):
[
  {{
    "provider": "AWS",
    "instance": "g5.xlarge",
    "gpu": "1x A10G (24GB)",
    "memory": "16GB",
    "cost_per_hour": 0.526
  }},
  ...
]

IMPORTANT: Use ACTUAL CURRENT spot prices from the websites, not estimates.
"""
        
        payload = {
            "custom_agent_id": SCOUT_AGENT_ID,
            "prompt": prompt,
            "stream": False
        }
        
        logger.info(f"Calling Scout AI agent {SCOUT_AGENT_ID} to fetch spot instance prices")
        
        response = requests.post(
            AGENT_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for web scraping
        )
        
        if response.status_code == 200:
            result = response.json()
            agent_response = result.get('response', '')
            
            logger.info(f"Scout AI Agent Response: {agent_response[:300]}...")
            
            # Parse JSON array from response
            import json
            import re
            
            # Try to find JSON array
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', agent_response, re.DOTALL)
            if json_match:
                instances = json.loads(json_match.group())
                
                # Validate we have 6 instances
                if len(instances) >= 6:
                    logger.info(f"Scout AI: Successfully fetched {len(instances)} spot instances with current pricing")
                    
                    # Log some examples
                    for i, inst in enumerate(instances[:3]):
                        logger.info(f"  {i+1}. {inst['provider']} {inst['instance']} - ${inst['cost_per_hour']}/hr")
                    
                    return instances[:6]  # Return exactly 6
                else:
                    logger.warning(f"Scout AI returned only {len(instances)} instances, expected 6")
                    # Fall through to backup
            else:
                logger.warning("Could not parse JSON array from Scout AI response")
        else:
            logger.error(f"Scout AI API call failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error calling Scout AI agent: {str(e)}")
    
    # Fallback to default instances if AI fails
    logger.warning("Using fallback spot instance prices (AI fetch failed)")
    return get_fallback_instances(datasize, workload_type)


def get_fallback_instances(datasize: str, workload_type: str) -> list:
    """Fallback spot instance list if AI agent fails"""
    # Parse data size
    datasize_value = float(''.join(filter(str.isdigit, datasize)))
    datasize_unit = ''.join(filter(str.isalpha, datasize)).upper()
    datasize_gb = datasize_value
    if datasize_unit == 'TB':
        datasize_gb = datasize_value * 1024
    elif datasize_unit == 'MB':
        datasize_gb = datasize_value / 1024
    
    # Default instances based on workload
    if workload_type in ["Training", "Fine-tuning"]:
        if datasize_gb > 20:
            return [
                {"provider": "AWS", "instance": "g5.12xlarge", "gpu": "4x A10G (24GB)", "memory": "192GB", "cost_per_hour": 5.67},
                {"provider": "AWS", "instance": "g5.8xlarge", "gpu": "1x A10G (24GB)", "memory": "128GB", "cost_per_hour": 3.89},
                {"provider": "AWS", "instance": "g4dn.12xlarge", "gpu": "4x T4 (16GB)", "memory": "192GB", "cost_per_hour": 3.912},
                {"provider": "GCP", "instance": "a2-highgpu-4g", "gpu": "4x A100 (40GB)", "memory": "340GB", "cost_per_hour": 14.69},
                {"provider": "GCP", "instance": "a2-highgpu-2g", "gpu": "2x A100 (40GB)", "memory": "170GB", "cost_per_hour": 7.35},
                {"provider": "GCP", "instance": "n1-highmem-16-v100", "gpu": "4x V100 (16GB)", "memory": "104GB", "cost_per_hour": 9.76},
            ]
        else:
            return [
                {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006},
                {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526},
                {"provider": "AWS", "instance": "g5.2xlarge", "gpu": "1x A10G (24GB)", "memory": "32GB", "cost_per_hour": 1.212},
                {"provider": "GCP", "instance": "n1-standard-8-t4", "gpu": "1x T4 (16GB)", "memory": "30GB", "cost_per_hour": 0.71},
                {"provider": "GCP", "instance": "n1-standard-4-t4", "gpu": "1x T4 (16GB)", "memory": "15GB", "cost_per_hour": 0.58},
                {"provider": "GCP", "instance": "a2-highgpu-1g", "gpu": "1x A100 (40GB)", "memory": "85GB", "cost_per_hour": 3.67},
            ]
    else:
        return [
            {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006},
            {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526},
            {"provider": "AWS", "instance": "g5.2xlarge", "gpu": "1x A10G (24GB)", "memory": "32GB", "cost_per_hour": 1.212},
            {"provider": "GCP", "instance": "n1-standard-4-t4", "gpu": "1x T4 (16GB)", "memory": "15GB", "cost_per_hour": 0.58},
            {"provider": "GCP", "instance": "n1-standard-8-t4", "gpu": "1x T4 (16GB)", "memory": "30GB", "cost_per_hour": 0.71},
            {"provider": "GCP", "instance": "a2-highgpu-1g", "gpu": "1x A100 (40GB)", "memory": "85GB", "cost_per_hour": 3.67},
        ]


def validate_migration_with_ai(job: dict, new_instance: dict, current_instance: dict, cost_savings_percent: float) -> bool:
    """
    Use AI agent to validate if migration should proceed
    
    Returns:
        bool: True if migration should proceed, False otherwise
    """
    try:
        prompt = f"""You are an expert at deciding whether to migrate ML workloads between GPU instances. Analyze this migration decision:

CURRENT INSTANCE:
- Provider: {current_instance.get('provider', 'N/A')}
- Instance: {current_instance.get('instance', 'N/A')}
- GPU: {current_instance.get('gpu', 'N/A')}
- Cost: ${current_instance.get('cost_per_hour', 0)}/hour

NEW INSTANCE (PROPOSED):
- Provider: {new_instance.get('provider', 'N/A')}
- Instance: {new_instance.get('instance', 'N/A')}
- GPU: {new_instance.get('gpu', 'N/A')}
- Cost: ${new_instance.get('cost_per_hour', 0)}/hour

COST SAVINGS: {cost_savings_percent:.1f}%

WORKLOAD DETAILS:
- Model: {job.get('model_name', 'N/A')}
- Workload Type: {job.get('workload_type', 'N/A')}
- Current Status: {job.get('status', 'N/A')}

CONSIDERATIONS:
1. Migration overhead (checkpoint save/load, ~1-2 minutes downtime)
2. Risk of interrupting training progress
3. Potential performance difference between GPUs
4. Cost savings vs migration risk

Should we proceed with this migration? Respond with ONLY a JSON object:
{{
    "should_migrate": true/false,
    "reasoning": "brief explanation",
    "risk_level": "low/medium/high",
    "confidence": 0-100
}}
"""
        
        payload = {
            "custom_agent_id": OPTIMIZER_AGENT_ID,
            "prompt": prompt,
            "stream": False
        }
        
        logger.info(f"Asking AI agent whether to proceed with migration ({cost_savings_percent:.1f}% savings)")
        
        response = requests.post(
            AGENT_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            agent_response = result.get('response', '')
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{[^{}]*"should_migrate"[^{}]*\}', agent_response, re.DOTALL)
            
            if json_match:
                decision = json.loads(json_match.group())
                should_migrate = decision.get('should_migrate', True)
                reasoning = decision.get('reasoning', 'AI analysis')
                confidence = decision.get('confidence', 50)
                
                logger.info(f"AI Migration Decision: {'PROCEED' if should_migrate else 'SKIP'} (confidence: {confidence}%)")
                logger.info(f"AI Reasoning: {reasoning}")
                
                return should_migrate
            else:
                logger.warning("Could not parse AI response, defaulting to PROCEED")
                return True
        else:
            logger.error(f"AI agent API failed: {response.status_code}, defaulting to PROCEED")
            return True
            
    except Exception as e:
        logger.error(f"Error calling AI agent for migration validation: {str(e)}")
        # Default to proceeding with migration if AI fails
        return True


async def scout_agent(workload_id: str, model_name: str, datasize: str, workload_type: str, budget: float):
    """Scout Agent - Searches for available GPU resources from AWS and GCP"""
    from utils.supabase_utils import update_workload_in_supabase, save_optimization_plan_to_supabase
    from agents.optimizer_agent import optimizer_agent
    
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


async def continuous_scout_monitor(workload_id: str, model_name: str, datasize: str, workload_type: str, budget: float):
    """Continuous Scout Monitor - Runs every 2 minutes to find better instances"""
    from utils.supabase_utils import save_optimization_plan_to_supabase
    from agents.migration_agent import migration_with_checkpoint
    
    logger.info(f"Continuous Scout Monitor: Starting for workload {workload_id}")
    
    iteration = 0
    COST_SAVINGS_THRESHOLD = 0.20  # 20% savings required to trigger migration
    
    while True:
        try:
            # Wait 2 minutes before next scan
            await asyncio.sleep(120)  # 2 minutes
            
            iteration += 1
            logger.info(f"Continuous Scout Monitor: Iteration {iteration} for workload {workload_id}")
            
            # Check if workload is still running
            job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
            if not job or job['status'] not in [JobStatus.RUNNING, JobStatus.MIGRATING]:
                logger.info(f"Continuous Scout Monitor: Workload {workload_id} is no longer running, stopping monitor")
                break
            
            # Get current instance cost
            current_migration_details = job.get('migration_details', {})
            current_optimizer_results = job.get('optimizer_results', {})
            current_cost = current_optimizer_results.get('estimated_cost', 999999)
            
            logger.info(f"Continuous Scout Monitor: Current cost: ${current_cost}/hour")
            
            # Search for new options (simulate the scouting)
            await asyncio.sleep(1)
            
            # Generate new GPU options similar to scout_agent
            datasize_value = float(''.join(filter(str.isdigit, datasize)))
            datasize_unit = ''.join(filter(str.isalpha, datasize)).upper()
            datasize_gb = datasize_value
            if datasize_unit == 'TB':
                datasize_gb = datasize_value * 1024
            elif datasize_unit == 'MB':
                datasize_gb = datasize_value / 1024
            
            # Generate random cost variations to simulate market changes
            cost_multiplier = random.uniform(0.60, 0.95)  # Simulate 5-40% cost reduction
            
            aws_options = []
            if workload_type in ["Training", "Fine-tuning"]:
                if datasize_gb > 20:
                    aws_options = [
                        {"provider": "AWS", "instance": "g5.12xlarge", "gpu": "4x A10G (24GB)", "memory": "192GB", "cost_per_hour": 5.67 * cost_multiplier},
                        {"provider": "AWS", "instance": "g5.8xlarge", "gpu": "1x A10G (24GB)", "memory": "128GB", "cost_per_hour": 3.89 * cost_multiplier},
                        {"provider": "AWS", "instance": "g4dn.12xlarge", "gpu": "4x T4 (16GB)", "memory": "192GB", "cost_per_hour": 3.912 * cost_multiplier},
                    ]
                else:
                    aws_options = [
                        {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006 * cost_multiplier},
                        {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526 * cost_multiplier},
                        {"provider": "AWS", "instance": "g5.2xlarge", "gpu": "1x A10G (24GB)", "memory": "32GB", "cost_per_hour": 1.212 * cost_multiplier},
                    ]
            else:
                aws_options = [
                    {"provider": "AWS", "instance": "g4dn.xlarge", "gpu": "1x T4 (16GB)", "memory": "16GB", "cost_per_hour": 0.526 * cost_multiplier},
                    {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1x A10G (24GB)", "memory": "16GB", "cost_per_hour": 1.006 * cost_multiplier},
                    {"provider": "AWS", "instance": "g4dn.2xlarge", "gpu": "1x T4 (16GB)", "memory": "32GB", "cost_per_hour": 0.752 * cost_multiplier},
                ]
            
            # Find the cheapest option
            if aws_options:
                best_option = min(aws_options, key=lambda x: x['cost_per_hour'])
                new_cost = best_option['cost_per_hour']
                cost_savings = (current_cost - new_cost) / current_cost
                
                logger.info(f"Continuous Scout Monitor: Found option: {best_option['instance']} at ${new_cost:.3f}/hour (savings: {cost_savings*100:.1f}%)")
                
                # Check if savings meet threshold (20%)
                if cost_savings >= COST_SAVINGS_THRESHOLD:
                    logger.info(f"Continuous Scout Monitor: 🎯 Cost savings of {cost_savings*100:.1f}% meets threshold ({COST_SAVINGS_THRESHOLD*100}%)!")
                    
                    # Use AI agent to validate migration decision
                    should_migrate = await asyncio.to_thread(
                        validate_migration_with_ai,
                        job,
                        best_option,
                        current_optimizer_results.get('recommended_resource', {}),
                        cost_savings * 100
                    )
                    
                    if not should_migrate:
                        logger.info(f"Continuous Scout Monitor: AI agent recommended NOT to migrate despite cost savings")
                        continue
                    
                    logger.info(f"Continuous Scout Monitor: AI agent validated migration decision")
                    logger.info(f"Continuous Scout Monitor: Triggering migration from ${current_cost:.3f}/hr to ${new_cost:.3f}/hr")
                    
                    # Update status to Found Better Deal
                    await db.jobs.update_one(
                        {"workload_id": workload_id},
                        {"$set": {
                            "status": JobStatus.FOUND_BETTER_DEAL,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }}
                    )
                    
                    # Trigger migration to the better instance
                    await asyncio.sleep(2)  # Brief pause to show status
                    
                    # Create new optimizer results
                    new_optimizer_results = {
                        "selected_option": best_option,
                        "estimated_cost": new_cost,
                        "cost_savings_percent": cost_savings * 100,
                        "previous_cost": current_cost,
                        "optimization_version": current_optimizer_results.get('optimization_version', 0) + 1,
                        "trigger": "continuous_scout_monitor",
                        "iteration": iteration
                    }
                    
                    # Save new optimization plan to Supabase
                    plan_data = {
                        "workload_id": workload_id,
                        "target_resource": best_option,
                        "optimizer_results": new_optimizer_results,
                        "optimization_version": new_optimizer_results['optimization_version']
                    }
                    save_optimization_plan_to_supabase(workload_id, plan_data)
                    
                    # Store old instance for termination after migration
                    old_instance_id = current_migration_details.get('ec2_instance_id')
                    
                    # Trigger migration with old instance info for cleanup
                    asyncio.create_task(migration_with_checkpoint(
                        workload_id, 
                        best_option, 
                        new_optimizer_results,
                        old_instance_id
                    ))
                    
                    # Don't break - continue monitoring for even better deals
                else:
                    logger.info(f"Continuous Scout Monitor: Cost savings of {cost_savings*100:.1f}% below threshold ({COST_SAVINGS_THRESHOLD*100}%), continuing to monitor")
            
        except Exception as e:
            logger.error(f"Continuous Scout Monitor: Error in iteration {iteration} - {str(e)}")
            await asyncio.sleep(120)  # Continue monitoring even after errors
