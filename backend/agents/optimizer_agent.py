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
AGENT_API_URL = "https://api.emergentagent.com/v1/agent/invoke"


def analyze_with_ai_agent(workload_details: dict, available_instances: list, current_instance: dict = None) -> dict:
    """
    Use AI agent to analyze workload and instances for migration decision
    
    Args:
        workload_details: Details about the ML workload (model, datasize, type, etc.)
        available_instances: List of available GPU instances with specs and costs
        current_instance: Current instance details if any (for migration decisions)
    
    Returns:
        dict with recommendation, selected_instance, reasoning, and should_migrate
    """
    try:
        # Prepare prompt for the AI agent
        prompt = f"""You are an expert GPU optimization agent. Analyze the following ML workload and available GPU instances to make the best decision.

WORKLOAD DETAILS:
- Model: {workload_details.get('model_name', 'N/A')}
- Data Size: {workload_details.get('datasize', 'N/A')}
- Workload Type: {workload_details.get('workload_type', 'N/A')}
- Duration: {workload_details.get('duration', 'N/A')}
- Budget: ${workload_details.get('budget', 'N/A')}
- Model Details: {workload_details.get('model_details', {})}

AVAILABLE INSTANCES:
"""
        for idx, instance in enumerate(available_instances, 1):
            prompt += f"\n{idx}. {instance['provider']} {instance['instance']}"
            prompt += f"\n   - GPU: {instance['gpu']}"
            prompt += f"\n   - Memory: {instance['memory']}"
            prompt += f"\n   - Cost: ${instance['cost_per_hour']}/hour"
        
        if current_instance:
            prompt += f"\n\nCURRENT INSTANCE:\n- {current_instance['provider']} {current_instance['instance']}"
            prompt += f"\n- GPU: {current_instance['gpu']}"
            prompt += f"\n- Cost: ${current_instance['cost_per_hour']}/hour"
            prompt += f"\n\nSHOULD WE MIGRATE? Consider cost savings, performance impact, and migration overhead."
        else:
            prompt += f"\n\nSELECT THE BEST INSTANCE for this workload considering:"
        
        prompt += """
1. Cost efficiency
2. GPU performance match for the model size and type
3. Memory requirements
4. Workload type (Training needs more compute than Inference)

Provide your analysis in this exact JSON format:
{
    "selected_instance_index": <number 1-6>,
    "should_migrate": <true/false>,
    "reasoning": "<brief explanation>",
    "cost_savings_percent": <number or null>,
    "confidence_score": <0-100>
}
"""
        
        # Call the AI agent
        payload = {
            "custom_agent_id": OPTIMIZER_AGENT_ID,
            "prompt": prompt,
            "stream": False
        }
        
        logger.info(f"Calling AI agent {OPTIMIZER_AGENT_ID} for optimization analysis")
        
        response = requests.post(
            AGENT_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            agent_response = result.get('response', '')
            
            logger.info(f"AI Agent Response: {agent_response[:200]}...")
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"selected_instance_index"[^{}]*\}', agent_response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Get the selected instance
                selected_idx = analysis.get('selected_instance_index', 1) - 1  # Convert to 0-based index
                if 0 <= selected_idx < len(available_instances):
                    selected_instance = available_instances[selected_idx]
                else:
                    selected_instance = available_instances[0]  # Default to first
                
                return {
                    'selected_instance': selected_instance,
                    'should_migrate': analysis.get('should_migrate', True),
                    'reasoning': analysis.get('reasoning', 'AI agent recommendation'),
                    'cost_savings_percent': analysis.get('cost_savings_percent'),
                    'confidence_score': analysis.get('confidence_score', 80),
                    'ai_powered': True
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning("Could not parse JSON from AI agent response, using fallback")
                return {
                    'selected_instance': min(available_instances, key=lambda x: x['cost_per_hour']),
                    'should_migrate': True,
                    'reasoning': 'Fallback to cheapest option (AI parsing failed)',
                    'cost_savings_percent': None,
                    'confidence_score': 50,
                    'ai_powered': False
                }
        else:
            logger.error(f"AI agent API call failed: {response.status_code}")
            return {
                'selected_instance': min(available_instances, key=lambda x: x['cost_per_hour']),
                'should_migrate': True,
                'reasoning': 'Fallback to cheapest option (API failed)',
                'cost_savings_percent': None,
                'confidence_score': 50,
                'ai_powered': False
            }
            
    except Exception as e:
        logger.error(f"Error calling AI agent: {str(e)}")
        # Fallback to simple cost-based selection
        return {
            'selected_instance': min(available_instances, key=lambda x: x['cost_per_hour']),
            'should_migrate': True,
            'reasoning': f'Fallback to cheapest option (Error: {str(e)})',
            'cost_savings_percent': None,
            'confidence_score': 50,
            'ai_powered': False
        }


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
    
    # Prepare workload details for AI agent
    workload_details = {
        'model_name': model_name,
        'datasize': job['datasize'],
        'workload_type': job['workload_type'],
        'duration': job['duration'],
        'budget': budget,
        'model_details': model_details
    }
    
    # Use AI agent to analyze and select best instance
    logger.info(f"Optimizer Agent: Calling AI agent for intelligent instance selection")
    ai_analysis = await asyncio.to_thread(
        analyze_with_ai_agent,
        workload_details,
        suitable_options,
        None  # No current instance for initial selection
    )
    
    best_option = ai_analysis['selected_instance']
    ai_reasoning = ai_analysis['reasoning']
    ai_confidence = ai_analysis['confidence_score']
    ai_powered = ai_analysis['ai_powered']
    
    logger.info(f"Optimizer Agent: AI selected {best_option['instance']} (Confidence: {ai_confidence}%)")
    logger.info(f"Optimizer Agent: Reasoning: {ai_reasoning}")
    
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
        "model_insights": model_details,
        "ai_analysis": {
            "agent_id": OPTIMIZER_AGENT_ID,
            "reasoning": ai_reasoning,
            "confidence_score": ai_confidence,
            "ai_powered": ai_powered
        }
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
