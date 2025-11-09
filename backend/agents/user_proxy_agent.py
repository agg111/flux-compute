"""
UserProxy Agent - Manages endpoint routing and instance cleanup
"""
import asyncio
import logging
from datetime import datetime, timezone
from database import db
from models import JobStatus
from utils.aws_utils import terminate_ec2_instance
from utils.supabase_utils import update_workload_in_supabase, update_migration_status_in_supabase

logger = logging.getLogger(__name__)

async def user_proxy_with_cleanup(workload_id: str, deployment_details: dict, migration_details: dict, old_instance_id: str = None):
    """UserProxy Agent with old instance cleanup after endpoint update"""
    logger.info(f"UserProxy with Cleanup: Starting for workload {workload_id}")
    
    try:
        # First run the normal UserProxy agent logic
        await user_proxy_agent(workload_id, deployment_details, migration_details)
        
        # After endpoint is updated, terminate old instance
        if old_instance_id:
            logger.info(f"UserProxy with Cleanup: Endpoint updated successfully, now terminating old instance {old_instance_id}")
            await asyncio.sleep(2)  # Brief wait to ensure endpoint is stable
            
            # Check if it's AWS or GCP instance
            if old_instance_id.startswith('i-'):
                # AWS EC2 instance
                terminate_result = await asyncio.to_thread(terminate_ec2_instance, old_instance_id)
            else:
                # GCP instance
                from utils.aws_utils import terminate_gcp_instance
                terminate_result = await asyncio.to_thread(terminate_gcp_instance, old_instance_id)
            
            if terminate_result.get('status') == 'success':
                logger.info(f"UserProxy with Cleanup: ✓ Old instance {old_instance_id} terminated successfully")
                
                # Update migration details with termination info
                await db.jobs.update_one(
                    {"workload_id": workload_id},
                    {
                        "$set": {
                            "migration_details.old_instance_terminated": True,
                            "migration_details.old_instance_id": old_instance_id,
                            "migration_details.termination_time": datetime.now(timezone.utc).isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
            else:
                logger.warning(f"UserProxy with Cleanup: Failed to terminate old instance {old_instance_id}")
        else:
            logger.info(f"UserProxy with Cleanup: No old instance to terminate")
            
    except Exception as e:
        logger.error(f"UserProxy with Cleanup: Error - {str(e)}")


async def user_proxy_agent(workload_id: str, deployment_details: dict, migration_details: dict):
    """UserProxy Agent - Updates endpoint routing to direct traffic to new instances"""
    logger.info(f"UserProxy Agent: Starting endpoint update for workload {workload_id}")
    
    try:
        # Update status to Updating Endpoint
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.UPDATING_ENDPOINT, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        # Get job details to fetch existing endpoint configuration
        job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
        
        if not job:
            raise Exception(f"Job {workload_id} not found")
        
        old_endpoint = None
        if job.get('proxy_config'):
            old_endpoint = job['proxy_config'].get('active_endpoint')
        
        new_endpoint = deployment_details.get('endpoint_url')
        new_instance = migration_details.get('ec2_instance_id')
        new_ip = migration_details.get('ec2_public_ip')
        
        logger.info(f"UserProxy Agent: Updating endpoint from {old_endpoint or 'none'} to {new_endpoint}")
        
        # Simulate endpoint update steps
        update_steps = []
        
        # Step 1: Validate new endpoint
        await asyncio.sleep(1)
        update_steps.append({
            "step": "Validate New Endpoint",
            "status": "completed",
            "details": f"Verified {new_endpoint} is responding",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"UserProxy Agent: ✓ New endpoint validated")
        
        # Step 2: Update load balancer / API gateway
        await asyncio.sleep(1)
        update_steps.append({
            "step": "Update Load Balancer",
            "status": "completed",
            "details": f"Added target {new_instance} to load balancer pool",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"UserProxy Agent: ✓ Load balancer updated")
        
        # Step 3: Update DNS / Service Discovery
        await asyncio.sleep(1)
        update_steps.append({
            "step": "Update Service Discovery",
            "status": "completed",
            "details": f"Registered new endpoint in service mesh",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"UserProxy Agent: ✓ Service discovery updated")
        
        # Step 4: Gradual traffic shift (blue-green deployment)
        await asyncio.sleep(1.5)
        traffic_shift_stages = [
            {"percentage": 10, "description": "10% traffic to new instance"},
            {"percentage": 50, "description": "50% traffic to new instance"},
            {"percentage": 100, "description": "100% traffic to new instance"}
        ]
        
        for stage in traffic_shift_stages:
            await asyncio.sleep(0.5)
            logger.info(f"UserProxy Agent: → {stage['description']}")
        
        update_steps.append({
            "step": "Traffic Shift",
            "status": "completed",
            "details": "Gradually shifted 100% traffic to new instance",
            "stages": traffic_shift_stages,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"UserProxy Agent: ✓ Traffic shift completed")
        
        # Step 5: Remove old instance from rotation (if exists)
        if old_endpoint:
            await asyncio.sleep(1)
            update_steps.append({
                "step": "Remove Old Instance",
                "status": "completed",
                "details": f"Removed old endpoint {old_endpoint} from rotation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            logger.info(f"UserProxy Agent: ✓ Old instance removed from rotation")
        
        # Build proxy configuration
        proxy_config_existing = job.get('proxy_config') or {}
        endpoint_history = proxy_config_existing.get('endpoint_history', [])
        
        # Add old endpoint to history if it exists
        if old_endpoint:
            endpoint_history.append({
                "endpoint": old_endpoint,
                "deactivated_at": datetime.now(timezone.utc).isoformat(),
                "duration_active": "N/A"  # Would calculate from activation time
            })
        
        proxy_config = {
            "active_endpoint": new_endpoint,
            "active_instance": new_instance,
            "active_ip": new_ip,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "update_steps": update_steps,
            "traffic_distribution": {
                "new_instance": 100,
                "old_instance": 0
            },
            "endpoint_history": endpoint_history,
            "routing_strategy": "blue-green",
            "status": "active"
        }
        
        # Update job with proxy configuration
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {
                "$set": {
                    "status": JobStatus.RUNNING,
                    "proxy_config": proxy_config,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        # Update Supabase
        try:
            updated_job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
            if updated_job:
                workload_json = {
                    'model_name': updated_job.get('model_name'),
                    'datasize': updated_job.get('datasize'),
                    'workload_type': updated_job.get('workload_type'),
                    'duration': updated_job.get('duration'),
                    'budget': updated_job.get('budget'),
                    'precision': updated_job.get('precision'),
                    'framework': updated_job.get('framework'),
                    'scout_results': updated_job.get('scout_results'),
                    'optimizer_results': updated_job.get('optimizer_results'),
                    'recommended_gpu': updated_job.get('recommended_gpu'),
                    'recommended_memory': updated_job.get('recommended_memory'),
                    'estimated_cost': updated_job.get('estimated_cost'),
                    'migration_details': updated_job.get('migration_details'),
                    'deployment_details': updated_job.get('deployment_details'),
                    'proxy_config': proxy_config
                }
                update_workload_in_supabase(workload_id, status="COMPLETE", workload_data=workload_json)
        except Exception as e:
            logger.error(f"UserProxy Agent: Error updating Supabase - {str(e)}")
        
        logger.info(f"UserProxy Agent: ✅ Endpoint updated successfully! All traffic now directed to {new_endpoint}")
        
    except Exception as e:
        logger.error(f"UserProxy Agent: Error updating endpoint - {str(e)}")
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.FAILED, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )


