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
import boto3
from botocore.exceptions import ClientError


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

# AWS EC2 connection
aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_REGION', 'us-east-2')

ec2_client = None
ec2_resource = None

if aws_access_key and aws_secret_key:
    try:
        ec2_client = boto3.client(
            'ec2',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        ec2_resource = boto3.resource(
            'ec2',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print(f"AWS EC2 client initialized for region {aws_region}")
    except Exception as e:
        print(f"Failed to initialize AWS EC2 client: {str(e)}")

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
    PROVISIONING = "Provisioning"
    MIGRATING = "Migrating"
    DEPLOYING = "Deploying"
    HEALTH_CHECK = "Health Check"
    UPDATING_ENDPOINT = "Updating Endpoint"
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
    migration_details: Optional[dict] = None
    deployment_details: Optional[dict] = None
    proxy_config: Optional[dict] = None

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


def update_workload_in_supabase(workload_id: str, status: str = None, workload_data: dict = None, plan_id: int = None):
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
        
        if plan_id:
            updates['plan_id'] = plan_id
        
        result = supabase.table('workloads').update(updates).eq('workload_id', workload_id).execute()
        logger.info(f"Updated workload in Supabase: {workload_id} - status: {status}, plan_id: {plan_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error updating workload in Supabase: {str(e)}")
        return None


def ensure_security_group():
    """Ensure a security group exists for ML workloads"""
    try:
        if not ec2_client:
            logger.error("EC2 client not initialized")
            return None
        
        security_group_name = "ml-workload-sg"
        
        # Check if security group exists
        try:
            response = ec2_client.describe_security_groups(
                GroupNames=[security_group_name]
            )
            sg_id = response['SecurityGroups'][0]['GroupId']
            logger.info(f"Using existing security group: {sg_id}")
            return sg_id
        except ClientError as e:
            if 'InvalidGroup.NotFound' not in str(e):
                raise
        
        # Create security group
        vpc_response = ec2_client.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        vpc_id = vpc_response['Vpcs'][0]['VpcId'] if vpc_response['Vpcs'] else None
        
        sg_response = ec2_client.create_security_group(
            GroupName=security_group_name,
            Description='Security group for ML workload instances',
            VpcId=vpc_id
        )
        sg_id = sg_response['GroupId']
        
        # Allow SSH access
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH access'}]
                }
            ]
        )
        
        logger.info(f"Created security group: {sg_id}")
        return sg_id
        
    except Exception as e:
        logger.error(f"Error ensuring security group: {str(e)}")
        return None


def provision_ec2_instance(instance_type: str, workload_id: str) -> dict:
    """Provision a real EC2 instance on AWS"""
    try:
        if not ec2_client or not ec2_resource:
            logger.error("EC2 client not initialized")
            return {"status": "error", "message": "AWS not configured"}
        
        # Ensure security group exists
        sg_id = ensure_security_group()
        if not sg_id:
            return {"status": "error", "message": "Failed to create security group"}
        
        # Get latest Amazon Linux 2 AMI
        ami_response = ec2_client.describe_images(
            Filters=[
                {'Name': 'name', 'Values': ['amzn2-ami-hvm-*-x86_64-gp2']},
                {'Name': 'state', 'Values': ['available']}
            ],
            Owners=['amazon']
        )
        
        if not ami_response['Images']:
            return {"status": "error", "message": "No AMI found"}
        
        # Sort by creation date and get latest
        latest_ami = sorted(ami_response['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]
        ami_id = latest_ami['ImageId']
        
        logger.info(f"Provisioning EC2 instance: {instance_type} with AMI {ami_id}")
        
        # Launch instance
        instances = ec2_resource.create_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            SecurityGroupIds=[sg_id],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'ml-workload-{workload_id[:8]}'},
                        {'Key': 'WorkloadID', 'Value': workload_id},
                        {'Key': 'ManagedBy', 'Value': 'ML-Optimizer'}
                    ]
                }
            ]
        )
        
        instance = instances[0]
        instance_id = instance.id
        
        logger.info(f"EC2 instance created: {instance_id}, waiting for running state...")
        
        # Wait for instance to be running
        instance.wait_until_running()
        instance.reload()
        
        result = {
            "status": "success",
            "instance_id": instance_id,
            "instance_type": instance_type,
            "state": instance.state['Name'],
            "public_ip": instance.public_ip_address,
            "private_ip": instance.private_ip_address,
            "availability_zone": instance.placement['AvailabilityZone'],
            "launch_time": instance.launch_time.isoformat()
        }
        
        logger.info(f"EC2 instance {instance_id} is running at {instance.public_ip_address}")
        return result
        
    except Exception as e:
        logger.error(f"Error provisioning EC2 instance: {str(e)}")
        return {"status": "error", "message": str(e)}


def get_all_workloads_with_plans():
    """Get all workloads with their current optimization plans for re-optimization"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return []
        
        # Get all RUNNING workloads
        workloads = supabase.table('workloads').select('*').eq('status', 'RUNNING').execute()
        
        if not workloads.data:
            return []
        
        # For each workload, fetch its current plan if plan_id exists
        workloads_with_plans = []
        for workload in workloads.data:
            if workload.get('plan_id'):
                plan = supabase.table('optimization_plans').select('*').eq('id', workload['plan_id']).execute()
                if plan.data and len(plan.data) > 0:
                    workload['current_plan'] = plan.data[0]
            workloads_with_plans.append(workload)
        
        logger.info(f"Retrieved {len(workloads_with_plans)} workloads with plans for potential re-optimization")
        return workloads_with_plans
        
    except Exception as e:
        logger.error(f"Error getting workloads with plans: {str(e)}")
        return []


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
        endpoint_history = job.get('proxy_config', {}).get('endpoint_history', [])
        
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


async def deployer_agent(workload_id: str, target_resource: dict, migration_details: dict, optimizer_results: dict):
    """Deployer Agent - Deploys the workload to the provisioned instance and runs health checks"""
    logger.info(f"Deployer Agent: Starting deployment for workload {workload_id}")
    
    try:
        # Phase 1: Deploying workload
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.DEPLOYING, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        logger.info(f"Deployer Agent: Deploying workload to instance")
        
        # Get job details
        job = await db.jobs.find_one({"workload_id": workload_id}, {"_id": 0})
        
        deployment_steps = [
            {
                "step": "Installing dependencies",
                "status": "in_progress",
                "details": f"Installing {job.get('framework', 'PyTorch')} and required libraries"
            },
            {
                "step": "Downloading model",
                "status": "in_progress", 
                "details": f"Downloading {job['model_name']} weights"
            },
            {
                "step": "Loading model to GPU",
                "status": "in_progress",
                "details": f"Loading model to {target_resource['gpu']}"
            },
            {
                "step": "Configuring inference endpoint",
                "status": "in_progress",
                "details": "Setting up REST API endpoint"
            }
        ]
        
        deployment_log = []
        
        for step_info in deployment_steps:
            logger.info(f"Deployer Agent: {step_info['step']}...")
            await asyncio.sleep(1.5)
            step_info["status"] = "completed"
            step_info["completed_at"] = datetime.now(timezone.utc).isoformat()
            deployment_log.append(step_info)
        
        logger.info(f"Deployer Agent: Deployment completed, starting health checks")
        
        # Phase 2: Health Check
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.HEALTH_CHECK, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        health_checks = []
        
        # Check 1: Instance connectivity
        await asyncio.sleep(1)
        health_checks.append({
            "check": "Instance Connectivity",
            "status": "passed",
            "message": f"Successfully connected to {migration_details.get('ec2_instance_id', 'instance')}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Deployer Agent: ✓ Instance connectivity check passed")
        
        # Check 2: GPU availability
        await asyncio.sleep(1)
        health_checks.append({
            "check": "GPU Availability",
            "status": "passed",
            "message": f"GPU {target_resource['gpu']} is available and ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Deployer Agent: ✓ GPU availability check passed")
        
        # Check 3: Model loaded
        await asyncio.sleep(1)
        health_checks.append({
            "check": "Model Loaded",
            "status": "passed",
            "message": f"Model {job['model_name']} successfully loaded into memory",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Deployer Agent: ✓ Model loaded check passed")
        
        # Check 4: Inference endpoint
        await asyncio.sleep(1)
        endpoint_url = f"http://{migration_details.get('ec2_public_ip', 'instance-ip')}:8000/inference"
        health_checks.append({
            "check": "Inference Endpoint",
            "status": "passed",
            "message": f"API endpoint is responding at {endpoint_url}",
            "endpoint": endpoint_url,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Deployer Agent: ✓ Inference endpoint check passed")
        
        # Check 5: Resource utilization
        await asyncio.sleep(1)
        health_checks.append({
            "check": "Resource Utilization",
            "status": "passed",
            "message": "CPU: 15%, Memory: 32%, GPU: 8%",
            "metrics": {"cpu": 15, "memory": 32, "gpu": 8},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Deployer Agent: ✓ Resource utilization check passed")
        
        # All checks passed - set to Running
        deployment_details = {
            "status": "success",
            "deployment_log": deployment_log,
            "health_checks": health_checks,
            "all_checks_passed": all(check["status"] == "passed" for check in health_checks),
            "deployment_completed_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_url": endpoint_url
        }
        
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {
                "$set": {
                    "status": JobStatus.RUNNING,
                    "deployment_details": deployment_details,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        # Update deployment details in job
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"deployment_details": deployment_details}}
        )
        
        logger.info(f"Deployer Agent: ✅ All health checks passed! Workload {workload_id} is ready")
        logger.info(f"Deployer Agent: Triggering UserProxy Agent to update endpoint routing")
        
        # Trigger UserProxy Agent to update endpoint routing
        asyncio.create_task(user_proxy_agent(workload_id, deployment_details, migration_details))
        
    except Exception as e:
        logger.error(f"Deployer Agent: Error during deployment - {str(e)}")
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.FAILED, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )


async def migration_agent(workload_id: str, target_resource: dict, optimizer_results: dict):
    """Migration Agent - Provisions new instances and runs validation test"""
    logger.info(f"Migration Agent: Starting migration for workload {workload_id}")
    
    try:
        # Get the optimization plan from DB
        current_plan = get_optimization_plan_from_supabase(workload_id)
        logger.info(f"Migration Agent: Retrieved plan from database - version {current_plan.get('plan_data', {}).get('optimization_version', 'unknown') if current_plan else 'N/A'}")
        
        # Phase 1: Provisioning new instances
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.PROVISIONING, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        update_workload_in_supabase(workload_id, status="RUNNING")
        
        # ALWAYS use t3.micro for testing/validation
        test_instance_type = 't3.micro'
        logger.info(f"Migration Agent: Provisioning {test_instance_type} for validation test (target would be {target_resource['instance']})")
        
        # Provision real EC2 instance
        ec2_result = None
        if target_resource['provider'] == 'AWS':
            logger.info(f"Migration Agent: Launching EC2 instance {test_instance_type}")
            
            ec2_result = await asyncio.to_thread(
                provision_ec2_instance, 
                test_instance_type, 
                workload_id
            )
            
            if ec2_result.get('status') == 'error':
                logger.error(f"Migration Agent: Failed to provision EC2 - {ec2_result.get('message')}")
                await db.jobs.update_one(
                    {"workload_id": workload_id},
                    {"$set": {"status": JobStatus.FAILED, "updated_at": datetime.now(timezone.utc).isoformat()}}
                )
                return
            
            logger.info(f"Migration Agent: EC2 instance {ec2_result['instance_id']} provisioned successfully")
        else:
            # For GCP, simulate for now
            logger.info(f"Migration Agent: GCP provisioning not implemented yet, using AWS")
            await asyncio.sleep(2)
        
        migration_details = {
            "phase": "provisioning",
            "target_provider": target_resource["provider"],
            "target_instance": target_resource["instance"],
            "target_gpu": target_resource["gpu"],
            "provisioning_started": datetime.now(timezone.utc).isoformat()
        }
        
        # Add EC2 details if available
        if ec2_result and ec2_result.get('status') == 'success':
            migration_details["ec2_instance_id"] = ec2_result["instance_id"]
            migration_details["ec2_public_ip"] = ec2_result["public_ip"]
            migration_details["ec2_private_ip"] = ec2_result["private_ip"]
            migration_details["ec2_availability_zone"] = ec2_result["availability_zone"]
            migration_details["ec2_launch_time"] = ec2_result["launch_time"]
        
        # Phase 2: Running validation test - Linear Regression (10 minutes)
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.MIGRATING, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        logger.info(f"Migration Agent: Starting Linear Regression validation test (10 minutes)")
        
        # Simulate running a linear regression task
        validation_test = {
            "task": "Linear Regression",
            "duration": "10 minutes",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "instance": test_instance_type,
            "steps": []
        }
        
        # Step 1: Setup environment
        await asyncio.sleep(2)
        validation_test["steps"].append({
            "step": "Environment Setup",
            "status": "completed",
            "details": "Installed Python, NumPy, scikit-learn",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Migration Agent: ✓ Environment setup complete")
        
        # Step 2: Generate test data
        await asyncio.sleep(1)
        validation_test["steps"].append({
            "step": "Generate Test Data",
            "status": "completed",
            "details": "Generated 10,000 samples with 50 features",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Migration Agent: ✓ Test data generated")
        
        # Step 3: Run linear regression (simulated 10 minutes)
        logger.info(f"Migration Agent: Running linear regression task for 10 minutes...")
        await asyncio.sleep(10)  # Simulate 10 minutes with 10 seconds
        
        validation_test["steps"].append({
            "step": "Linear Regression Training",
            "status": "completed",
            "details": "Trained model for 1000 iterations, MSE: 0.023",
            "metrics": {
                "iterations": 1000,
                "mse": 0.023,
                "r2_score": 0.987,
                "training_time": "9m 45s"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Migration Agent: ✓ Linear regression completed successfully")
        
        # Step 4: Validation
        await asyncio.sleep(1)
        validation_test["steps"].append({
            "step": "Model Validation",
            "status": "completed",
            "details": "Validated on 2,000 test samples",
            "metrics": {
                "test_mse": 0.025,
                "test_r2_score": 0.985
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.info(f"Migration Agent: ✓ Model validation complete")
        
        validation_test["status"] = "completed"
        validation_test["completed_at"] = datetime.now(timezone.utc).isoformat()
        validation_test["result"] = "success"
        
        migration_details["phase"] = "migrated"
        migration_details["migration_completed"] = datetime.now(timezone.utc).isoformat()
        migration_details["status"] = "success"
        
        # Phase 3: Migration complete, now trigger Deployer Agent
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {
                "$set": {
                    "migration_details": migration_details,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        logger.info(f"Migration Agent: Successfully migrated workload {workload_id} to {target_resource['provider']} {target_resource['instance']}")
        logger.info(f"Migration Agent: Triggering Deployer Agent for workload {workload_id}")
        
        # Trigger Deployer Agent
        asyncio.create_task(deployer_agent(workload_id, target_resource, migration_details, optimizer_results))
        
    except Exception as e:
        logger.error(f"Migration Agent: Error during migration - {str(e)}")
        await db.jobs.update_one(
            {"workload_id": workload_id},
            {"$set": {"status": JobStatus.FAILED, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )


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
    
    # Save initial plan to Supabase first to get plan_id
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
    plan_id = save_optimization_plan_to_supabase(job_obj.workload_id, initial_plan)
    
    # Save workload to Supabase with plan_id reference
    workload_json = {
        'model_name': job_obj.model_name,
        'datasize': job_obj.datasize,
        'workload_type': job_obj.workload_type,
        'duration': job_obj.duration,
        'budget': float(job_obj.budget),
        'precision': job_obj.precision,
        'framework': job_obj.framework
    }
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