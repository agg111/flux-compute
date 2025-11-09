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
import json


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
s3_client = None

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
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        print(f"AWS EC2 client initialized for region {aws_region}")
        print(f"AWS S3 client initialized for region {aws_region}")
    except Exception as e:
        print(f"Failed to initialize AWS clients: {str(e)}")

# S3 bucket for checkpoints
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-workload-checkpoints-gpu-scout')

# Ensure S3 bucket exists
def ensure_s3_bucket():
    """Ensure S3 bucket exists for storing checkpoints"""
    try:
        if not s3_client:
            return False
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
            print(f"S3 bucket {S3_BUCKET_NAME} already exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if aws_region == 'us-east-1':
                        s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
                    else:
                        s3_client.create_bucket(
                            Bucket=S3_BUCKET_NAME,
                            CreateBucketConfiguration={'LocationConstraint': aws_region}
                        )
                    print(f"Created S3 bucket: {S3_BUCKET_NAME}")
                    return True
                except ClientError as create_error:
                    print(f"Failed to create S3 bucket: {str(create_error)}")
                    return False
            else:
                print(f"Error checking S3 bucket: {str(e)}")
                return False
    except Exception as e:
        print(f"Error ensuring S3 bucket: {str(e)}")
        return False

# Initialize S3 bucket on startup
if s3_client:
    ensure_s3_bucket()

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


def save_migration_to_supabase(workload_id: str, migration_data: dict):
    """Save migration record to Supabase migrations table"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return None
        
        data = {
            'workload_id': workload_id,
            'migration_count': migration_data.get('migration_count', 0),
            'instance_id': migration_data.get('instance_id'),
            'instance_type': migration_data.get('instance_type'),
            'public_ip': migration_data.get('public_ip'),
            'private_ip': migration_data.get('private_ip'),
            'availability_zone': migration_data.get('availability_zone'),
            'started_at': migration_data.get('started_at'),
            'ended_at': migration_data.get('ended_at'),
            'cost_per_hour': migration_data.get('cost_per_hour'),
            'previous_cost_per_hour': migration_data.get('previous_cost_per_hour'),
            'cost_improvement_percent': migration_data.get('cost_improvement_percent'),
            'gpu_type': migration_data.get('gpu_type'),
            'memory': migration_data.get('memory'),
            'migration_reason': migration_data.get('migration_reason'),
            'status': migration_data.get('status', 'active'),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        result = supabase.table('migrations').insert(data).execute()
        
        if result.data and len(result.data) > 0:
            logger.info(f"Saved migration record to Supabase for workload {workload_id}, migration_count: {migration_data.get('migration_count')}")
            return result.data[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Error saving migration to Supabase: {str(e)}")
        return None


def update_migration_status_in_supabase(workload_id: str, instance_id: str, status: str, ended_at: str = None):
    """Update migration record status in Supabase"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return
        
        update_data = {
            'status': status,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        if ended_at:
            update_data['ended_at'] = ended_at
        
        supabase.table('migrations').update(update_data).eq('workload_id', workload_id).eq('instance_id', instance_id).execute()
        logger.info(f"Updated migration status to {status} for instance {instance_id}")
        
    except Exception as e:
        logger.error(f"Error updating migration status in Supabase: {str(e)}")


def get_migrations_for_workload(workload_id: str):
    """Get all migration records for a workload"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return []
        
        result = supabase.table('migrations').select('*').eq('workload_id', workload_id).order('migration_count').execute()
        
        if result.data:
            logger.info(f"Retrieved {len(result.data)} migration records for workload {workload_id}")
            return result.data
        
        return []
        
    except Exception as e:
        logger.error(f"Error retrieving migrations from Supabase: {str(e)}")
        return []


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


def generate_user_data_script(workload_id: str) -> str:
    """Generate user-data script to deploy training on EC2 instance"""
    # Read the training script
    script_path = Path(__file__).parent / 'training_script.py'
    with open(script_path, 'r') as f:
        training_script_content = f.read()
    
    # Create base64 encoded training script
    import base64
    encoded_script = base64.b64encode(training_script_content.encode()).decode()
    
    user_data = f"""#!/bin/bash
# Update system
yum update -y

# Install Python 3 and pip
yum install -y python3 python3-pip

# Install required packages
pip3 install numpy scikit-learn boto3

# Create training directory
mkdir -p /home/ec2-user/ml-training
cd /home/ec2-user/ml-training

# Decode and save training script
echo "{encoded_script}" | base64 -d > training_script.py
chmod +x training_script.py

# Set environment variables
export WORKLOAD_ID="{workload_id}"
export S3_BUCKET="{S3_BUCKET_NAME}"
export AWS_REGION="{aws_region}"
export AWS_ACCESS_KEY_ID="{aws_access_key}"
export AWS_SECRET_ACCESS_KEY="{aws_secret_key}"
export CHECK_MIGRATION_INTERVAL="10"
export TOTAL_ITERATIONS="1000"

# Run training script in background with logging
nohup python3 training_script.py > training.log 2>&1 &

# Save PID for monitoring
echo $! > training.pid
"""
    return user_data


def provision_ec2_instance(instance_type: str, workload_id: str, deploy_training: bool = False) -> dict:
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
        
        # Generate user-data script if deploying training
        user_data = generate_user_data_script(workload_id) if deploy_training else None
        
        # Launch instance
        launch_params = {
            'ImageId': ami_id,
            'InstanceType': instance_type,
            'MinCount': 1,
            'MaxCount': 1,
            'SecurityGroupIds': [sg_id],
            'TagSpecifications': [
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'ml-workload-{workload_id[:8]}'},
                        {'Key': 'WorkloadID', 'Value': workload_id},
                        {'Key': 'ManagedBy', 'Value': 'ML-Optimizer'}
                    ]
                }
            ]
        }
        
        if user_data:
            launch_params['UserData'] = user_data
            logger.info(f"Deploying training script via user-data")
        
        instances = ec2_resource.create_instances(**launch_params)
        
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


def terminate_ec2_instance(instance_id: str) -> dict:
    """Terminate an EC2 instance"""
    try:
        if not ec2_client:
            logger.error("EC2 client not initialized")
            return {"status": "error", "message": "AWS not configured"}
        
        logger.info(f"Terminating EC2 instance: {instance_id}")
        
        response = ec2_client.terminate_instances(InstanceIds=[instance_id])
        
        if response['TerminatingInstances']:
            instance_state = response['TerminatingInstances'][0]
            logger.info(f"EC2 instance {instance_id} is terminating: {instance_state['CurrentState']['Name']}")
            return {
                "status": "success",
                "instance_id": instance_id,
                "state": instance_state['CurrentState']['Name']
            }
        
        return {"status": "error", "message": "Failed to terminate instance"}
        
    except Exception as e:
        logger.error(f"Error terminating EC2 instance: {str(e)}")
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


# Import agents from separate files
from agents.scout_agent import scout_agent, continuous_scout_monitor
from agents.optimizer_agent import optimizer_agent
from agents.migration_agent import migration_agent, migration_with_checkpoint
from agents.user_proxy_agent import user_proxy_agent, user_proxy_with_cleanup
from agents.deployer_agent import deployer_agent

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