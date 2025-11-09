"""
AWS EC2 and S3 utility functions
"""
import logging
import base64
import json
from pathlib import Path
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from config.aws_config import ec2_client, ec2_resource, s3_client, S3_BUCKET_NAME, aws_access_key, aws_secret_key, aws_region

logger = logging.getLogger(__name__)


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


def ensure_security_group():
    """Ensure security group exists for EC2 instances"""
    try:
        if not ec2_client:
            logger.error("EC2 client not initialized")
            return None
        
        group_name = 'ml-optimizer-sg'
        
        # Check if security group exists
        try:
            response = ec2_client.describe_security_groups(GroupNames=[group_name])
            sg_id = response['SecurityGroups'][0]['GroupId']
            logger.info(f"Security group {group_name} already exists: {sg_id}")
            return sg_id
        except ClientError as e:
            if e.response['Error']['Code'] != 'InvalidGroup.NotFound':
                raise
        
        # Get default VPC
        vpcs = ec2_client.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
        if not vpcs['Vpcs']:
            logger.error("No default VPC found")
            return None
        
        vpc_id = vpcs['Vpcs'][0]['VpcId']
        
        # Create security group
        sg_response = ec2_client.create_security_group(
            GroupName=group_name,
            Description='Security group for ML workload optimizer',
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
    script_path = Path(__file__).parent.parent / 'training_script.py'
    with open(script_path, 'r') as f:
        training_script_content = f.read()
    
    # Create base64 encoded training script
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


def provision_gcp_instance(instance_type: str, workload_id: str, deploy_training: bool = False) -> dict:
    """Provision a real GCP Compute Engine instance"""
    try:
        from google.cloud import compute_v1
        from config.gcp_config import compute_client, gcp_project_id, gcp_zone
        import time
        
        if not compute_client:
            logger.error("GCP Compute client not initialized")
            return {"status": "error", "message": "GCP not configured"}
        
        # For testing, use e2-micro (cheapest instance)
        machine_type = "e2-micro"  # Similar to AWS t3.micro
        instance_name = f"ml-workload-{workload_id[:8]}"
        
        logger.info(f"Provisioning GCP instance: {machine_type} with name {instance_name}")
        
        # Configure instance
        instance = compute_v1.Instance()
        instance.name = instance_name
        instance.machine_type = f"zones/{gcp_zone}/machineTypes/{machine_type}"
        
        # Boot disk configuration
        boot_disk = compute_v1.AttachedDisk()
        initialize_params = compute_v1.AttachedDiskInitializeParams()
        initialize_params.source_image = "projects/debian-cloud/global/images/family/debian-11"
        initialize_params.disk_size_gb = 10
        boot_disk.initialize_params = initialize_params
        boot_disk.auto_delete = True
        boot_disk.boot = True
        instance.disks = [boot_disk]
        
        # Network configuration
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = "global/networks/default"
        
        # Add external IP
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]
        
        # Labels
        instance.labels = {
            "workload_id": workload_id[:8],
            "managed_by": "ml-optimizer"
        }
        
        # Startup script if deploying training
        if deploy_training:
            metadata_items = []
            startup_script = generate_user_data_script(workload_id)  # Reuse AWS user-data script
            
            metadata_item = compute_v1.Items()
            metadata_item.key = "startup-script"
            metadata_item.value = startup_script
            metadata_items.append(metadata_item)
            
            metadata = compute_v1.Metadata()
            metadata.items = metadata_items
            instance.metadata = metadata
            
            logger.info(f"Deploying training script via startup-script")
        
        # Create instance
        operation = compute_client.insert(
            project=gcp_project_id,
            zone=gcp_zone,
            instance_resource=instance
        )
        
        logger.info(f"GCP instance creation initiated: {instance_name}, waiting for completion...")
        
        # Wait for operation to complete
        start_time = time.time()
        timeout = 120  # 2 minutes timeout
        
        while time.time() - start_time < timeout:
            if operation.status == compute_v1.Operation.Status.DONE:
                break
            time.sleep(2)
        
        if operation.status != compute_v1.Operation.Status.DONE:
            logger.error(f"GCP instance creation timed out")
            return {"status": "error", "message": "Instance creation timed out"}
        
        # Get instance details
        instance_details = compute_client.get(
            project=gcp_project_id,
            zone=gcp_zone,
            instance=instance_name
        )
        
        # Extract IP addresses
        public_ip = None
        private_ip = None
        
        if instance_details.network_interfaces:
            private_ip = instance_details.network_interfaces[0].network_i_p
            if instance_details.network_interfaces[0].access_configs:
                public_ip = instance_details.network_interfaces[0].access_configs[0].nat_i_p
        
        result = {
            "status": "success",
            "provider": "GCP",
            "instance_id": instance_name,
            "instance_type": machine_type,
            "state": "running",
            "public_ip": public_ip,
            "private_ip": private_ip,
            "zone": gcp_zone,
            "launch_time": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"GCP instance {instance_name} is running at {public_ip}")
        return result
        
    except Exception as e:
        logger.error(f"Error provisioning GCP instance: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def terminate_gcp_instance(instance_id: str) -> dict:
    """Simulate GCP instance termination"""
    try:
        logger.info(f"Simulating GCP instance termination: {instance_id}")
        
        import time
        time.sleep(2)
        
        logger.info(f"GCP instance {instance_id} terminated (simulated)")
        return {
            "status": "success",
            "instance_id": instance_id,
            "state": "terminated"
        }
        
    except Exception as e:
        logger.error(f"Error terminating GCP instance: {str(e)}")
        return {"status": "error", "message": str(e)}


def create_checkpoint_request_flag(workload_id: str) -> bool:
    """Create checkpoint request flag in S3"""
    try:
        checkpoint_request_key = f"migration_requests/{workload_id}/checkpoint_request.flag"
        request_data = {
            'workload_id': workload_id,
            'requested_at': datetime.now(timezone.utc).isoformat(),
            'reason': 'migration_to_better_instance'
        }
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=checkpoint_request_key,
            Body=json.dumps(request_data)
        )
        logger.info(f"Checkpoint request flag created in S3")
        return True
    except Exception as e:
        logger.error(f"Failed to create checkpoint request: {str(e)}")
        return False
