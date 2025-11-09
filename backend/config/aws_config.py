"""
AWS EC2 and S3 client configuration
"""
import os
import boto3
import logging

logger = logging.getLogger(__name__)

# AWS credentials
aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_region = os.environ.get('AWS_REGION', 'us-east-2')

# S3 bucket for checkpoints
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'ml-workload-checkpoints-gpu-scout')

# Initialize clients
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
