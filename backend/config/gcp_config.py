"""
GCP Compute Engine client configuration
"""
import os
import logging

logger = logging.getLogger(__name__)

# GCP configuration
gcp_project_id = os.environ.get('GCP_PROJECT_ID', 'yc-vibecon')
gcp_project_number = os.environ.get('GCP_PROJECT_NUMBER', '517414014196')
gcp_zone = os.environ.get('GCP_ZONE', 'us-central1-a')

# Initialize GCP Compute client
compute_client = None

try:
    from google.cloud import compute_v1
    
    # Initialize compute client (uses Application Default Credentials)
    compute_client = compute_v1.InstancesClient()
    
    print(f"GCP Compute Engine client initialized for project {gcp_project_id}")
except ImportError:
    print("google-cloud-compute not installed. Run: pip install google-cloud-compute")
except Exception as e:
    print(f"Failed to initialize GCP Compute client: {str(e)}")
    print("Make sure GCP credentials are configured (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)")
