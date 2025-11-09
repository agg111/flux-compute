"""
Pydantic models and enums for the ML Workload Optimization API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
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


class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.now)


class StatusCheckCreate(BaseModel):
    client_name: str
