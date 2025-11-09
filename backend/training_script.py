#!/usr/bin/env python3
"""
Linear Regression Training Script with S3 Checkpointing
This script runs on EC2 instances and saves checkpoint to S3 only when migration is requested
"""

import numpy as np
import pickle
import time
import json
import os
import sys
from datetime import datetime
import boto3
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuration from environment variables
WORKLOAD_ID = os.environ.get('WORKLOAD_ID', 'unknown')
S3_BUCKET = os.environ.get('S3_BUCKET', 'ml-workload-checkpoints')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
CHECK_MIGRATION_INTERVAL = int(os.environ.get('CHECK_MIGRATION_INTERVAL', '10'))  # Check every N iterations
TOTAL_ITERATIONS = int(os.environ.get('TOTAL_ITERATIONS', '1000'))  # ~10 minutes

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print(f"Starting Linear Regression Training for workload: {WORKLOAD_ID}")
print(f"S3 Bucket: {S3_BUCKET}")
print(f"Total Iterations: {TOTAL_ITERATIONS}")
print(f"Checkpoint Interval: {CHECKPOINT_INTERVAL} iterations")


def save_checkpoint_to_s3(checkpoint_data, iteration):
    """Save checkpoint to S3"""
    try:
        checkpoint_key = f"checkpoints/{WORKLOAD_ID}/checkpoint_{iteration}.pkl"
        metadata_key = f"checkpoints/{WORKLOAD_ID}/metadata.json"
        
        # Save checkpoint pickle
        checkpoint_bytes = pickle.dumps(checkpoint_data)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=checkpoint_key,
            Body=checkpoint_bytes
        )
        
        # Save metadata
        metadata = {
            'workload_id': WORKLOAD_ID,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_key': checkpoint_key,
            'total_iterations': TOTAL_ITERATIONS
        }
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=metadata_key,
            Body=json.dumps(metadata)
        )
        
        print(f"✓ Checkpoint saved to S3: {checkpoint_key}")
        return True
    except Exception as e:
        print(f"✗ Error saving checkpoint: {str(e)}")
        return False


def load_checkpoint_from_s3():
    """Load latest checkpoint from S3"""
    try:
        metadata_key = f"checkpoints/{WORKLOAD_ID}/metadata.json"
        
        # Try to get metadata
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=metadata_key)
        metadata = json.loads(response['Body'].read())
        
        checkpoint_key = metadata['checkpoint_key']
        iteration = metadata['iteration']
        
        # Load checkpoint
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=checkpoint_key)
        checkpoint_data = pickle.loads(response['Body'].read())
        
        print(f"✓ Checkpoint loaded from S3: iteration {iteration}")
        return checkpoint_data, iteration
    except s3_client.exceptions.NoSuchKey:
        print("No existing checkpoint found, starting from scratch")
        return None, 0
    except Exception as e:
        print(f"✗ Error loading checkpoint: {str(e)}")
        return None, 0


def generate_synthetic_data(n_samples=10000, n_features=50):
    """Generate synthetic data for linear regression"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.randn(n_samples) * 0.1
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    print("\n" + "="*60)
    print("PHASE 1: Data Preparation")
    print("="*60)
    
    # Generate data
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    print(f"✓ Generated training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"✓ Generated test data: {X_test.shape[0]} samples")
    
    print("\n" + "="*60)
    print("PHASE 2: Model Initialization")
    print("="*60)
    
    # Try to load existing checkpoint
    checkpoint_data, start_iteration = load_checkpoint_from_s3()
    
    if checkpoint_data:
        print(f"✓ Resuming from iteration {start_iteration}")
        model = checkpoint_data['model']
        training_history = checkpoint_data['history']
    else:
        print("✓ Starting new training")
        model = SGDRegressor(max_iter=1, warm_start=True, random_state=42)
        training_history = []
        start_iteration = 0
    
    print("\n" + "="*60)
    print("PHASE 3: Training Loop")
    print("="*60)
    print(f"Training from iteration {start_iteration} to {TOTAL_ITERATIONS}")
    print("-"*60)
    
    training_start_time = time.time()
    
    for iteration in range(start_iteration, TOTAL_ITERATIONS):
        # Train for one iteration
        model.partial_fit(X_train, y_train)
        
        # Calculate metrics every 10 iterations
        if iteration % 10 == 0 or iteration == TOTAL_ITERATIONS - 1:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            training_history.append({
                'iteration': iteration,
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Iteration {iteration}/{TOTAL_ITERATIONS} | "
                  f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f} | "
                  f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        
        # Save checkpoint periodically
        if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                'model': model,
                'history': training_history,
                'iteration': iteration + 1,
                'X_train_shape': X_train.shape,
                'timestamp': datetime.now().isoformat()
            }
            save_checkpoint_to_s3(checkpoint_data, iteration + 1)
        
        # Simulate training time (each iteration takes ~0.6 seconds for 10-minute total)
        time.sleep(0.6)
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    print("\n" + "="*60)
    print("PHASE 4: Final Results")
    print("="*60)
    
    # Final evaluation
    y_pred_test = model.predict(X_test)
    final_mse = mean_squared_error(y_test, y_pred_test)
    final_r2 = r2_score(y_test, y_pred_test)
    
    print(f"✓ Training completed in {training_duration:.2f} seconds")
    print(f"✓ Final Test MSE: {final_mse:.4f}")
    print(f"✓ Final Test R²: {final_r2:.4f}")
    print(f"✓ Total Iterations: {TOTAL_ITERATIONS}")
    
    # Save final checkpoint
    final_checkpoint = {
        'model': model,
        'history': training_history,
        'iteration': TOTAL_ITERATIONS,
        'X_train_shape': X_train.shape,
        'final_metrics': {
            'mse': float(final_mse),
            'r2': float(final_r2),
            'training_duration': training_duration
        },
        'timestamp': datetime.now().isoformat()
    }
    save_checkpoint_to_s3(final_checkpoint, TOTAL_ITERATIONS)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Fatal Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
