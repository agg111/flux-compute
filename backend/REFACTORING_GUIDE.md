# Backend Code Refactoring Guide

## New Structure

The backend code has been reorganized into a modular structure for better maintainability:

```
backend/
├── server.py (old monolithic file - kept for reference)
├── server_new.py (new streamlined entry point)
├── models.py (Pydantic models and enums)
├── database.py (MongoDB connection)
├── routes.py (API endpoints)
├── training_script.py (ML training script)
├── config/
│   ├── aws_config.py (AWS EC2/S3 client initialization)
│   └── supabase_config.py (Supabase client initialization)
├── utils/
│   ├── aws_utils.py (EC2/S3 operations)
│   ├── supabase_utils.py (Supabase operations)
│   └── helpers.py (Helper functions)
└── agents/
    ├── __init__.py
    ├── scout_agent.py (Scout and Continuous Scout agents)
    ├── optimizer_agent.py (Optimizer agent)
    ├── migration_agent.py (Migration agents)
    ├── user_proxy_agent.py (UserProxy agents)
    └── deployer_agent.py (Deployer agent)
```

## What's Been Created

### Core Files
- **models.py**: All Pydantic models (Job, JobCreate, JobUpdate, Enums)
- **database.py**: MongoDB connection and client
- **routes.py**: All API endpoints with clean imports
- **server_new.py**: Streamlined FastAPI application entry point

### Configuration
- **config/aws_config.py**: AWS EC2, S3 client setup
- **config/supabase_config.py**: Supabase client setup

### Utilities
- **utils/aws_utils.py**: EC2 provisioning, termination, S3 operations, user-data generation
- **utils/supabase_utils.py**: All Supabase CRUD operations for workloads, plans, migrations
- **utils/helpers.py**: HuggingFace model fetching and other helpers

### Agents (To Be Created)
The agent files are large and contain the business logic. They need to be extracted from server.py:
- scout_agent.py
- optimizer_agent.py
- migration_agent.py
- user_proxy_agent.py
- deployer_agent.py

## Migration Plan

### Option 1: Complete the Refactoring
Extract all agent functions from server.py into separate files in the agents/ directory.

### Option 2: Hybrid Approach (Recommended for Testing)
Keep agents in server.py for now, but use the new modular structure for:
- Models
- Database connections
- AWS/Supabase utilities
- API routes

Update server.py to import from the new modules instead of having everything inline.

### Option 3: Incremental Migration
1. Test current setup with new utility files
2. Gradually move agents one by one
3. Update imports as we go

## Benefits of New Structure

1. **Modularity**: Each file has a single responsibility
2. **Reusability**: Utility functions can be easily reused
3. **Testability**: Easier to write unit tests for individual modules
4. **Maintainability**: Easier to find and fix bugs
5. **Scalability**: Easy to add new features without cluttering files
6. **Collaboration**: Multiple developers can work on different modules

## Next Steps

1. Decide on migration approach
2. Create agent files or update imports in server.py
3. Test the refactored code
4. Update supervisor config if needed (use server_new.py instead of server.py)
5. Remove old server.py once everything works

## Import Examples

### Before (in server.py)
```python
# Everything in one file
ec2_client = boto3.client(...)
supabase = create_client(...)
def provision_ec2_instance(...):
    ...
```

### After (modular)
```python
# In routes.py
from config.aws_config import ec2_client
from utils.aws_utils import provision_ec2_instance
from utils.supabase_utils import save_workload_to_supabase
```

## Testing Checklist

- [ ] All imports resolve correctly
- [ ] API endpoints work
- [ ] Jobs can be created
- [ ] Agents trigger properly
- [ ] AWS operations work
- [ ] Supabase operations work
- [ ] Migration tracking works
- [ ] Frontend can communicate with backend
