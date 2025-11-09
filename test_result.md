#====================================================================================================
# START - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================

# THIS SECTION CONTAINS CRITICAL TESTING INSTRUCTIONS FOR BOTH AGENTS
# BOTH MAIN_AGENT AND TESTING_AGENT MUST PRESERVE THIS ENTIRE BLOCK

# Communication Protocol:
# If the `testing_agent` is available, main agent should delegate all testing tasks to it.
#
# You have access to a file called `test_result.md`. This file contains the complete testing state
# and history, and is the primary means of communication between main and the testing agent.
#
# Main and testing agents must follow this exact format to maintain testing data. 
# The testing data must be entered in yaml format Below is the data structure:
# 
## user_problem_statement: {problem_statement}
## backend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.py"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## frontend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.js"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## metadata:
##   created_by: "main_agent"
##   version: "1.0"
##   test_sequence: 0
##   run_ui: false
##
## test_plan:
##   current_focus:
##     - "Task name 1"
##     - "Task name 2"
##   stuck_tasks:
##     - "Task name with persistent issues"
##   test_all: false
##   test_priority: "high_first"  # or "sequential" or "stuck_first"
##
## agent_communication:
##     -agent: "main"  # or "testing" or "user"
##     -message: "Communication message between agents"

# Protocol Guidelines for Main agent
#
# 1. Update Test Result File Before Testing:
#    - Main agent must always update the `test_result.md` file before calling the testing agent
#    - Add implementation details to the status_history
#    - Set `needs_retesting` to true for tasks that need testing
#    - Update the `test_plan` section to guide testing priorities
#    - Add a message to `agent_communication` explaining what you've done
#
# 2. Incorporate User Feedback:
#    - When a user provides feedback that something is or isn't working, add this information to the relevant task's status_history
#    - Update the working status based on user feedback
#    - If a user reports an issue with a task that was marked as working, increment the stuck_count
#    - Whenever user reports issue in the app, if we have testing agent and task_result.md file so find the appropriate task for that and append in status_history of that task to contain the user concern and problem as well 
#
# 3. Track Stuck Tasks:
#    - Monitor which tasks have high stuck_count values or where you are fixing same issue again and again, analyze that when you read task_result.md
#    - For persistent issues, use websearch tool to find solutions
#    - Pay special attention to tasks in the stuck_tasks list
#    - When you fix an issue with a stuck task, don't reset the stuck_count until the testing agent confirms it's working
#
# 4. Provide Context to Testing Agent:
#    - When calling the testing agent, provide clear instructions about:
#      - Which tasks need testing (reference the test_plan)
#      - Any authentication details or configuration needed
#      - Specific test scenarios to focus on
#      - Any known issues or edge cases to verify
#
# 5. Call the testing agent with specific instructions referring to test_result.md
#
# IMPORTANT: Main agent must ALWAYS update test_result.md BEFORE calling the testing agent, as it relies on this file to understand what to test next.

#====================================================================================================
# END - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================



#====================================================================================================
# Testing Data - Main Agent and testing sub agent should log testing data below this section
#====================================================================================================

user_problem_statement: |
  Implement live migration system for ML workloads:
  1. Deploy real Python linear regression script on EC2 instances
  2. Continuous scout agent runs every 2 minutes to find better instances
  3. Trigger migration when ≥20% cost savings found
  4. Save model checkpoints to S3 during migration
  5. Resume training on new instance from checkpoint
  6. UserProxy updates endpoint routing
  7. Terminate old instance after successful migration

backend:
  - task: "S3 Client Initialization and Bucket Setup"
    implemented: true
    working: true
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: true
        agent: "main"
        comment: "Added S3 client initialization, ensure_s3_bucket() function. Bucket ml-workload-checkpoints-gpu-scout created successfully on startup."
  
  - task: "Linear Regression Training Script with S3 Checkpointing"
    implemented: true
    working: "NA"
    file: "/app/backend/training_script.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Created complete training script with scikit-learn. Saves checkpoints to S3 every 50 iterations. Can resume from checkpoint. 1000 iterations total (~10 mins). Needs integration testing on EC2."
  
  - task: "EC2 User-Data Script Generation"
    implemented: true
    working: true
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: true
        agent: "main"
        comment: "Created generate_user_data_script() that deploys training script to EC2 via base64 encoding. Installs dependencies and starts training automatically."
  
  - task: "EC2 Provisioning with Training Deployment"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Modified provision_ec2_instance() to accept deploy_training parameter. When true, deploys training script via user-data. Needs testing to verify script execution on EC2."
  
  - task: "EC2 Instance Termination"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "medium"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Created terminate_ec2_instance() function to terminate old instances. Includes error handling and logging. Needs testing."
  
  - task: "Continuous Scout Monitor Agent"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Created continuous_scout_monitor() that runs every 2 minutes (120s). Checks for cost savings ≥20%. Simulates market changes with random multipliers. Triggers migration when threshold met."
  
  - task: "Migration with Checkpoint Handling"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Created migration_with_checkpoint() agent. Checkpoints training, provisions new instance, resumes from S3. Tracks old instance for cleanup. Needs end-to-end testing."
  
  - task: "UserProxy with Instance Cleanup"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Created user_proxy_with_cleanup() agent. Updates endpoint routing, then terminates old instance. Records termination in migration_details. Needs testing."
  
  - task: "Original Migration Agent Enhancement"
    implemented: true
    working: "NA"
    file: "/app/backend/server.py"
    stuck_count: 0
    priority: "high"
    needs_retesting: true
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Modified migration_agent() to deploy training script (deploy_training=True). Triggers continuous_scout_monitor at the end. Needs testing."

frontend:
  - task: "Display Migration Status"
    implemented: false
    working: "NA"
    file: "N/A"
    stuck_count: 0
    priority: "low"
    needs_retesting: false
    status_history:
      - working: "NA"
        agent: "main"
        comment: "Frontend not modified yet. May need UI updates to show continuous scout status and migration history."

metadata:
  created_by: "main_agent"
  version: "1.0"
  test_sequence: 0
  run_ui: false

test_plan:
  current_focus:
    - "S3 Client Initialization and Bucket Setup"
    - "EC2 Provisioning with Training Deployment"
    - "Continuous Scout Monitor Agent"
    - "Migration with Checkpoint Handling"
    - "Complete end-to-end workflow"
  stuck_tasks: []
  test_all: false
  test_priority: "high_first"

agent_communication:
  - agent: "main"
    message: |
      Implementation complete! Created live migration system with:
      
      ✅ S3 Integration: Bucket created, checkpoint save/load ready
      ✅ Training Script: Real linear regression with scikit-learn, S3 checkpointing
      ✅ EC2 Deployment: User-data script deploys training automatically
      ✅ Continuous Scout: Runs every 2 minutes, 20% threshold
      ✅ Live Migration: Checkpoint → New instance → Resume → Cleanup
      ✅ Instance Cleanup: Old instances terminated after migration
      
      Ready for backend testing:
      1. Test job submission and initial EC2 provisioning with training
      2. Verify training script deploys and runs on EC2
      3. Test continuous scout monitor (runs every 2 min)
      4. Test migration trigger (≥20% savings)
      5. Verify checkpoint save/load to S3
      6. Test endpoint update and old instance termination
      
      IMPORTANT: This will create real EC2 instances and store data in S3.
      Testing should verify:
      - S3 bucket access
      - EC2 instance creation with user-data
      - Training script execution on EC2 (check user-data logs)
      - Checkpoint files in S3
      - Instance termination after migration