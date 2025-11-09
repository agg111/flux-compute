"""
Supabase utility functions for workload and migration tracking
"""
import logging
from datetime import datetime, timezone
from config.supabase_config import supabase

logger = logging.getLogger(__name__)


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
        supabase.table('workloads').upsert(data).execute()
        logger.info(f"Saved/Updated workload {workload_id} in Supabase with status {status}")
        
    except Exception as e:
        logger.error(f"Error saving workload to Supabase: {str(e)}")


def update_workload_in_supabase(workload_id: str, status: str = None, workload_data: dict = None):
    """Update workload in Supabase"""
    try:
        if not supabase:
            logger.warning("Supabase not configured")
            return
        
        update_data = {'updated_at': datetime.now(timezone.utc).isoformat()}
        
        if status:
            update_data['status'] = status
        if workload_data:
            update_data['workload_data'] = workload_data
        
        supabase.table('workloads').update(update_data).eq('workload_id', workload_id).execute()
        logger.info(f"Updated workload {workload_id} in Supabase")
        
    except Exception as e:
        logger.error(f"Error updating workload in Supabase: {str(e)}")


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
