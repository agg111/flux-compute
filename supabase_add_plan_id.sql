-- Add plan_id column to workloads table
-- Run this SQL in your Supabase SQL Editor

-- Add the plan_id column as a foreign key to optimization_plans
ALTER TABLE workloads 
ADD COLUMN plan_id BIGINT;

-- Create foreign key constraint
ALTER TABLE workloads
ADD CONSTRAINT fk_workloads_plan_id
FOREIGN KEY (plan_id)
REFERENCES optimization_plans(id)
ON DELETE SET NULL;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_workloads_plan_id ON workloads(plan_id);

-- Optional: Add comment to explain the column
COMMENT ON COLUMN workloads.plan_id IS 'References the current optimization plan ID from optimization_plans table';
