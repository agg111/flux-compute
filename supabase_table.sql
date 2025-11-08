-- Create optimization_plans table in Supabase
-- Run this SQL in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS optimization_plans (
    id BIGSERIAL PRIMARY KEY,
    workload_id UUID NOT NULL UNIQUE,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on workload_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_optimization_plans_workload_id 
ON optimization_plans(workload_id);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_optimization_plans_created_at 
ON optimization_plans(created_at DESC);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE optimization_plans ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (adjust based on your security needs)
CREATE POLICY "Allow all operations on optimization_plans"
ON optimization_plans
FOR ALL
USING (true)
WITH CHECK (true);
