-- Create simplified workloads table in Supabase
-- Run this SQL in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS workloads (
    id BIGSERIAL PRIMARY KEY,
    workload_id UUID NOT NULL UNIQUE,
    workload_data JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETE')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_workloads_workload_id ON workloads(workload_id);
CREATE INDEX IF NOT EXISTS idx_workloads_status ON workloads(status);
CREATE INDEX IF NOT EXISTS idx_workloads_created_at ON workloads(created_at DESC);

-- Enable Row Level Security
ALTER TABLE workloads ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations
CREATE POLICY "Allow all operations on workloads"
ON workloads
FOR ALL
USING (true)
WITH CHECK (true);

-- Create optimization_plans table (keeping this for optimization history)
CREATE TABLE IF NOT EXISTS optimization_plans (
    id BIGSERIAL PRIMARY KEY,
    workload_id UUID NOT NULL UNIQUE,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for optimization_plans
CREATE INDEX IF NOT EXISTS idx_optimization_plans_workload_id ON optimization_plans(workload_id);
CREATE INDEX IF NOT EXISTS idx_optimization_plans_created_at ON optimization_plans(created_at DESC);

-- Enable Row Level Security
ALTER TABLE optimization_plans ENABLE ROW LEVEL SECURITY;

-- Create policy for optimization_plans
CREATE POLICY "Allow all operations on optimization_plans"
ON optimization_plans
FOR ALL
USING (true)
WITH CHECK (true);
