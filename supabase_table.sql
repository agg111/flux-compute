-- Create tables in Supabase
-- Run this SQL in your Supabase SQL Editor

-- 1. Workloads Table - Store all workload submissions
CREATE TABLE IF NOT EXISTS workloads (
    id BIGSERIAL PRIMARY KEY,
    workload_id UUID NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    datasize TEXT NOT NULL,
    workload_type TEXT NOT NULL,
    duration TEXT NOT NULL,
    budget NUMERIC NOT NULL,
    precision TEXT,
    framework TEXT,
    status TEXT NOT NULL DEFAULT 'Pending',
    recommended_gpu TEXT,
    recommended_memory TEXT,
    estimated_cost NUMERIC,
    scout_results JSONB,
    optimizer_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Optimization Plans Table - Store optimization history
CREATE TABLE IF NOT EXISTS optimization_plans (
    id BIGSERIAL PRIMARY KEY,
    workload_id UUID NOT NULL UNIQUE,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for workloads table
CREATE INDEX IF NOT EXISTS idx_workloads_workload_id 
ON workloads(workload_id);

CREATE INDEX IF NOT EXISTS idx_workloads_status 
ON workloads(status);

CREATE INDEX IF NOT EXISTS idx_workloads_created_at 
ON workloads(created_at DESC);

-- Create indexes for optimization_plans table
CREATE INDEX IF NOT EXISTS idx_optimization_plans_workload_id 
ON optimization_plans(workload_id);

CREATE INDEX IF NOT EXISTS idx_optimization_plans_created_at 
ON optimization_plans(created_at DESC);

-- Enable Row Level Security
ALTER TABLE workloads ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_plans ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations (adjust based on your security needs)
CREATE POLICY "Allow all operations on workloads"
ON workloads
FOR ALL
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on optimization_plans"
ON optimization_plans
FOR ALL
USING (true)
WITH CHECK (true);
