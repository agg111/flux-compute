-- Create migrations table to track workload migration history
CREATE TABLE IF NOT EXISTS migrations (
    id BIGSERIAL PRIMARY KEY,
    workload_id TEXT NOT NULL,
    migration_count INTEGER NOT NULL DEFAULT 0,
    instance_id TEXT NOT NULL,
    instance_type TEXT,
    recommended_instance_type TEXT,
    provider TEXT,
    public_ip TEXT,
    private_ip TEXT,
    availability_zone TEXT,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    cost_per_hour DECIMAL(10, 3),
    actual_cost_per_hour DECIMAL(10, 3),
    previous_cost_per_hour DECIMAL(10, 3),
    cost_improvement_percent DECIMAL(5, 2),
    gpu_type TEXT,
    memory TEXT,
    migration_reason TEXT,
    status TEXT DEFAULT 'active',
    ai_powered BOOLEAN DEFAULT false,
    ai_confidence INTEGER,
    checkpoint_s3_key TEXT,
    checkpoint_iteration INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on workload_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_migrations_workload_id ON migrations(workload_id);

-- Create index on migration_count for ordering
CREATE INDEX IF NOT EXISTS idx_migrations_count ON migrations(workload_id, migration_count);

-- Comments for documentation
COMMENT ON TABLE migrations IS 'Tracks migration history for ML workloads across EC2 instances';
COMMENT ON COLUMN migrations.migration_count IS '0 = initial instance, 1+ = subsequent migrations';
COMMENT ON COLUMN migrations.cost_improvement_percent IS 'Percentage improvement in cost (positive = cheaper, negative = more expensive)';
COMMENT ON COLUMN migrations.status IS 'active, terminated, failed';
