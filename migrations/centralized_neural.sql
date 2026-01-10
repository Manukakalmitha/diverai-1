-- Create table for global synaptic weights (Fusion logic)
CREATE TABLE IF NOT EXISTS neural_state (
    id TEXT PRIMARY KEY DEFAULT 'global_master',
    alpha FLOAT DEFAULT 0.35,
    beta FLOAT DEFAULT 0.30,
    gamma FLOAT DEFAULT 0.20,
    delta FLOAT DEFAULT 0.10,
    omega FLOAT DEFAULT 0.50,
    iterations INTEGER DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    updated_by UUID REFERENCES auth.users(id)
);

-- Initialize the master state if it doesn't exist
INSERT INTO neural_state (id, alpha, beta, gamma, delta, omega, iterations)
VALUES ('global_master', 0.35, 0.30, 0.20, 0.10, 0.50, 0)
ON CONFLICT (id) DO NOTHING;

-- Create table for neural model weights (BLOBs/JSON for LSTM)
CREATE TABLE IF NOT EXISTS neural_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id),
    name TEXT NOT NULL,
    model_json JSONB,
    weights_url TEXT, -- Point to Storage if weights are large
    accuracy FLOAT,
    iterations INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- RLS Policies
ALTER TABLE neural_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE neural_models ENABLE ROW LEVEL SECURITY;

-- Everyone can read the global master state
CREATE POLICY "Public read neural master" ON neural_state 
    FOR SELECT USING (id = 'global_master');

-- Only Pro users or specific triggers should update the master (for now, let's allow authenticated users to contribute)
CREATE POLICY "Authenticated users update neural master" ON neural_state
    FOR UPDATE USING (auth.role() = 'authenticated')
    WITH CHECK (auth.role() = 'authenticated');

-- Model isolation
CREATE POLICY "Users can manage their own models" ON neural_models
    FOR ALL USING (auth.uid() = user_id);
