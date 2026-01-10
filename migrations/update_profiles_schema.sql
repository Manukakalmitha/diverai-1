-- migrations/update_profiles_schema.sql
ALTER TABLE profiles
  ADD COLUMN IF NOT EXISTS subscription_status TEXT DEFAULT 'trial',
  ADD COLUMN IF NOT EXISTS current_period_end TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT;

-- Index for faster lookups during webhooks
CREATE INDEX IF NOT EXISTS idx_profiles_stripe_subscription_id ON profiles (stripe_subscription_id);
