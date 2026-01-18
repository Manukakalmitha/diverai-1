-- device_limits_and_security.sql

-- 1. Create Tables for Security & Limits

CREATE TABLE IF NOT EXISTS public.whitelist_devices (
    fingerprint TEXT PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE TABLE IF NOT EXISTS public.device_accounts (
    fingerprint TEXT PRIMARY KEY,
    account_count INTEGER DEFAULT 0,
    last_signup TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE TABLE IF NOT EXISTS public.rate_limits (
    key TEXT PRIMARY KEY,
    count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

CREATE TABLE IF NOT EXISTS public.ai_cache (
    hash TEXT PRIMARY KEY,
    response JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()),
    expires_at TIMESTAMP WITH TIME ZONE
);

ALTER TABLE public.rate_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ai_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.device_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.whitelist_devices ENABLE ROW LEVEL SECURITY;

-- 2. Add Restriction Columns to Profiles

ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS is_restricted BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS restriction_reason TEXT,
ADD COLUMN IF NOT EXISTS restricted_at TIMESTAMP WITH TIME ZONE;

-- 3. Redefine handle_new_user to include Device Limiting

CREATE OR REPLACE FUNCTION public.handle_new_user() 
RETURNS TRIGGER AS $$
DECLARE
    device_fp TEXT;
    current_count INTEGER;
    is_whitelisted BOOLEAN;
BEGIN
  -- Insert the new profile
  INSERT INTO public.profiles (id, email, full_name, avatar_url)
  VALUES (
    new.id, 
    new.email, 
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'), 
    new.raw_user_meta_data->>'avatar_url'
  );

  -- Device Limiting Logic
  device_fp := new.raw_user_meta_data->>'device_fingerprint';
  
  IF device_fp IS NOT NULL THEN
      -- Check whitelist
      SELECT EXISTS(SELECT 1 FROM public.whitelist_devices WHERE fingerprint = device_fp) INTO is_whitelisted;
      
      IF NOT is_whitelisted THEN
          -- Upsert and increment count
          INSERT INTO public.device_accounts (fingerprint, account_count, last_signup)
          VALUES (device_fp, 1, now())
          ON CONFLICT (fingerprint) 
          DO UPDATE SET 
              account_count = device_accounts.account_count + 1,
              last_signup = now()
          RETURNING account_count INTO current_count;
          
          -- Soft Limit: Mark restricted if count > 1
          IF current_count > 1 THEN
              UPDATE public.profiles
              SET 
                  is_restricted = true,
                  restriction_reason = 'Device limit reached',
                  restricted_at = now()
              WHERE id = new.id;
          END IF;
      END IF;
  END IF;

  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 4. RPC for Rate Limiting (Rolling Window)

CREATE OR REPLACE FUNCTION check_rate_limit(
    user_id UUID, 
    endpoint TEXT, 
    cost INTEGER, 
    limit_count INTEGER, 
    window_seconds INTEGER
) 
RETURNS BOOLEAN AS $$
DECLARE
    rate_key TEXT;
    current_window_start TIMESTAMP WITH TIME ZONE;
    current_count INTEGER;
BEGIN
    rate_key := user_id || ':' || endpoint;
    
    -- Check if record exists and is within window
    SELECT window_start, count INTO current_window_start, current_count
    FROM public.rate_limits
    WHERE key = rate_key;
    
    IF NOT FOUND OR current_window_start < (now() - (window_seconds || ' seconds')::interval) THEN
        -- New window or new user
        INSERT INTO public.rate_limits (key, count, window_start)
        VALUES (rate_key, cost, now())
        ON CONFLICT (key) DO UPDATE SET
            count = EXCLUDED.count,
            window_start = EXCLUDED.window_start;
        RETURN TRUE;
    ELSE
        -- Within window, check limit
        IF current_count + cost > limit_count THEN
            RETURN FALSE;
        ELSE
            UPDATE public.rate_limits
            SET count = count + cost
            WHERE key = rate_key;
            RETURN TRUE;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 5. RPC for Cache Cleanup

CREATE OR REPLACE FUNCTION cleanup_cache()
RETURNS VOID AS $$
BEGIN
    -- Delete expired
    DELETE FROM public.ai_cache WHERE expires_at < now();
    
    -- Cap size at 50,000 (delete oldest)
    DELETE FROM public.ai_cache
    WHERE hash IN (
        SELECT hash FROM public.ai_cache
        ORDER BY created_at DESC
        OFFSET 50000
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant access to service role for tables
GRANT ALL ON public.rate_limits TO service_role;
GRANT ALL ON public.ai_cache TO service_role;
GRANT ALL ON public.device_accounts TO service_role;
GRANT ALL ON public.whitelist_devices TO service_role;
