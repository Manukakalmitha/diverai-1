-- migrations/referrals.sql

-- 1. Create referrals table to track who referred whom
CREATE TABLE IF NOT EXISTS public.referrals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    referrer_id UUID REFERENCES public.profiles(id) NOT NULL,
    referred_user_id UUID REFERENCES public.profiles(id) NOT NULL,
    reward_granted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_referral UNIQUE (referred_user_id)
);

-- 2. Add referral tracking columns to profiles
ALTER TABLE public.profiles 
ADD COLUMN IF NOT EXISTS referral_code TEXT UNIQUE,
ADD COLUMN IF NOT EXISTS referred_by UUID REFERENCES public.profiles(id),
ADD COLUMN IF NOT EXISTS referral_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS pro_expires_at TIMESTAMPTZ;

-- 3. Function to generate a unique referral code
CREATE OR REPLACE FUNCTION generate_referral_code() 
RETURNS TEXT AS $$
DECLARE
    new_code TEXT;
    done BOOLEAN DEFAULT FALSE;
BEGIN
    WHILE NOT done LOOP
        new_code := upper(substring(md5(random()::text) from 1 for 8));
        DONE := NOT EXISTS (SELECT 1 FROM profiles WHERE referral_code = new_code);
    END LOOP;
    RETURN new_code;
END;
$$ LANGUAGE plpgsql;

-- 4. Update handle_new_user to handle referral logic
CREATE OR REPLACE FUNCTION public.handle_new_user() 
RETURNS trigger as $$
DECLARE
    referrer_id UUID;
BEGIN
    -- Get referrer from metadata if exists
    referrer_id := (new.raw_user_meta_data->>'referred_by')::UUID;

    INSERT INTO public.profiles (
        id, 
        email, 
        subscription_tier,
        referral_code,
        referred_by
    )
    VALUES (
        new.id, 
        new.email, 
        CASE WHEN referrer_id IS NOT NULL THEN 'pro' ELSE 'free' END, -- Grant Pro if referred
        generate_referral_code(),
        referrer_id
    );

    -- If there's a referrer, record it and update their profile
    IF referrer_id IS NOT NULL THEN
        INSERT INTO public.referrals (referrer_id, referred_user_id, reward_granted)
        VALUES (referrer_id, new.id, TRUE);

        -- Extend/Grant Pro to referrer (30 days)
        UPDATE public.profiles
        SET 
            subscription_tier = 'pro',
            pro_expires_at = COALESCE(pro_expires_at, NOW()) + INTERVAL '30 days',
            referral_count = referral_count + 1
        WHERE id = referrer_id;
    END IF;

    RETURN new;
END;
$$ language plpgsql security definer;

-- Enable RLS on referrals
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Referrers can view their own referrals." ON public.referrals
    FOR SELECT USING (auth.uid() = referrer_id);
    
CREATE POLICY "Users can view their own referred status." ON public.referrals
    FOR SELECT USING (auth.uid() = referred_user_id);
