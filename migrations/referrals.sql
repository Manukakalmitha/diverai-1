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

-- 2. Add referral tracking columns to profiles (Safe idempotent check)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='profiles' AND column_name='referral_code') THEN
        ALTER TABLE public.profiles ADD COLUMN referral_code TEXT UNIQUE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='profiles' AND column_name='referred_by') THEN
        ALTER TABLE public.profiles ADD COLUMN referred_by UUID REFERENCES public.profiles(id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='profiles' AND column_name='referral_count') THEN
        ALTER TABLE public.profiles ADD COLUMN referral_count INT DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='profiles' AND column_name='pro_expires_at') THEN
        ALTER TABLE public.profiles ADD COLUMN pro_expires_at TIMESTAMPTZ;
    END IF;
END $$;

-- 3. Function to generate a unique referral code
CREATE OR REPLACE FUNCTION public.generate_referral_code() 
RETURNS TEXT 
LANGUAGE plpgsql
SET search_path = public
AS $$
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
$$;

-- 4. Update handle_new_user to handle referral logic (Robust fail-safe version)
CREATE OR REPLACE FUNCTION public.handle_new_user() 
RETURNS trigger 
LANGUAGE plpgsql 
SECURITY DEFINER SET search_path = public
AS $$
DECLARE
    referrer_id UUID;
    new_referral_code TEXT;
BEGIN
    -- Try generating code, default to null on error
    BEGIN
        new_referral_code := public.generate_referral_code();
    EXCEPTION WHEN OTHERS THEN
        new_referral_code := NULL;
    END;

    -- Try finding referrer, default to null on error
    BEGIN
        IF new.raw_user_meta_data->>'referred_by' IS NOT NULL THEN
            SELECT id INTO referrer_id 
            FROM public.profiles 
            WHERE referral_code = (new.raw_user_meta_data->>'referred_by');
        END IF;
    EXCEPTION WHEN OTHERS THEN
        referrer_id := NULL;
    END;

    -- Main Profile Insert
    INSERT INTO public.profiles (
        id, 
        email, 
        full_name,
        avatar_url,
        subscription_tier,
        referral_code,
        referred_by
    )
    VALUES (
        new.id, 
        new.email, 
        coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'), -- Handle Google/Email metadata differences
        new.raw_user_meta_data->>'avatar_url',
        CASE WHEN referrer_id IS NOT NULL THEN 'pro' ELSE 'free' END, -- Grant Pro if referred
        new_referral_code,
        referrer_id
    );

    -- Handle Referral Logic (Fail Open: Don't block signup if this fails)
    IF referrer_id IS NOT NULL THEN
        BEGIN
            INSERT INTO public.referrals (referrer_id, referred_user_id, reward_granted)
            VALUES (referrer_id, new.id, TRUE);

            -- Extend/Grant Pro to referrer (30 days)
            UPDATE public.profiles
            SET 
                subscription_tier = 'pro',
                pro_expires_at = COALESCE(pro_expires_at, NOW()) + INTERVAL '30 days',
                referral_count = referral_count + 1
            WHERE id = referrer_id;
        EXCEPTION WHEN OTHERS THEN
            -- Log error if possible, but allow transaction to proceed
            NULL; 
        END;
    END IF;

    RETURN new;
END;
$$;

-- Enable RLS on referrals
ALTER TABLE public.referrals ENABLE ROW LEVEL SECURITY;

-- Idempotent policy creation
DROP POLICY IF EXISTS "Referrers can view their own referrals." ON public.referrals;
CREATE POLICY "Referrers can view their own referrals." ON public.referrals
    FOR SELECT USING (auth.uid() = referrer_id);
    
DROP POLICY IF EXISTS "Users can view their own referred status." ON public.referrals;
CREATE POLICY "Users can view their own referred status." ON public.referrals
    FOR SELECT USING (auth.uid() = referred_user_id);
