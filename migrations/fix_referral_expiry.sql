-- migrations/fix_referral_expiry.sql

-- 1. Update handle_new_user to set pro_expires_at for referees
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
        pro_expires_at, -- Added this column to insert
        referral_code,
        referred_by
    )
    VALUES (
        new.id, 
        new.email, 
        coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'),
        new.raw_user_meta_data->>'avatar_url',
        CASE WHEN referrer_id IS NOT NULL THEN 'pro' ELSE 'free' END,
        CASE WHEN referrer_id IS NOT NULL THEN NOW() + INTERVAL '30 days' ELSE NULL END, -- Grant 30 days to referee
        new_referral_code,
        referrer_id
    );

    -- Handle Referral Logic for Referrer
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
            NULL; 
        END;
    END IF;

    RETURN new;
END;
$$;
