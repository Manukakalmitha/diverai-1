import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// Helper for SHA-256
async function sha256(message: string) {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
}

serve(async (req) => {
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    try {
        // 1. Initialization
        const authHeader = req.headers.get('Authorization') ?? '';
        const apiKeyHeader = req.headers.get('apikey') ?? '';
        const token = authHeader.replace('Bearer ', '').trim();

        // Initialize Supabase Client
        const supabaseClient = createClient(
            // @ts-ignore
            Deno.env.get('SUPABASE_URL') ?? '',
            // @ts-ignore
            Deno.env.get('SUPABASE_ANON_KEY') ?? '',
            { global: { headers: { Authorization: authHeader } } }
        )

        // 2. Authentication (STRICT: No Guest Access)
        const { data: { user }, error: authError } = await supabaseClient.auth.getUser();

        if (authError || !user) {
            console.error("[Access] Blocked: Unauthorized");
            return new Response(JSON.stringify({ error: 'Unauthorized', code: 'AUTH_REQUIRED' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                status: 401,
            });
        }

        console.log(`[Flow] User Authenticated: ${user.email}`);

        // 3. Restriction Check (Soft Device Limit)
        const { data: profile } = await supabaseClient
            .from('profiles')
            .select('is_restricted')
            .eq('id', user.id)
            .single();

        if (profile?.is_restricted) {
            console.warn(`[Access] Blocked: Restricted User ${user.email}`);
            return new Response(JSON.stringify({ error: 'Account restricted. Contact support.', code: 'ACCOUNT_RESTRICTED' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                status: 403,
            });
        }

        // 4. Rate Limiting (Rolling Window)
        // 10 requests per minute
        const { data: allowed, error: rateError } = await supabaseClient.rpc('check_rate_limit', {
            user_id: user.id,
            endpoint: 'detect_ticker',
            cost: 1,
            limit_count: 10,
            window_seconds: 60
        });

        if (rateError) {
            console.error("Rate limit RPC error:", rateError);
            // Fail open or closed? Fail open to avoid blocking users on system error, or closed for security.
            // Faking fail closed for safety.
        }

        if (!allowed) {
            console.warn(`[RateLimit] User ${user.email} exceeded limit.`);
            return new Response(JSON.stringify({ error: 'Rate limit exceeded', code: 'RATE_LIMIT_EXCEEDED' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                status: 429,
            });
        }

        const body = await req.json().catch(() => ({}));
        const { image } = body; // base64 data url

        if (!image) { throw new Error("No image data provided"); }

        // 5. Caching Strategy
        const base64Content = image.split(',')[1] || image;
        const imageHash = await sha256(base64Content);

        // Check Cache
        const { data: cached } = await supabaseClient
            .from('ai_cache')
            .select('response')
            .eq('hash', imageHash)
            .gt('expires_at', new Date().toISOString())
            .single();

        if (cached?.response) {
            console.log("[Cache] Hit");
            return new Response(JSON.stringify(cached.response), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            });
        }

        console.log("[Cache] Miss - Calling OCR");

        // 6. Processing (OCR)
        const formData = new FormData();
        formData.append('base64Image', `data:image/jpg;base64,${base64Content}`);
        formData.append('language', 'eng');
        formData.append('isOverlayRequired', 'false');
        formData.append('filetype', 'JPG');

        const apiKey = 'helloworld'; // In prod, use env var

        const ocrResponse = await fetch('https://api.ocr.space/parse/image', {
            method: 'POST',
            headers: { 'apikey': apiKey },
            body: formData
        });

        const ocrResult = await ocrResponse.json();

        if (ocrResult.IsErroredOnProcessing) {
            throw new Error(ocrResult.ErrorMessage?.[0] || "OCR.space error");
        }

        const text = ocrResult.ParsedResults?.[0]?.ParsedText || "";
        const responseData = { text };

        // 7. Store in Cache (TTL 24h)
        const expiry = new Date();
        expiry.setHours(expiry.getHours() + 24);

        await supabaseClient.from('ai_cache').upsert({
            hash: imageHash,
            response: responseData,
            created_at: new Date().toISOString(),
            expires_at: expiry.toISOString()
        });

        // Probabilistic Cleanup (10% chance)
        if (Math.random() < 0.1) {
            // Fire and forget
            supabaseClient.rpc('cleanup_cache').then(({ error }) => {
                if (error) console.error("Cache cleanup error:", error);
            });
        }

        return new Response(
            JSON.stringify(responseData),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )

    } catch (error) {
        console.error("Cloud OCR Error:", error);
        return new Response(
            JSON.stringify({ error: (error as any).message }),
            { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 400 }
        )
    }
})
