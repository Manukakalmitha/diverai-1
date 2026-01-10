import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

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
        // 1. Diagnostics & Initialization
        const authHeader = req.headers.get('Authorization') ?? '';
        const apiKeyHeader = req.headers.get('apikey') ?? '';
        const token = authHeader.replace('Bearer ', '').trim();

        console.log("[Diagnostics] Incoming Metadata:", {
            method: req.method,
            hasAuth: !!authHeader,
            hasApiKey: !!apiKeyHeader,
            isGuestCandidate: token === apiKeyHeader
        });

        // Initialize Supabase Client
        // Note: For guests, we don't pass the Authorization header to avoid getUser() failing
        const clientConfig = (token && token !== apiKeyHeader)
            ? { global: { headers: { Authorization: authHeader } } }
            : {};

        const supabaseClient = createClient(
            // @ts-ignore
            Deno.env.get('SUPABASE_URL') ?? '',
            // @ts-ignore
            Deno.env.get('SUPABASE_ANON_KEY') ?? '',
            clientConfig
        )

        // 2. Authentication Protocol
        let user = null;
        let isGuest = false;

        if (token && token !== apiKeyHeader) {
            try {
                const { data, error: authError } = await supabaseClient.auth.getUser();
                if (authError) {
                    console.warn("[Auth] Verification Failed:", authError.message);
                    // If it's a Malformed JWT or similar, we might still allow it if it matches ANON_KEY
                    if (token === Deno.env.get('SUPABASE_ANON_KEY')) {
                        isGuest = true;
                    } else {
                        return new Response(JSON.stringify({ error: 'Unauthorized', details: authError.message }), {
                            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                            status: 401,
                        });
                    }
                } else {
                    user = data?.user;
                }
            } catch (err) {
                console.error("[Auth] Critical Failure:", err);
            }
        } else {
            isGuest = true;
        }

        if (!user && !isGuest) {
            console.error("[Access] Blocked: Invalid Security Context");
            return new Response(JSON.stringify({ error: 'Unauthorized', code: 'SEC_INVALID' }), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                status: 401,
            });
        }

        console.log(`[Flow] Node: ${user ? `Authenticated(${user.email})` : 'Guest/Anon'}`);

        const body = await req.json().catch(() => ({}));
        const { image } = body; // base64 data url

        if (!image) { throw new Error("No image data provided"); }

        // Plan C: Using OCR.space API (Fast, Reliable, No Workers)
        // We strip the "data:image/jpeg;base64," part
        const base64Content = image.split(',')[1];

        const formData = new FormData();
        formData.append('base64Image', `data:image/jpg;base64,${base64Content}`);
        formData.append('language', 'eng');
        formData.append('isOverlayRequired', 'false');
        formData.append('filetype', 'JPG');

        // Using K88890778488957 (Demo Key) or 'helloworld' - helloworld is default for free
        // For production, the user should get their own free key from ocr.space
        const apiKey = 'helloworld';

        console.log("Calling OCR.space API...");
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
        console.log("OCR Extracted Text:", text.substring(0, 50) + "...");

        return new Response(
            JSON.stringify({ text }),
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
