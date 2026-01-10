// supabase/functions/lemon_checkout.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";

const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
};

Deno.serve(async (req) => {
    if (req.method === "OPTIONS") {
        return new Response("ok", { headers: corsHeaders });
    }

    try {
        const authHeader = req.headers.get("Authorization");
        if (!authHeader) {
            return new Response(JSON.stringify({ error: "No authorization header" }), {
                status: 401,
                headers: { ...corsHeaders, "Content-Type": "application/json" }
            });
        }

        const supabase = createClient(
            Deno.env.get("SUPABASE_URL") ?? "",
            Deno.env.get("SUPABASE_ANON_KEY") ?? "",
            { global: { headers: { Authorization: authHeader } } }
        );

        const { data: { user }, error: authError } = await supabase.auth.getUser();
        if (authError || !user) {
            return new Response(JSON.stringify({ error: "Invalid token" }), {
                status: 401,
                headers: { ...corsHeaders, "Content-Type": "application/json" }
            });
        }

        const body = await req.json();
        const { variantId, portal } = body;

        const apiKey = Deno.env.get("LEMON_SQUEEZY_API_KEY");
        if (!apiKey) {
            throw new Error("Lemon Squeezy API key not configured");
        }

        if (portal) {
            // For Lemon Squeezy, we usually redirect to the 'My Orders' portal or a custom link.
            // If they have a customer ID, we might be able to get a specific link, 
            // but usually, a generic 'https://app.lemonsqueezy.com/my-orders' works if they are logged in there.
            // Or we can just return the store URL.
            return new Response(JSON.stringify({ url: "https://app.lemonsqueezy.com/my-orders" }), {
                status: 200,
                headers: { ...corsHeaders, "Content-Type": "application/json" },
            });
        }

        if (!variantId) {
            throw new Error("Variant ID is required");
        }

        // Create a checkout via Lemon Squeezy API
        const response = await fetch("https://api.lemonsqueezy.com/v1/checkouts", {
            method: "POST",
            headers: {
                "Accept": "application/vnd.api+json",
                "Content-Type": "application/vnd.api+json",
                "Authorization": `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                data: {
                    type: "checkouts",
                    attributes: {
                        checkout_data: {
                            email: user.email,
                            custom: {
                                user_id: user.id
                            }
                        }
                    },
                    relationships: {
                        store: {
                            data: {
                                type: "stores",
                                id: Deno.env.get("LEMON_SQUEEZY_STORE_ID")
                            }
                        },
                        variant: {
                            data: {
                                type: "variants",
                                id: String(variantId)
                            }
                        }
                    }
                }
            })
        });

        const checkoutData = await response.json();
        if (!response.ok) {
            console.error("Lemon Squeezy API Error:", checkoutData);
            throw new Error(checkoutData.errors?.[0]?.detail || "Failed to create checkout");
        }

        return new Response(JSON.stringify({ url: checkoutData.data.attributes.url }), {
            status: 200,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
    } catch (err) {
        return new Response(JSON.stringify({ error: err.message }), {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
    }
});
