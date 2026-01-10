// supabase/functions/payhere_webhook.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";
import { crypto } from "https://deno.land/std@0.203.0/crypto/mod.ts";
import { encodeHex } from "https://deno.land/std@0.203.0/encoding/hex.ts";

const corsHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

async function md5(text: string) {
    const msgUint8 = new TextEncoder().encode(text);
    const hashBuffer = await crypto.subtle.digest("MD5", msgUint8);
    return encodeHex(hashBuffer).toUpperCase();
}

Deno.serve(async (req) => {
    // Handle CORS
    if (req.method === "OPTIONS") {
        return new Response("ok", { headers: corsHeaders });
    }

    try {
        const formData = await req.formData();
        const data: Record<string, string> = {};
        formData.forEach((value, key) => {
            data[key] = value.toString();
        });

        console.log("PayHere Webhook Received:", data);

        const merchantId = data.merchant_id;
        const orderId = data.order_id;
        const payhereAmount = data.payhere_amount;
        const payhereCurrency = data.payhere_currency;
        const statusCode = data.status_code;
        const md5sigReceived = data.md5sig;
        const userId = data.custom_1;

        const merchantSecret = Deno.env.get("PAYHERE_MERCHANT_SECRET");
        if (!merchantSecret) {
            throw new Error("PAYHERE_MERCHANT_SECRET not configured");
        }

        // Verify Signature
        // md5sig = upper(md5(merchant_id + order_id + payhere_amount + payhere_currency + status_code + upper(md5(merchant_secret))))
        const hashedSecret = await md5(merchantSecret);
        const signatureSource = merchantId + orderId + payhereAmount + payhereCurrency + statusCode + hashedSecret;
        const expectedSig = await md5(signatureSource);

        if (md5sigReceived !== expectedSig) {
            console.error("Signature mismatch!", { received: md5sigReceived, expected: expectedSig });
            // For security, even if signature fails, we return a 200 to prevent leaking info to attackers, 
            // but we log it and don't process.
            return new Response("invalid_signature", { status: 400 });
        }

        // Check if payment was successful (Status 2 = Success)
        if (statusCode === "2") {
            const supabase = createClient(
                Deno.env.get("SUPABASE_URL") ?? "",
                Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "" // Use service role to bypass RLS
            );

            console.log(`Upgrading user ${userId} to pro...`);

            const { error: updateError } = await supabase
                .from("profiles")
                .update({
                    subscription_tier: "pro",
                    updated_at: new Date().toISOString()
                })
                .eq("id", userId);

            if (updateError) {
                console.error("Database update error:", updateError);
                throw updateError;
            }

            console.log("Subscription updated successfully.");
        } else {
            console.log(`Payment status: ${statusCode}. No action taken.`);
        }

        return new Response("ok", {
            status: 200,
            headers: { ...corsHeaders, "Content-Type": "text/plain" }
        });

    } catch (err) {
        console.error("Webhook Processing Error:", err.message);
        return new Response(err.message, { status: 500 });
    }
});
