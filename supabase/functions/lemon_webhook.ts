// supabase/functions/lemon_webhook.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";
import { crypto } from "https://deno.land/std@0.208.0/crypto/mod.ts";

const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const webhookSecret = Deno.env.get("LEMON_SQUEEZY_WEBHOOK_SECRET")!;

Deno.serve(async (req) => {
    const signature = req.headers.get("x-signature");
    const body = await req.text();

    // Verify signature
    const hmac = await crypto.subtle.importKey(
        "raw",
        new TextEncoder().encode(webhookSecret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["verify"]
    );

    const verified = await crypto.subtle.verify(
        "HMAC",
        hmac,
        hexToBytes(signature || ""),
        new TextEncoder().encode(body)
    );

    if (!verified) {
        return new Response("Invalid signature", { status: 401 });
    }

    const payload = JSON.parse(body);
    const eventName = payload.meta.event_name;
    const customData = payload.meta.custom_data;
    const userId = customData?.user_id;

    if (!userId) {
        console.warn("No user_id found in webhook payload");
        return new Response("No user_id", { status: 200 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    if (eventName === "subscription_created" || eventName === "subscription_updated") {
        const status = payload.data.attributes.status;
        const tier = status === "active" ? "pro" : "free";

        const { error } = await supabase
            .from("profiles")
            .update({
                subscription_tier: tier,
                lemon_customer_id: String(payload.data.attributes.customer_id),
                lemon_subscription_id: String(payload.data.id),
            })
            .eq("id", userId);

        if (error) {
            console.error("Error updating profile:", error);
            return new Response("Error updating profile", { status: 500 });
        }
    } else if (eventName === "subscription_cancelled" || eventName === "subscription_expired") {
        await supabase
            .from("profiles")
            .update({ subscription_tier: "free" })
            .eq("id", userId);
    }

    return new Response("Success", { status: 200 });
});

function hexToBytes(hex: string) {
    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < bytes.length; i++) {
        bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes;
}
