// supabase/functions/stripe_webhook.ts
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.0";
import Stripe from "npm:stripe@14.12.0";

Deno.serve(async (req) => {
    const signature = req.headers.get("stripe-signature");
    if (!signature) return new Response("No signature", { status: 400 });

    try {
        const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "", {
            apiVersion: "2023-10-16",
        });
        const supabase = createClient(
            Deno.env.get("SUPABASE_URL") ?? "",
            Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? ""
        );

        const body = await req.text();
        const event = await stripe.webhooks.constructEventAsync(
            body,
            signature,
            Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? ""
        );

        switch (event.type) {
            case "checkout.session.completed": {
                const session = event.data.object;
                const userId = session.metadata?.user_id;
                const subscriptionId = session.subscription;

                if (userId) {
                    await supabase
                        .from("profiles")
                        .update({
                            subscription_tier: "pro",
                            subscription_status: "active",
                            stripe_subscription_id: subscriptionId,
                            stripe_customer_id: session.customer
                        })
                        .eq("id", userId);
                }
                break;
            }

            case "customer.subscription.updated":
            case "customer.subscription.deleted": {
                const subscription = event.data.object;
                const status = subscription.status;
                const tier = status === "active" ? "pro" : "free";

                await supabase
                    .from("profiles")
                    .update({
                        subscription_tier: tier,
                        subscription_status: status,
                        current_period_end: new Date(subscription.current_period_end * 1000).toISOString()
                    })
                    .eq("stripe_subscription_id", subscription.id);
                break;
            }
        }

        return new Response(JSON.stringify({ ok: true }), {
            status: 200,
            headers: { "Content-Type": "application/json" }
        });
    } catch (err) {
        console.error(`Webhook Error: ${err.message}`);
        return new Response(`Webhook Error: ${err.message}`, { status: 400 });
    }
});
