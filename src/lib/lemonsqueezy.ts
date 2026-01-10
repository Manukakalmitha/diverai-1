// src/lib/lemonsqueezy.ts
import { supabase } from "./supabase";

/**
 * Initiates a Lemon Squeezy Checkout Session for the current user.
 * Returns the URL to which the frontend should redirect.
 * @param {string} variantId - The variant ID from Lemon Squeezy.
 */
export async function createLemonCheckout(variantId: string) {
    const { data, error } = await supabase.functions.invoke("lemon_checkout", {
        body: { variantId },
    });

    if (error) {
        console.error("Lemon Squeezy Checkout Error:", error);
        throw error;
    }

    // The edge function should return { url: "https://app.lemonsqueezy.com/checkout/..." }
    return data.url;
}

/** 
 * Redirects user to Lemon Squeezy Customer Portal 
 * (Note: Lemon Squeezy portal is usually a direct link or handled via Checkout link)
 * We can implement a specific portal link if needed, but often managing via checkout works too.
 */
export async function openCustomerPortal() {
    const { data, error } = await supabase.functions.invoke("lemon_checkout", {
        body: { portal: true },
    });

    if (error) {
        console.error("Lemon Squeezy Portal Error:", error);
        throw error;
    }

    return data.url;
}
