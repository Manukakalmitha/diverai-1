import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
    console.error('Missing Supabase Environment Variables!');
}

// Custom storage handler for Chrome Extensions to ensure persistence
const customStorage = {
    getItem: (key) => {
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            return new Promise((resolve) => {
                chrome.storage.local.get([key], (result) => resolve(result[key] || null));
            });
        }
        return localStorage.getItem(key);
    },
    setItem: (key, value) => {
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ [key]: value });
        } else {
            localStorage.setItem(key, value);
        }
    },
    removeItem: (key) => {
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.remove([key]);
        } else {
            localStorage.removeItem(key);
        }
    },
};

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
        storage: customStorage,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
    }
});

/**
 * Update the user's subscription tier and Stripe IDs in the profiles table.
 * @param {string} userId - Supabase auth UID.
 * @param {string} tier - 'free' or 'pro'.
 * @param {string} stripeCustomerId - Stripe customer identifier.
 * @param {string} stripeSubscriptionId - Stripe subscription identifier.
 */
export async function updateSubscription(userId, tier, stripeCustomerId, stripeSubscriptionId) {
    const { error } = await supabase
        .from('profiles')
        .update({
            subscription_tier: tier,
            stripe_customer_id: stripeCustomerId,
            stripe_subscription_id: stripeSubscriptionId,
        })
        .eq('id', userId);
    if (error) throw error;
    return true;
}

/** Retrieve subscription info for a user */
export async function getSubscriptionInfo(userId) {
    const { data, error } = await supabase
        .from('profiles')
        .select('subscription_tier, stripe_customer_id, stripe_subscription_id, billing_info')
        .eq('id', userId)
        .single();
    if (error) throw error;
    return data;
}
