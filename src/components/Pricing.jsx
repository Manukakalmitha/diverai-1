import React, { useState } from 'react';
import { Check, Loader2, Zap, Shield, X } from 'lucide-react';
import { supabase } from '../lib/supabase';

const Pricing = ({ user, profile, onClose, onUpgrade }) => {
    const [loading, setLoading] = useState(false);

    const handleUpgrade = async () => {
        if (!user) {
            alert("Please log in to upgrade.");
            return;
        }
        setLoading(true);
        try {
            const { createLemonCheckout } = await import('../lib/lemonsqueezy');
            // Using a placeholder Variant ID. The user will need to update this or provide it.
            const VARIANT_ID = "615964"; // Example variant ID
            const url = await createLemonCheckout(VARIANT_ID);
            if (url) {
                window.location.href = url;
            } else {
                throw new Error("No checkout URL returned");
            }
        } catch (err) {
            console.error("Lemon Squeezy Checkout Error:", err);
            alert("Payment initialization failed: " + (err.message || "Unknown error"));
        } finally {
            setLoading(false);
        }
    };

    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) onClose();
    };

    return (
        <div
            className="w-full flex items-center justify-center p-4 animate-in fade-in duration-300"
        >
            <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-8 relative">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute -top-12 right-0 md:-right-12 p-2 text-slate-400 hover:text-white transition-colors"
                >
                    <X className="w-8 h-8" />
                </button>

                {/* Free Plan */}
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 flex flex-col relative overflow-hidden">
                    <div className="mb-8">
                        <h3 className="text-xl font-bold text-white mb-2">Starter</h3>
                        <div className="text-3xl font-black text-slate-300">Free</div>
                        <p className="text-slate-500 text-sm mt-2">Perfect for casual analysis.</p>
                    </div>
                    <ul className="space-y-4 mb-8 flex-1">
                        <li className="flex gap-3 text-slate-400 text-sm"><Check className="w-5 h-5 text-emerald-500" /> 3 Uploads per day</li>
                        <li className="flex gap-3 text-slate-400 text-sm"><Check className="w-5 h-5 text-emerald-500" /> Core OCR Engine</li>
                        <li className="flex gap-3 text-slate-400 text-sm"><Check className="w-5 h-5 text-emerald-500" /> Basic Live Data</li>
                    </ul>
                    <button
                        onClick={onClose}
                        className="w-full py-3 rounded-xl border border-slate-700 text-white font-bold hover:bg-slate-800 transition-colors"
                    >
                        Continue Free
                    </button>
                </div>

                {/* Pro Plan */}
                <div className="bg-gradient-to-br from-emerald-900/20 to-slate-900 border border-emerald-500/30 rounded-3xl p-8 flex flex-col relative overflow-hidden shadow-2xl shadow-emerald-500/10">
                    <div className="absolute top-0 right-0 bg-emerald-500 text-slate-950 text-xs font-black px-3 py-1 rounded-bl-xl uppercase tracking-wider">Most Popular</div>
                    <div className="mb-8">
                        <h3 className="text-xl font-bold text-emerald-400 mb-2 flex items-center gap-2"><Zap className="w-5 h-5" /> Pro Quant</h3>
                        <div className="text-3xl font-black text-white">$29<span className="text-lg text-slate-500 font-medium">/mo</span></div>
                        <p className="text-emerald-500/60 text-sm mt-2">For serious day traders.</p>
                    </div>
                    <ul className="space-y-4 mb-8 flex-1">
                        <li className="flex gap-3 text-white text-sm"><Check className="w-5 h-5 text-emerald-400" /> <strong>Unlimited</strong> Uploads</li>
                        <li className="flex gap-3 text-white text-sm"><Check className="w-5 h-5 text-emerald-400" /> Priority OCR Processing</li>
                        <li className="flex gap-3 text-white text-sm"><Check className="w-5 h-5 text-emerald-400" /> Institutional Risk Metrics (Sharpe/Vol)</li>
                        <li className="flex gap-3 text-white text-sm"><Check className="w-5 h-5 text-emerald-400" /> <strong>Stealth Mode</strong> (Incognito Analysis)</li>
                        <li className="flex gap-3 text-white text-sm"><Check className="w-5 h-5 text-emerald-400" /> PDF / CSV Report Export</li>
                    </ul>
                    <button
                        onClick={handleUpgrade}
                        disabled={loading || (profile?.subscription_tier === 'pro')}
                        className="w-full py-3 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : (profile?.subscription_tier === 'pro' ? 'Current Plan' : 'Upgrade Now')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Pricing;
