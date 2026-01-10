import React, { useState } from 'react';
import { Check, Loader2, Zap, Shield, X, ArrowRight } from 'lucide-react';
import { supabase } from '../lib/supabase';
import GooglePayButton from './GooglePayButton';

const Pricing = ({ user, profile, onClose, onUpgrade }) => {
    const [loading, setLoading] = useState(false);
    const [payMethod, setPayMethod] = useState(null); // 'gpay' or 'card'

    const PAYHERE_LINK = "https://payhere.lk/pay/o8f8b823";

    const handleUpgrade = async (method = 'card') => {
        if (!user) {
            alert("Please log in to upgrade.");
            return;
        }
        setLoading(true);
        setPayMethod(method);

        try {
            // Since this is a direct PayHere link, we simply redirect to it.
            // If you want to track which user paid, you can append their ID:
            const redirectUrl = `${PAYHERE_LINK}?custom_1=${user.id}`;

            // Give a tiny delay for the "loading" animation effect
            setTimeout(() => {
                window.open(redirectUrl, '_blank');
                setLoading(false);
                setPayMethod(null);
            }, 800);
        } catch (err) {
            console.error("Payment Redirect Error:", err);
            alert("Payment initialization failed. Please try again.");
            setLoading(false);
            setPayMethod(null);
        }
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
                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 flex flex-col relative overflow-hidden group">
                    <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                    <div className="mb-8 relative z-10">
                        <h3 className="text-xl font-bold text-white mb-2">Starter</h3>
                        <div className="text-4xl font-black text-white">Free</div>
                        <p className="text-slate-500 text-sm mt-3 font-medium">Perfect for exploring the bayesian core.</p>
                    </div>
                    <ul className="space-y-4 mb-10 flex-1 relative z-10">
                        <li className="flex gap-3 text-slate-400 text-sm font-medium"><Check className="w-5 h-5 text-emerald-500 shrink-0" /> 3 Neural scans per day</li>
                        <li className="flex gap-3 text-slate-400 text-sm font-medium"><Check className="w-5 h-5 text-emerald-500 shrink-0" /> Core Optical Engine</li>
                        <li className="flex gap-3 text-slate-400 text-sm font-medium"><Check className="w-5 h-5 text-emerald-500 shrink-0" /> Standard Latency</li>
                    </ul>
                    <button
                        onClick={onClose}
                        className="w-full py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold hover:bg-slate-800 hover:text-white transition-all text-sm uppercase tracking-widest"
                    >
                        Current Architecture
                    </button>
                </div>

                {/* Pro Plan */}
                <div className="bg-slate-950 border border-emerald-500/30 rounded-[32px] p-8 flex flex-col relative overflow-hidden shadow-2xl shadow-emerald-500/10 group">
                    <div className="absolute top-0 right-0 bg-emerald-500 text-slate-950 text-[10px] font-black px-4 py-1.5 rounded-bl-2xl uppercase tracking-[0.2em] z-20">Priority Access</div>

                    {/* Animated background flare */}
                    <div className="absolute -top-24 -right-24 w-64 h-64 bg-emerald-500/10 blur-[100px] rounded-full pointer-events-none"></div>

                    <div className="mb-8 relative z-10">
                        <h3 className="text-xl font-bold text-emerald-400 mb-2 flex items-center gap-2 tracking-tight">
                            <Zap className="w-6 h-6 fill-current" /> Pro Quant
                        </h3>
                        <div className="flex items-baseline gap-1">
                            <div className="text-5xl font-black text-white tracking-tighter">$29</div>
                            <div className="text-slate-500 font-bold text-lg uppercase tracking-widest">/mo</div>
                        </div>
                        <p className="text-slate-400 text-sm mt-3 font-medium">Unlocked neural bandwidth for desk traders.</p>
                    </div>

                    <ul className="space-y-4 mb-10 flex-1 relative z-10">
                        <li className="flex gap-3 text-white text-sm font-bold"><Check className="w-5 h-5 text-emerald-400 shrink-0" /> Unlimited Neural Analysis</li>
                        <li className="flex gap-3 text-white text-sm font-medium opacity-80"><Check className="w-5 h-5 text-emerald-400 shrink-0" /> Institutional OCR Priority</li>
                        <li className="flex gap-3 text-white text-sm font-medium opacity-80"><Check className="w-5 h-5 text-emerald-400 shrink-0" /> Pattern Recognition Alpha</li>
                        <li className="flex gap-3 text-white text-sm font-medium opacity-80"><Check className="w-5 h-5 text-emerald-400 shrink-0" /> Multi-Timeframe Confluence</li>
                        <li className="flex gap-3 text-white text-sm font-medium opacity-80"><Check className="w-5 h-5 text-emerald-400 shrink-0" /> PDF Intelligence Reports</li>
                    </ul>

                    <div className="space-y-4 relative z-10">
                        <GooglePayButton
                            onClick={() => handleUpgrade('gpay')}
                            disabled={loading || (profile?.subscription_tier === 'pro')}
                        />

                        <button
                            onClick={() => handleUpgrade('card')}
                            disabled={loading || (profile?.subscription_tier === 'pro')}
                            className="w-full py-4 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-xl shadow-emerald-500/20 text-xs uppercase tracking-widest group/btn active:scale-95"
                        >
                            {loading && payMethod === 'card' ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <>
                                    {profile?.subscription_tier === 'pro' ? 'Active Subscription' : 'Upgrade with Card'}
                                    <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                                </>
                            )}
                        </button>
                    </div>

                    <p className="text-[10px] text-slate-500 font-bold text-center mt-6 uppercase tracking-widest opacity-60">
                        Secure SSL Encryption • Powered by PayHere
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Pricing;
