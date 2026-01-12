import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { supabase } from '../lib/supabase';
import {
    Users,
    Gift,
    Link as LinkIcon,
    Copy,
    Check,
    Zap,
    ChevronRight,
    Share2,
    Trophy,
    ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';

const ReferralPage = () => {
    const { user, profile } = useAppContext();
    const [referrals, setReferrals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [copied, setCopied] = useState(false);

    useEffect(() => {
        const fetchReferralData = async () => {
            if (!user) return;
            try {
                const { data, error } = await supabase
                    .from('referrals')
                    .select('*, referred_user:referred_user_id(email)')
                    .eq('referrer_id', user.id);

                if (error) throw error;
                setReferrals(data || []);
            } catch (err) {
                console.error("Referral fetch error:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchReferralData();
    }, [user]);

    const referralLink = profile?.referral_code
        ? `${window.location.origin}/signup?ref=${profile.referral_code}`
        : '';

    const handleCopy = () => {
        navigator.clipboard.writeText(referralLink);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white p-6 md:p-12 lg:p-24">
            <div className="max-w-6xl mx-auto space-y-12">

                {/* Hero Section */}
                <div className="relative overflow-hidden rounded-[40px] bg-gradient-to-br from-blue-600 to-indigo-700 p-8 md:p-16 border border-white/10 shadow-2xl">
                    <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
                    <div className="relative z-10 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                        <div className="space-y-8">
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-xs font-bold uppercase tracking-widest text-white backdrop-blur-md">
                                <Trophy className="w-3 h-3 text-yellow-400" /> Referral Rewards Program
                            </div>
                            <h1 className="text-4xl md:text-6xl font-black tracking-tighter leading-none">
                                GROW THE NETWORK.<br />
                                <span className="text-blue-100">CLAIM PRO STATUS.</span>
                            </h1>
                            <p className="text-lg text-blue-100/80 font-medium max-w-md leading-relaxed">
                                Give your friends 30 days of Pro analysis. Get 30 days of Pro for every successful referral. No limits.
                            </p>
                        </div>

                        <div className="bg-slate-950/40 backdrop-blur-2xl rounded-3xl p-8 border border-white/10 space-y-6">
                            <div className="space-y-2">
                                <label className="text-[10px] font-black text-blue-200 uppercase tracking-widest">Your Referral Link</label>
                                <div className="flex gap-3">
                                    <div className="flex-1 bg-slate-900 border border-white/10 rounded-xl px-4 py-3 font-mono text-sm text-blue-400 overflow-hidden truncate">
                                        {profile?.referral_code ? referralLink : 'Initializing Code...'}
                                    </div>
                                    <button
                                        onClick={handleCopy}
                                        className="bg-white text-slate-950 p-3 rounded-xl hover:bg-blue-200 transition-all shadow-xl shadow-white/5 active:scale-95"
                                    >
                                        {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                                    </button>
                                </div>
                            </div>
                            <button className="w-full bg-blue-500 hover:bg-blue-400 text-white font-black py-4 rounded-xl transition-all shadow-xl shadow-blue-500/20 uppercase tracking-widest text-xs flex items-center justify-center gap-2">
                                <Share2 className="w-4 h-4" /> Share with Traders
                            </button>
                        </div>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:bg-slate-900 transition-all group">
                        <div className="w-12 h-12 bg-blue-500/20 rounded-2xl flex items-center justify-center mb-6 border border-blue-500/20 group-hover:scale-110 transition-transform">
                            <Users className="w-6 h-6 text-blue-400" />
                        </div>
                        <div className="text-4xl font-black text-white mb-2">{profile?.referral_count || 0}</div>
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">Active Referrals</div>
                    </div>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:bg-slate-900 transition-all group">
                        <div className="w-12 h-12 bg-emerald-500/20 rounded-2xl flex items-center justify-center mb-6 border border-emerald-500/20 group-hover:scale-110 transition-transform">
                            <Zap className="w-6 h-6 text-emerald-400" />
                        </div>
                        <div className="text-4xl font-black text-white mb-2">{(profile?.referral_count || 0) * 30}</div>
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">Days of Pro Gained</div>
                    </div>
                    <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 hover:bg-slate-900 transition-all group">
                        <div className="w-12 h-12 bg-purple-500/20 rounded-2xl flex items-center justify-center mb-6 border border-purple-500/20 group-hover:scale-110 transition-transform">
                            <Gift className="w-6 h-6 text-purple-400" />
                        </div>
                        <div className="text-4xl font-black text-white mb-2">30D</div>
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">Reward Per Friend</div>
                    </div>
                </div>

                {/* Recent Referrals */}
                <div className="space-y-6">
                    <h3 className="text-2xl font-black text-white tracking-tight uppercase">Recent Activity</h3>
                    <div className="bg-slate-900/30 border border-slate-800 rounded-3xl overflow-hidden">
                        {loading ? (
                            <div className="p-12 text-center text-slate-500 animate-pulse uppercase tracking-[0.2em] font-black italic">Syncing Network Nodes...</div>
                        ) : referrals.length > 0 ? (
                            <table className="w-full text-left">
                                <thead className="bg-slate-900/50 border-b border-slate-800">
                                    <tr>
                                        <th className="px-8 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Referree</th>
                                        <th className="px-8 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Status</th>
                                        <th className="px-8 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Date</th>
                                        <th className="px-8 py-4 text-[10px] font-black text-slate-500 uppercase tracking-widest text-right">Reward</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/50">
                                    {referrals.map((ref) => (
                                        <tr key={ref.id} className="hover:bg-slate-900/50 transition-colors">
                                            <td className="px-8 py-4">
                                                <div className="text-sm font-bold text-white">{ref.referred_user?.email || 'Anonymous'}</div>
                                            </td>
                                            <td className="px-8 py-4">
                                                <div className="inline-flex items-center gap-2 px-2 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-[10px] font-bold text-emerald-400 uppercase tracking-wider">
                                                    <Check className="w-3 h-3" /> Verified
                                                </div>
                                            </td>
                                            <td className="px-8 py-4 text-xs text-slate-500 font-mono">
                                                {new Date(ref.created_at).toLocaleDateString()}
                                            </td>
                                            <td className="px-8 py-4 text-right">
                                                <div className="text-sm font-black text-blue-400">+30D PRO</div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <div className="p-16 text-center space-y-4 opacity-50">
                                <Users className="w-12 h-12 text-slate-600 mx-auto" />
                                <p className="text-slate-500 font-bold max-w-xs mx-auto">No referrals detected yet. Start sharing your link to gain rewards.</p>
                                <button onClick={handleCopy} className="text-blue-500 hover:text-blue-400 font-black uppercase text-xs tracking-widest">Copy link now <ArrowRight className="w-3 h-3 inline ml-1" /></button>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer Disclaimer */}
                <div className="pt-12 border-t border-slate-900 text-center">
                    <p className="text-[10px] text-slate-600 font-bold uppercase tracking-[0.3em] leading-relaxed">
                        Terms of Service apply. Rewards are granted upon email verification of the referred account.<br />
                        Abuse of the referral system will lead to immediate account termination.
                    </p>
                </div>

            </div>
        </div>
    );
};

export default ReferralPage;
