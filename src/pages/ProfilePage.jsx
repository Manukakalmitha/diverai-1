import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import {
    User, Shield, Crown, Activity, Target, Clock,
    ArrowUpRight, CreditCard, LogOut, ChevronRight,
    TrendingUp, Zap, Sparkles, Star
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';

export default function ProfilePage() {
    const { user, profile, refreshProfile } = useAppContext();
    const [stats, setStats] = useState({
        totalAnalyses: 0,
        accuracy: 98.4,
        wins: 0,
        losses: 0,
        recentHistory: []
    });
    const [loading, setLoading] = useState(true);
    const [isEditing, setIsEditing] = useState(false);
    const [editData, setEditData] = useState({
        full_name: '',
        phone: '',
        billing_info: {
            address: '',
            city: '',
            zip: ''
        }
    });
    const [uploading, setUploading] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        if (user) {
            fetchUserStats();
            if (profile) {
                setEditData({
                    full_name: profile.full_name || '',
                    phone: profile.phone || '',
                    billing_info: profile.billing_info || { address: '', city: '', zip: '' }
                });
            }
        } else {
            setLoading(false);
        }
    }, [user, profile]);

    const handleAvatarUpload = async (event) => {
        try {
            setUploading(true);
            if (!event.target.files || event.target.files.length === 0) {
                throw new Error('You must select an image to upload.');
            }

            const file = event.target.files[0];
            const fileExt = file.name.split('.').pop();
            const fileName = `${user.id}-${Math.random()}.${fileExt}`;
            const filePath = `${fileName}`;

            const { error: uploadError } = await supabase.storage
                .from('avatars')
                .upload(filePath, file);

            if (uploadError) throw uploadError;

            const { data: { publicUrl } } = supabase.storage
                .from('avatars')
                .getPublicUrl(filePath);

            const { error: updateError } = await supabase
                .from('profiles')
                .update({ avatar_url: publicUrl })
                .eq('id', user.id);

            if (updateError) throw updateError;
            refreshProfile();
            alert('Avatar updated successfully!');
        } catch (error) {
            if (error.message?.includes('bucket not found')) {
                alert('Storage Error: The "avatars" bucket does not exist. Please create it in your Supabase Dashboard -> Storage.');
            } else {
                alert(error.message);
            }
        } finally {
            setUploading(false);
        }
    };

    const handleSaveProfile = async () => {
        try {
            setLoading(true);
            const { error } = await supabase
                .from('profiles')
                .update({
                    full_name: editData.full_name,
                    phone: editData.phone,
                    billing_info: editData.billing_info
                })
                .eq('id', user.id);

            if (error) throw error;
            refreshProfile();
            setIsEditing(false);
            alert('Profile updated successfully!');
        } catch (error) {
            alert(error.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchUserStats = async () => {
        try {
            const { data, error } = await supabase
                .from('prediction_history')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });

            if (error) throw error;

            if (data) {
                const total = data.length;
                const wins = data.filter(item => item.data?.feedback === 'win').length;
                const losses = data.filter(item => item.data?.feedback === 'loss').length;

                // Dynamic accuracy calculation if they have feedback, else default to global baseline
                const dynamicAccuracy = total > 0 && (wins + losses) > 0
                    ? ((wins / (wins + losses)) * 100).toFixed(1)
                    : 98.4;

                setStats({
                    totalAnalyses: total,
                    accuracy: dynamicAccuracy,
                    wins,
                    losses,
                    recentHistory: data.slice(0, 3).map(item => item.data)
                });
            }
        } catch (err) {
            console.error("Error fetching stats:", err);
        } finally {
            setLoading(false);
        }
    };

    if (!user) {
        return (
            <div className="min-h-[80vh] flex flex-col items-center justify-center text-center px-6">
                <div className="w-20 h-20 bg-slate-900 rounded-3xl flex items-center justify-center mb-6 border border-slate-800 shadow-2xl">
                    <User className="w-10 h-10 text-slate-500" />
                </div>
                <h2 className="text-3xl font-black text-white mb-2">Access Restricted</h2>
                <p className="text-slate-500 max-w-sm mb-8 font-bold">Please log in to view your terminal profile and billing architecture.</p>
                <button
                    onClick={() => navigate('/')}
                    className="px-8 py-3 bg-emerald-500 text-slate-950 font-black rounded-xl uppercase tracking-widest text-xs hover:bg-emerald-400 transition-all shadow-xl shadow-emerald-500/20"
                >
                    Return to Base
                </button>
            </div>
        );
    }

    const uploadLimit = profile?.subscription_tier === 'pro' ? '∞' : 3;
    const usagePercent = profile?.subscription_tier === 'pro' ? 100 : Math.min(100, (profile?.upload_count || 0) / 3 * 100);

    return (
        <div className="min-h-screen py-10 md:py-20 px-6 animate-in fade-in duration-700">
            <div className="max-w-5xl mx-auto space-y-10">

                {/* Profile Header */}
                <div className="flex flex-col md:flex-row items-center gap-8 bg-slate-900/40 border border-slate-800 rounded-[40px] p-8 md:p-12 relative overflow-hidden shadow-2xl">
                    <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
                        <User className="w-64 h-64 text-emerald-500" />
                    </div>

                    <div className="relative group">
                        <div className="w-32 h-32 md:w-40 md:h-40 bg-gradient-to-br from-emerald-500 to-blue-600 rounded-[48px] p-1 shadow-2xl overflow-hidden relative">
                            <div className="w-full h-full bg-slate-900 rounded-[44px] flex items-center justify-center overflow-hidden">
                                {profile?.avatar_url ? (
                                    <img src={profile.avatar_url} alt="Avatar" className="w-full h-full object-cover" />
                                ) : (
                                    <User className="w-16 h-16 md:w-20 md:h-20 text-white" />
                                )}
                            </div>
                            {profile?.subscription_tier === 'pro' && (
                                <label className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer flex flex-col items-center justify-center gap-2">
                                    <ArrowUpRight className="w-6 h-6 text-white" />
                                    <span className="text-[10px] font-black uppercase text-white tracking-widest">Update</span>
                                    <input type="file" className="hidden" accept="image/*" onChange={handleAvatarUpload} disabled={uploading} />
                                </label>
                            )}
                        </div>
                        <div className="absolute -bottom-2 -right-2 bg-emerald-500 text-slate-950 p-3 rounded-2xl shadow-xl">
                            <Shield className="w-6 h-6" />
                        </div>
                    </div>

                    <div className="flex-1 text-center md:text-left">
                        <div className="flex flex-col md:flex-row md:items-center gap-3 mb-4">
                            <h1 className="text-3xl md:text-5xl font-black text-white tracking-tighter" title={user.id}>
                                {profile?.full_name || user.email?.split('@')[0]}
                            </h1>
                            <span className={`px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-[0.2em] self-center md:self-auto border ${profile?.subscription_tier === 'pro' ? 'bg-amber-500/10 text-amber-500 border-amber-500/20' : 'bg-slate-800 text-slate-500 border-slate-700'}`}>
                                {profile?.subscription_tier === 'pro' ? 'PRO QUANT' : 'FREE ARCHITECTURE'}
                            </span>
                        </div>
                        <div className="flex flex-wrap justify-center md:justify-start gap-4 text-xs font-mono text-slate-500">
                            <div className="flex items-center gap-2"><Clock className="w-4 h-4" /> Member since {new Date(user.created_at).toLocaleDateString()}</div>
                            <div className="flex items-center gap-2"><CreditCard className="w-4 h-4" /> ID: {user.id.slice(0, 8)}...</div>
                        </div>
                    </div>

                    <div className="flex flex-col gap-3 w-full md:w-auto">
                        <button
                            onClick={async () => {
                                try {
                                    const { openCustomerPortal } = await import('../lib/lemonsqueezy');
                                    const url = await openCustomerPortal();
                                    if (url) window.location.href = url;
                                } catch (err) {
                                    alert("Could not open billing portal: " + err.message);
                                }
                            }}
                            className="px-6 py-3 bg-white text-slate-950 font-black rounded-2xl text-[10px] uppercase tracking-widest hover:bg-slate-200 transition-all flex items-center justify-center gap-2"
                        >
                            <Crown className="w-4 h-4" /> Manage Plan
                        </button>
                        <button
                            onClick={() => { supabase.auth.signOut(); navigate('/'); }}
                            className="px-6 py-3 bg-slate-800 text-slate-400 font-bold rounded-2xl text-[10px] uppercase tracking-widest hover:bg-rose-500/10 hover:text-rose-500 transition-all flex items-center justify-center gap-2"
                        >
                            <LogOut className="w-4 h-4" /> Sign Out
                        </button>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-emerald-500/30 transition-colors group">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 bg-emerald-500/10 rounded-2xl text-emerald-400 group-hover:scale-110 transition-transform"><Activity className="w-6 h-6" /></div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Neural Load</span>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-end justify-between">
                                <div className="text-4xl font-black text-white">{profile?.upload_count || 0}<span className="text-xl text-slate-600 font-medium tracking-tight"> / {uploadLimit}</span></div>
                                <div className="text-[10px] font-bold text-slate-500 mb-1">Daily Capacity</div>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)] transition-all duration-1000"
                                    style={{ width: `${usagePercent}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-blue-500/30 transition-colors group">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 bg-blue-500/10 rounded-2xl text-blue-400 group-hover:scale-110 transition-transform"><Target className="w-6 h-6" /></div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Model Trust</span>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-end justify-between">
                                <div className="text-4xl font-black text-white">{stats.accuracy}<span className="text-xl text-slate-600">%</span></div>
                                <div className="text-[10px] font-bold text-slate-500 mb-1">Avg. Confidence</div>
                            </div>
                            <div className="flex items-center justify-between pt-2">
                                <div className="text-[10px] font-black text-emerald-400 uppercase tracking-widest flex items-center gap-1.5"><TrendingUp className="w-3 h-3" /> {stats.wins} Wins</div>
                                <div className="text-[10px] font-black text-rose-400 uppercase tracking-widest flex items-center gap-1.5">{stats.losses} Losses <LogOut className="w-3 h-3 rotate-180" /></div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-amber-500/30 transition-colors group">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 bg-amber-500/10 rounded-2xl text-amber-500 group-hover:scale-110 transition-transform"><Zap className="w-6 h-6" /></div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Subscription</span>
                        </div>
                        <div className="space-y-4">
                            <div className="text-xl font-black text-white uppercase tracking-tight">
                                {profile?.subscription_tier === 'pro' ? 'Unlimited Pro Access' : 'Trial Architecture'}
                            </div>
                            <p className="text-xs text-slate-500 leading-relaxed font-bold italic">
                                {profile?.subscription_tier === 'pro'
                                    ? "Your neural engine priority is set to Institutional tier with unlimited bandwidth."
                                    : "You are currently running on the public Bayesian baseline. Upgrade for priority OCR."}
                            </p>
                            <button onClick={() => navigate('/pricing')} className="text-amber-400 text-[10px] font-black uppercase tracking-widest flex items-center gap-2 group-hover:gap-3 transition-all">
                                {profile?.subscription_tier === 'pro' ? 'View Benefits' : 'Optimize Engine'} <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Account Settings (Pro Only) */}
                {profile?.subscription_tier === 'pro' && (
                    <div className="bg-slate-900/40 border border-slate-800 rounded-[40px] p-8 md:p-12 shadow-2xl relative overflow-hidden">
                        <div className="flex items-center justify-between mb-10">
                            <div className="flex items-center gap-4">
                                <div className="p-3 bg-emerald-500/10 rounded-2xl text-emerald-400">
                                    <User className="w-6 h-6" />
                                </div>
                                <div>
                                    <h3 className="text-2xl font-black text-white tracking-tight leading-none mb-1">Account Customization</h3>
                                    <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">Premium Intelligence Dashboard</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsEditing(!isEditing)}
                                className={`px-6 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${isEditing ? 'bg-rose-500/10 text-rose-500' : 'bg-emerald-500 text-slate-950'}`}
                            >
                                {isEditing ? 'Cancel Edit' : 'Edit Profile'}
                            </button>
                        </div>

                        {isEditing ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                                <div className="space-y-6">
                                    <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Contact Information</h4>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">Full Name</label>
                                            <input
                                                type="text"
                                                value={editData.full_name}
                                                onChange={(e) => setEditData({ ...editData, full_name: e.target.value })}
                                                placeholder="John Doe"
                                                className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">Phone Number</label>
                                            <input
                                                type="text"
                                                value={editData.phone}
                                                onChange={(e) => setEditData({ ...editData, phone: e.target.value })}
                                                placeholder="+1 (555) 000-0000"
                                                className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all"
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-6">
                                    <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Billing Architecture</h4>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">Street Address</label>
                                            <input
                                                type="text"
                                                value={editData.billing_info.address}
                                                onChange={(e) => setEditData({ ...editData, billing_info: { ...editData.billing_info, address: e.target.value } })}
                                                placeholder="123 Innovation Dr"
                                                className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all"
                                            />
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">City</label>
                                                <input
                                                    type="text"
                                                    value={editData.billing_info.city}
                                                    onChange={(e) => setEditData({ ...editData, billing_info: { ...editData.billing_info, city: e.target.value } })}
                                                    placeholder="Neo Tokyo"
                                                    className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">ZIP Code</label>
                                                <input
                                                    type="text"
                                                    value={editData.billing_info.zip}
                                                    onChange={(e) => setEditData({ ...editData, billing_info: { ...editData.billing_info, zip: e.target.value } })}
                                                    placeholder="10001"
                                                    className="w-full bg-slate-950 border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="md:col-span-2 pt-6">
                                    <button
                                        onClick={handleSaveProfile}
                                        className="w-full py-4 bg-emerald-500 text-slate-950 font-black rounded-2xl shadow-xl shadow-emerald-500/20 hover:bg-emerald-400 transition-all active:scale-95 uppercase tracking-widest text-xs"
                                    >
                                        Deploy Configuration Changes
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Public Identity</p>
                                    <div className="bg-slate-950/30 p-4 rounded-2xl border border-slate-800/50">
                                        <div className="text-white font-black">{profile.full_name || 'Not Configured'}</div>
                                        <div className="text-[10px] text-slate-500 font-bold">{editData.phone || 'No phone recorded'}</div>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Billing Node</p>
                                    <div className="bg-slate-950/30 p-4 rounded-2xl border border-slate-800/50">
                                        <div className="text-white font-black">{editData.billing_info.address || 'Address Hidden'}</div>
                                        <div className="text-[10px] text-slate-500 font-bold">{editData.billing_info.city} {editData.billing_info.zip}</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Recent Benchmarks */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-[32px] overflow-hidden">
                    <div className="px-8 py-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
                        <h3 className="text-lg font-black text-white uppercase tracking-widest flex items-center gap-3">
                            <Activity className="w-5 h-5 text-emerald-400" /> Recent Neural Logs
                        </h3>
                        <button
                            onClick={() => navigate('/analysis')}
                            className="text-[10px] font-black text-slate-500 uppercase tracking-widest hover:text-white transition-colors"
                        >
                            View Full History
                        </button>
                    </div>
                    <div className="p-4 md:p-8">
                        {stats.recentHistory.length === 0 ? (
                            <div className="text-center py-12 space-y-4">
                                <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mx-auto border border-slate-700/50">
                                    <Activity className="w-8 h-8 text-slate-600" />
                                </div>
                                <p className="text-slate-500 text-sm font-bold uppercase tracking-widest">No Recent Activity Detected</p>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {stats.recentHistory.map((item, i) => (
                                    <div key={i} className="flex items-center justify-between p-5 bg-slate-900 border border-slate-800 rounded-2xl group hover:border-slate-700 transition-all cursor-pointer" onClick={() => navigate('/analysis')}>
                                        <div className="flex items-center gap-6">
                                            {item.imageUrl ? (
                                                <div className="w-16 h-12 rounded-lg bg-slate-800 border border-slate-700 overflow-hidden flex-shrink-0">
                                                    <img src={item.imageUrl} alt={item.ticker} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                                                </div>
                                            ) : (
                                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-xs ${item.direction.includes('Bullish') ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                                    {item.ticker.slice(0, 3)}
                                                </div>
                                            )}
                                            <div>
                                                <div className="text-white font-black uppercase tracking-tight mb-1 text-sm">{item.ticker} <span className="text-slate-500 text-[10px] ml-2 font-medium">{item.pattern.name}</span></div>
                                                <div className="text-[10px] text-slate-500 font-mono">{new Date(item.date).toLocaleDateString()} • {item.confidence}% Match</div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <span className={`text-[9px] font-black px-3 py-1 rounded-md uppercase tracking-tighter ${item.feedback === 'win' ? 'bg-emerald-500 text-slate-950 shadow-lg shadow-emerald-500/20' : (item.feedback === 'loss' ? 'bg-rose-500 text-white shadow-lg shadow-rose-500/20' : 'bg-slate-800 text-slate-500')}`}>
                                                {item.feedback || 'Pending'}
                                            </span>
                                            <ArrowUpRight className="w-5 h-5 text-slate-700 group-hover:text-white transition-colors" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
}
