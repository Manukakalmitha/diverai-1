import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import {
    User, Shield, Crown, Activity, Target, Clock,
    ArrowUpRight, CreditCard, LogOut, ChevronRight,
    TrendingUp, Zap, Sparkles, Star
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

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
                <div className="w-20 h-20 bg-black-ash rounded-3xl flex items-center justify-center mb-6 border border-slate-800 shadow-2xl">
                    <User className="w-10 h-10 text-slate-500" />
                </div>
                <h2 className="text-3xl font-black text-white mb-2">Access Restricted</h2>
                <p className="text-slate-500 max-w-sm mb-8 font-bold">Please log in to view your terminal profile and billing architecture.</p>
                <button
                    onClick={() => navigate('/')}
                    className="btn-flame px-8 !py-3"
                >
                    Return to Base
                </button>
            </div>
        );
    }

    const uploadLimit = profile?.subscription_tier === 'pro' ? '∞' : 3;
    const usagePercent = profile?.subscription_tier === 'pro' ? 100 : Math.min(100, (profile?.upload_count || 0) / 3 * 100);

    return (
        <div className="min-h-screen bg-black py-10 md:py-20 px-6 animate-in fade-in duration-700">
            <Helmet>
                <title>User Profile | Diver AI - Trading Architecture</title>
                <meta name="description" content="Manage your Diver AI profile, billing, and neural engine settings. View your trade accuracy and history." />
            </Helmet>
            <div className="max-w-5xl mx-auto space-y-10">

                {/* Profile Header */}
                <div className="flex flex-col md:flex-row items-center gap-6 md:gap-8 bg-black-ash/40 border border-slate-800 rounded-[32px] md:rounded-[40px] p-6 md:p-12 relative overflow-hidden shadow-2xl">
                    <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none hidden md:block">
                        <User className="w-64 h-64 text-brand" />
                    </div>

                    <div className="relative group">
                        <div className="w-32 h-32 md:w-40 md:h-40 bg-gradient-to-br from-brand to-brand-dark rounded-[48px] p-1 shadow-2xl overflow-hidden relative">
                            <div className="w-full h-full bg-black-ash rounded-[44px] flex items-center justify-center overflow-hidden">
                                {profile?.avatar_url ? (
                                    <img src={profile.avatar_url} alt="Avatar" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
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
                        <div className="absolute -bottom-2 -right-2 bg-brand text-slate-950 p-3 rounded-2xl shadow-xl">
                            <Shield className="w-6 h-6" />
                        </div>
                    </div>

                    <div className="flex-1 text-center md:text-left min-w-0">
                        <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 mb-3 md:mb-4">
                            <h1 className="text-2xl md:text-5xl font-black text-white tracking-tighter truncate" title={user.id}>
                                {profile?.full_name || user.email?.split('@')[0]}
                            </h1>
                            <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-[0.2em] self-center md:self-auto border shrink-0 ${profile?.subscription_tier === 'pro' ? 'bg-amber-500/10 text-amber-500 border-amber-500/20' : 'bg-slate-800 text-slate-500 border-slate-700'}`}>
                                {profile?.subscription_tier === 'pro' ? 'PRO QUANT' : 'FREE ARCH'}
                            </span>
                        </div>
                        <div className="flex flex-wrap justify-center md:justify-start gap-3 md:gap-4 text-[10px] md:text-xs font-mono text-slate-500">
                            <div className="flex items-center gap-1.5"><Clock className="w-3 h-3 md:w-4 md:h-4" /> Since {new Date(user.created_at).toLocaleDateString()}</div>
                            <div className="flex items-center gap-1.5"><CreditCard className="w-3 h-3 md:w-4 md:h-4" /> ID: {user.id.slice(0, 8)}...</div>
                        </div>
                    </div>

                    <div className="flex flex-col gap-3 w-full md:w-auto">
                        <button
                            onClick={() => navigate('/pricing')}
                            className="btn-flame px-6 !py-3"
                        >
                            <Crown className="w-4 h-4" /> View Pricing
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
                    <div className="bg-black-ash border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-brand/30 transition-colors group">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 bg-brand/10 rounded-2xl text-brand group-hover:scale-110 transition-transform"><Activity className="w-6 h-6" /></div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Neural Load</span>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-end justify-between">
                                <div className="text-4xl font-black text-white">{profile?.upload_count || 0}<span className="text-xl text-slate-600 font-medium tracking-tight"> / {uploadLimit}</span></div>
                                <div className="text-[10px] font-bold text-slate-500 mb-1">Daily Capacity</div>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-brand shadow-[0_0_15px_rgba(245,158,11,0.5)] transition-all duration-1000"
                                    style={{ width: `${usagePercent}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="bg-black-ash border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-brand/30 transition-colors group">
                        <div className="flex items-center justify-between mb-6">
                            <div className="p-3 bg-brand/10 rounded-2xl text-brand group-hover:scale-110 transition-transform"><Target className="w-6 h-6" /></div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Model Trust</span>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-end justify-between">
                                <div className="text-4xl font-black text-white">{stats.accuracy}<span className="text-xl text-slate-600">%</span></div>
                                <div className="text-[10px] font-bold text-slate-500 mb-1">Avg. Confidence</div>
                            </div>
                            <div className="flex items-center justify-between pt-2">
                                <div className="text-[10px] font-black text-brand uppercase tracking-widest flex items-center gap-1.5"><TrendingUp className="w-3 h-3" /> {stats.wins} Wins</div>
                                <div className="text-[10px] font-black text-rose-400 uppercase tracking-widest flex items-center gap-1.5">{stats.losses} Losses <LogOut className="w-3 h-3 rotate-180" /></div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-black-ash border border-slate-800 rounded-[32px] p-8 shadow-2xl hover:border-amber-500/30 transition-colors group">
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
                    <div className="bg-black-ash/40 border border-slate-800 rounded-[32px] md:rounded-[40px] p-6 md:p-12 shadow-2xl relative overflow-hidden">
                        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 md:mb-10 gap-4">
                            <div className="flex items-center gap-4">
                                <div className="p-3 bg-brand/10 rounded-2xl text-brand">
                                    <User className="w-6 h-6" />
                                </div>
                                <div>
                                    <h3 className="text-xl md:text-2xl font-black text-white tracking-tight leading-none mb-1">Account Customization</h3>
                                    <p className="text-[10px] md:text-xs text-slate-500 font-bold uppercase tracking-widest">Premium Intelligence Dashboard</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsEditing(!isEditing)}
                                className={`w-full md:w-auto px-6 py-3 md:py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${isEditing ? 'bg-rose-500/10 text-rose-500' : 'bg-brand text-slate-950 hover:bg-brand-light'}`}
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
                                                className="w-full bg-black border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-brand/50 focus:ring-1 focus:ring-brand/50 outline-none transition-all"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">Phone Number</label>
                                            <input
                                                type="text"
                                                value={editData.phone}
                                                onChange={(e) => setEditData({ ...editData, phone: e.target.value })}
                                                placeholder="+1 (555) 000-0000"
                                                className="w-full bg-black border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-brand/50 focus:ring-1 focus:ring-brand/50 outline-none transition-all"
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
                                                className="w-full bg-black border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-brand/50 focus:ring-1 focus:ring-brand/50 outline-none transition-all"
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
                                                    className="w-full bg-black border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-brand/50 focus:ring-1 focus:ring-brand/50 outline-none transition-all"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-black text-white uppercase tracking-widest ml-1">ZIP Code</label>
                                                <input
                                                    type="text"
                                                    value={editData.billing_info.zip}
                                                    onChange={(e) => setEditData({ ...editData, billing_info: { ...editData.billing_info, zip: e.target.value } })}
                                                    placeholder="10001"
                                                    className="w-full bg-black border border-slate-800 rounded-2xl p-4 text-sm font-bold text-white focus:border-brand/50 focus:ring-1 focus:ring-brand/50 outline-none transition-all"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="md:col-span-2 pt-6">
                                    <button
                                        onClick={handleSaveProfile}
                                        className="btn-flame w-full !py-4"
                                    >
                                        Deploy Configuration Changes
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Public Identity</p>
                                    <div className="bg-black/30 p-4 rounded-2xl border border-slate-800/50">
                                        <div className="text-white font-black">{profile.full_name || 'Not Configured'}</div>
                                        <div className="text-[10px] text-slate-500 font-bold">{editData.phone || 'No phone recorded'}</div>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Billing Node</p>
                                    <div className="bg-black/30 p-4 rounded-2xl border border-slate-800/50">
                                        <div className="text-white font-black">{editData.billing_info.address || 'Address Hidden'}</div>
                                        <div className="text-[10px] text-slate-500 font-bold">{editData.billing_info.city} {editData.billing_info.zip}</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Recent Benchmarks */}
                <div className="bg-black-ash/40 border border-slate-800 rounded-[32px] overflow-hidden">
                    <div className="px-8 py-6 border-b border-slate-800 flex items-center justify-between bg-black-ash/60">
                        <h3 className="text-lg font-black text-white uppercase tracking-widest flex items-center gap-3">
                            <Activity className="w-5 h-5 text-brand" /> Recent Neural Logs
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
                                    <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 md:p-5 bg-black-ash border border-slate-800 rounded-2xl group hover:border-slate-700 transition-all cursor-pointer gap-4" onClick={() => navigate('/analysis')}>
                                        <div className="flex items-center gap-4 md:gap-6 w-full sm:w-auto">
                                            {item.imageUrl ? (
                                                <div className="w-14 h-12 md:w-16 md:h-12 rounded-lg bg-slate-800 border border-slate-700 overflow-hidden flex-shrink-0">
                                                    <img src={item.imageUrl} alt={item.ticker} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" referrerPolicy="no-referrer" />
                                                </div>
                                            ) : (
                                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-xs shrink-0 ${item.direction.includes('Bullish') ? 'bg-brand/10 text-brand' : 'bg-rose-500/10 text-rose-400'}`}>
                                                    {item.ticker.slice(0, 3)}
                                                </div>
                                            )}
                                            <div className="min-w-0">
                                                <div className="text-white font-black uppercase tracking-tight mb-0.5 text-sm truncate flex items-center gap-2">
                                                    {item.ticker}
                                                    {item.pattern?.name && <span className="text-slate-500 text-[9px] font-medium px-1.5 py-0.5 bg-slate-800 rounded hidden sm:inline-block">{item.pattern.name}</span>}
                                                </div>
                                                <div className="text-[10px] text-slate-500 font-mono truncate">{new Date(item.date).toLocaleDateString()} • {item.confidence}% Match</div>
                                                {item.pattern?.name && <div className="sm:hidden text-[9px] text-slate-600 font-medium mt-1 uppercase">{item.pattern.name}</div>}
                                            </div>
                                        </div>
                                        <div className="flex items-center justify-between sm:justify-end gap-3 w-full sm:w-auto border-t border-slate-800/50 pt-3 sm:pt-0 sm:border-0">
                                            <span className={`text-[9px] font-black px-3 py-1 rounded-md uppercase tracking-tighter ${item.feedback === 'win' ? 'bg-brand text-slate-950 shadow-lg shadow-brand-dark/20' : (item.feedback === 'loss' ? 'bg-rose-500 text-white shadow-lg shadow-rose-500/20' : 'bg-slate-800 text-slate-500')}`}>
                                                {item.feedback || 'Pending'}
                                            </span>
                                            <ArrowUpRight className="w-4 h-4 md:w-5 md:h-5 text-slate-700 group-hover:text-white transition-colors" />
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
