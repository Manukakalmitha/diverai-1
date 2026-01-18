import React, { useState } from 'react';
import { supabase } from '../lib/supabase';
import { Mail, Lock, Loader2, X, AlertCircle } from 'lucide-react';
import { getThumbmark } from '@thumbmarkjs/thumbmarkjs';

const AuthModal = ({ onClose, onSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [isEmailSent, setIsEmailSent] = useState(false);
    const [loading, setLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const [resendCooldown, setResendCooldown] = useState(0);
    const [fingerprint, setFingerprint] = useState(null);

    React.useEffect(() => {
        getThumbmark().then(fp => setFingerprint(fp));
    }, []);

    const handleAuth = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            if (isLogin) {
                const { error } = await supabase.auth.signInWithPassword({ email, password });
                if (error) throw error;
                if (window.chrome && chrome.runtime && chrome.runtime.sendMessage) {
                    // Try to sync with extension if we have an ID from URL
                    const params = new URLSearchParams(window.location.search);
                    const extId = params.get('extId');
                    if (extId && chrome?.runtime?.sendMessage) {
                        try {
                            chrome.runtime.sendMessage(extId, {
                                type: 'AUTH_SYNC',
                                session: {
                                    access_token: data.session.access_token,
                                    refresh_token: data.session.refresh_token
                                }
                            }, (response) => {
                                if (chrome.runtime.lastError) {
                                    console.info("Extension sync skipped: Connection could not be established.");
                                }
                            });
                        } catch (err) {
                            console.warn("Auth sync error:", err);
                        }
                    }
                }
                onSuccess();
                onClose();
            } else {
                const { error } = await supabase.auth.signUp({
                    email,
                    password,
                    options: {
                        data: {
                            device_fingerprint: fingerprint
                        },
                        emailRedirectTo: `${window.location.origin}/analysis`
                    }
                });
                if (error) throw error;
                setIsEmailSent(true);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        setLoading(true);
        setError(null);
        try {
            const { error } = await supabase.auth.signInWithOAuth({
                provider: 'google',
                options: {
                    redirectTo: `${window.location.origin}/analysis`
                }
            });
            if (error) throw error;
        } catch (err) {
            setError(err.message);
            setLoading(false);
        }
    };

    const handleResend = async () => {
        if (resendCooldown > 0) return;
        setLoading(true);
        try {
            const { error } = await supabase.auth.resend({
                type: 'signup',
                email: email,
                options: {
                    redirectTo: `${window.location.origin}/`
                }
            });
            if (error) throw error;
            setResendCooldown(60);
            const timer = setInterval(() => {
                setResendCooldown((prev) => {
                    if (prev <= 1) {
                        clearInterval(timer);
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const [checkingStatus, setCheckingStatus] = useState(false);

    React.useEffect(() => {
        let interval;
        if (isEmailSent) {
            // Poll for session every 5 seconds
            interval = setInterval(async () => {
                const { data: { session } } = await supabase.auth.getSession();
                if (session) {
                    clearInterval(interval);
                    clearInterval(interval);

                    if (window.chrome && chrome.runtime && chrome.runtime.sendMessage) {
                        const params = new URLSearchParams(window.location.search);
                        const extId = params.get('extId');
                        if (extId && chrome?.runtime?.sendMessage) {
                            try {
                                chrome.runtime.sendMessage(extId, {
                                    type: 'AUTH_SYNC',
                                    session: {
                                        access_token: session.access_token,
                                        refresh_token: session.refresh_token
                                    }
                                }, () => {
                                    if (chrome.runtime.lastError) {
                                        // Silent fail - extension might not be ready
                                    }
                                });
                            } catch (e) { }
                        }
                    }

                    onSuccess();
                    onClose();
                }
            }, 5000);
        }
        return () => interval && clearInterval(interval);
    }, [isEmailSent, onSuccess, onClose]);

    const checkVerificationStatus = async () => {
        setCheckingStatus(true);
        try {
            const { data: { session }, error } = await supabase.auth.getSession();
            if (error) throw error;
            if (session) {
                if (window.chrome && chrome.runtime && chrome.runtime.sendMessage) {
                    const params = new URLSearchParams(window.location.search);
                    const extId = params.get('extId');
                    if (extId && chrome?.runtime?.sendMessage) {
                        try {
                            chrome.runtime.sendMessage(extId, {
                                type: 'AUTH_SYNC',
                                session: {
                                    access_token: session.access_token,
                                    refresh_token: session.refresh_token
                                }
                            }, () => {
                                if (chrome.runtime.lastError) {
                                    // Silent fail
                                }
                            });
                        } catch (e) { }
                    }
                }
                onSuccess();
                onClose();
            } else {
                setError("Verification still pending. Please click the link in your email.");
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setCheckingStatus(false);
        }
    };

    if (isEmailSent) {
        return (
            <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
                <div className="bg-black border border-slate-800 rounded-[32px] w-full max-w-md p-8 relative shadow-2xl animate-in zoom-in-95 duration-500 overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-brand via-brand-light to-brand-dark"></div>

                    <button onClick={onClose} className="absolute top-6 right-6 text-slate-500 hover:text-white transition-colors">
                        <X className="w-5 h-5" />
                    </button>

                    <div className="text-center space-y-6 pt-4">
                        <div className="relative mx-auto w-20 h-20">
                            <div className="absolute inset-0 bg-brand/20 rounded-full animate-ping"></div>
                            <div className="relative bg-brand rounded-full w-20 h-20 flex items-center justify-center shadow-xl shadow-brand/20">
                                <Mail className="w-10 h-10 text-slate-950 animate-bounce" />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <h2 className="text-3xl font-black text-white tracking-tighter uppercase">Check Your Inbox</h2>
                            <p className="text-slate-400 text-sm font-medium leading-relaxed">
                                We've sent a magic link to <br />
                                <span className="text-brand font-bold">{email}</span>
                            </p>
                        </div>

                        <div className="p-4 bg-black/50 rounded-2xl border border-slate-800/50 text-left">
                            <p className="text-[11px] text-slate-500 font-bold uppercase tracking-widest leading-relaxed">
                                <span className="text-brand mr-2">●</span> Click the link in the email to verify your account.
                                <br />
                                <span className="text-brand mr-2">●</span> Your app will automatically update once verified.
                            </p>
                        </div>

                        {error && (
                            <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-[10px] rounded-xl font-bold flex items-center gap-2">
                                <AlertCircle className="w-4 h-4" /> {error}
                            </div>
                        )}

                        <div className="flex flex-col gap-3">
                            <button
                                onClick={checkVerificationStatus}
                                disabled={checkingStatus}
                                className="w-full py-3.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-2xl transition-all flex items-center justify-center gap-2 text-xs uppercase tracking-widest disabled:opacity-50"
                            >
                                {checkingStatus ? <Loader2 className="w-4 h-4 animate-spin" /> : 'I Have Verified'}
                            </button>
                            <button
                                onClick={handleResend}
                                disabled={loading || resendCooldown > 0}
                                className="w-full py-3.5 bg-slate-800 hover:bg-slate-700 text-white font-black rounded-2xl transition-all flex items-center justify-center gap-2 text-xs uppercase tracking-widest disabled:opacity-50"
                            >
                                {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : 'Resend Email'}
                            </button>
                            <button
                                onClick={() => setIsEmailSent(false)}
                                className="text-slate-500 hover:text-white text-xs font-black uppercase tracking-[0.2em] transition-colors"
                            >
                                Back to Login
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="bg-black border border-slate-800 rounded-[32px] w-full max-w-md p-8 relative shadow-2xl animate-in zoom-in-95 duration-300 overflow-hidden">
                <button onClick={onClose} className="absolute top-6 right-6 text-slate-500 hover:text-white transition-colors"><X className="w-5 h-5" /></button>

                <div className="text-center mb-8 pt-4">
                    <div className="bg-brand/10 w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4">
                        <Lock className="w-6 h-6 text-brand" />
                    </div>
                    <h2 className="text-3xl font-black text-white mb-2 tracking-tighter uppercase">{isLogin ? 'Quant Access' : 'Secure Entry'}</h2>
                    <p className="text-slate-500 text-xs font-black uppercase tracking-widest leading-none">Terminal Authentication Sequence</p>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-2xl flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
                        <AlertCircle className="w-5 h-5 shrink-0" /> <span className="font-bold">{error}</span>
                    </div>
                )}

                <div className="space-y-4">
                    <button
                        onClick={handleGoogleLogin}
                        disabled={loading}
                        className="w-full bg-white hover:bg-slate-200 text-slate-900 font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-3 disabled:opacity-50 text-xs uppercase tracking-[0.2em]"
                    >
                        <img src="https://www.svgrepo.com/show/355037/google.svg" alt="Google" className="w-5 h-5" />
                        <span>Continue with Google</span>
                    </button>

                    <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t border-slate-800" />
                        </div>
                        <div className="relative flex justify-center text-[10px] uppercase">
                            <span className="bg-black px-2 text-slate-500 font-black tracking-widest">Or use encryption key</span>
                        </div>
                    </div>
                </div>

                <form onSubmit={handleAuth} className="space-y-5">
                    <div className="space-y-2">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Email Terminal</label>
                        <div className="relative">
                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 w-4 h-4" />
                            <input
                                type="email"
                                required
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full bg-black border border-slate-800 rounded-2xl py-3.5 pl-11 pr-4 text-white focus:outline-none focus:border-brand/50 transition-all font-mono text-sm"
                                placeholder="TRADER@QUANT.AI"
                                autocomplete="username"
                            />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest pl-1">Encryption Key</label>
                        <div className="relative">
                            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 w-4 h-4" />
                            <input
                                type="password"
                                required
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full bg-black border border-slate-800 rounded-2xl py-3.5 pl-11 pr-4 text-white focus:outline-none focus:border-brand/50 transition-all font-mono text-sm"
                                placeholder="••••••••"
                                autocomplete={isLogin ? "current-password" : "new-password"}
                            />
                        </div>
                    </div>

                    <button
                        disabled={loading}
                        className="btn-flame w-full !py-4"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : (isLogin ? 'Initialize Session' : 'Create Profile')}
                    </button>
                </form>

                <div className="mt-8 text-center text-[10px] font-black uppercase tracking-widest text-slate-500">
                    {isLogin ? "Neural uplink missing? " : "Existing profile found? "}
                    <button onClick={() => setIsLogin(!isLogin)} className="text-brand hover:text-brand-light transition-colors ml-1 underline decoration-2 underline-offset-4">
                        {isLogin ? 'Register' : 'Login'}
                    </button>
                </div>
            </div>
        </div>

    );
};

export default AuthModal;
