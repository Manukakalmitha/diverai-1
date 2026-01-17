import { Activity, Book, Crown, LogOut, User, Menu, X, ChevronRight, AlertTriangle } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { useAppContext } from '../context/AppContext';
import MobileNav from './MobileNav';

import TickerTape from './TickerTape';

const Layout = ({ children, showNav = true, showTicker = true, navStyle }) => {
    const { user, profile } = useAppContext();
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-black text-slate-200 font-sans selection:bg-blue-500/30 overflow-x-hidden">
            {/* Maintenance Banner */}
            <div className="bg-amber-500 text-slate-950 font-black py-2.5 px-6 flex items-center justify-center gap-3 text-[10px] uppercase tracking-[0.2em] relative z-[100] shadow-2xl">
                <AlertTriangle className="w-4 h-4" />
                <span>System Undergoing Maintenance - Neural Core Synchronization in Progress</span>
                <AlertTriangle className="w-4 h-4" />
            </div>

            {/* Professional Header */}
            {showNav && (
                <header className={`flex flex-col z-50 sticky top-0 shadow-sm border-b border-ash bg-black/80 backdrop-blur-md ${navStyle === 'minimal' ? 'h-14' : ''}`}>
                    <div className={`flex items-center justify-between px-6 ${navStyle === 'minimal' ? 'h-14' : 'h-16'} bg-transparent`}>
                        <Link to="/" className="flex items-center gap-2 cursor-pointer group">
                            <div className="bg-brand rounded-lg p-1.5 shadow-sm group-hover:bg-brand-light transition-colors">
                                <Activity className="text-slate-950 w-5 h-5" />
                            </div>
                            <span className="text-xl font-bold tracking-tight text-white">Diver<span className="text-brand">AI</span></span>
                        </Link>

                        {navStyle !== 'minimal' && (
                            <nav className="hidden md:flex items-center gap-8">
                                <Link to="/what-is-diver-ai" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">Product</Link>
                                <Link to="/articles" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">Markets</Link>
                                <Link to="/docs" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">Documentation</Link>
                                <Link to="/pricing" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">Pricing</Link>
                                <Link to="/referral" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">Referral</Link>
                            </nav>
                        )}

                        <div className="flex items-center gap-4">
                            {user ? (
                                <div className="flex items-center gap-4">
                                    <Link to="/profile" className="flex items-center gap-3 cursor-pointer hover:bg-slate-800/50 p-1.5 rounded-lg transition-colors">
                                        <div className="text-right hidden sm:block">
                                            <div className="text-xs font-semibold text-white leading-none mb-1">{user.email}</div>
                                            <div className="text-[10px] text-brand-light font-bold uppercase leading-none">
                                                {profile?.subscription_tier === 'pro' ? 'PRO PLAN' : 'BASIC'}
                                            </div>
                                        </div>
                                        <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
                                            {profile?.avatar_url ? (
                                                <img src={profile.avatar_url} alt="Profile" className="w-full h-full object-cover rounded-full" referrerPolicy="no-referrer" />
                                            ) : (
                                                <span className="text-xs font-bold text-slate-400">{user.email[0].toUpperCase()}</span>
                                            )}
                                        </div>
                                    </Link>
                                    <button
                                        onClick={() => { supabase.auth.signOut(); navigate('/'); }}
                                        className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
                                    >
                                        <LogOut className="w-4 h-4" />
                                    </button>
                                </div>
                            ) : (
                                <div className="flex items-center gap-3">
                                    <Link
                                        to="/login"
                                        className="text-sm font-semibold text-slate-300 hover:text-white transition-colors"
                                    >
                                        Sign In
                                    </Link>
                                    <Link
                                        to="/signup"
                                        className="px-4 py-2 bg-brand hover:bg-brand-light text-slate-950 font-bold rounded-lg text-sm transition-colors shadow-sm"
                                    >
                                        Get Started
                                    </Link>
                                </div>
                            )}
                        </div>
                    </div>
                    {showTicker && <TickerTape />}
                </header>
            )}

            <main className="relative z-10 w-full pb-20 md:pb-0">
                {children}
            </main>
            <MobileNav />
        </div>
    );
};

export default Layout;
