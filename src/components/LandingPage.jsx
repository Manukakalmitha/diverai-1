import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ArrowRight,
    Activity,
    ShieldCheck,
    Zap,
    Cpu,
    Globe,
    BarChart3,
    Lock,
    ChevronDown,
    Plus,
    Minus,
    CheckCircle2,
    Quote,
    MessageSquare,
    HelpCircle,
    Layers,
    Search,
    ChevronRight,
    Sparkles,
    Star,
    TrendingUp,
    Share,
    MoreVertical,
    Download,
    X,
    Terminal,
    Maximize2,
    TrendingDown,
    Wifi,
    Chrome,
    Smartphone,
    Monitor,
    FileText,
    Megaphone
} from 'lucide-react';
import ReviewSection from './ReviewSection';

// Announcement Banner Component
const AnnouncementBanner = () => {
    const BANNER_VERSION = 'jan-2026-v4'; // Change this to show banner again for new announcements

    // Check localStorage synchronously to avoid flash
    const getInitialDismissedState = () => {
        if (typeof window === 'undefined') return true;
        return localStorage.getItem(`banner-dismissed-${BANNER_VERSION}`) === 'true';
    };

    const [isDismissed, setIsDismissed] = useState(getInitialDismissedState);

    const handleDismiss = () => {
        setIsDismissed(true);
        localStorage.setItem(`banner-dismissed-${BANNER_VERSION}`, 'true');
    };

    if (isDismissed) return null;

    return (
        <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="relative bg-gradient-to-r from-emerald-600/90 via-emerald-500/90 to-blue-600/90 text-white overflow-hidden"
        >
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay pointer-events-none"></div>
            <div className="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-center gap-4 relative">
                <div className="flex items-center gap-3">
                    <div className="hidden sm:flex items-center justify-center w-6 h-6 bg-white/20 rounded-full backdrop-blur-sm">
                        <Megaphone className="w-3.5 h-3.5" />
                    </div>
                    <p className="text-xs sm:text-sm font-medium text-center">
                        <span className="font-bold">What's New:</span> Diver AI v5.0 is here with MTF Alignment and Institutional VWAP!
                    </p>
                </div>
                <a
                    href="/updates"
                    className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-full text-xs font-bold transition-all hover:scale-105"
                >
                    See Updates <ArrowRight className="w-3 h-3" />
                </a>
                <button
                    onClick={handleDismiss}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-white/20 rounded-full transition-colors"
                    aria-label="Dismiss announcement"
                >
                    <X className="w-4 h-4" />
                </button>
            </div>
        </motion.div>
    );
};


// Sample testimonials data
const testimonials = [
    {
        text: "Diver AI transformed my trading strategy with precise forecasts.",
        name: "Alex Johnson",
        role: "Professional Trader",
    },
    {
        text: "The insights are crystal clear and actionable.",
        name: "Maria Chen",
        role: "Quant Analyst",
    },
    {
        text: "A game-changer for market analysis.",
        name: "Liam Patel",
        role: "Portfolio Manager",
    },
];

// Sample FAQs data
const faqs = [
    {
        q: "How does Diver AI work?",
        a: "It uses optical pattern recognition and probabilistic forecasting to analyze market data.",
    },
    {
        q: "Is there a free trial?",
        a: "Yes, you get a limited number of free scans each day.",
    },
    {
        q: "What data sources are used?",
        a: "We aggregate data from multiple exchanges and financial APIs.",
    },
];

const InstallationGuideModal = ({ isOpen, onClose, platform }) => {
    if (!isOpen) return null;

    const instructions = {
        ios: {
            title: "Add to iOS Home Screen",
            steps: [
                { icon: Share, text: "Tap 'Share' in Safari toolbar" },
                { icon: Plus, text: "Select 'Add to Home Screen'" },
                { icon: CheckCircle2, text: "Tap 'Add' to confirm" }
            ]
        },
        android: {
            title: "Install on Android",
            steps: [
                { icon: MoreVertical, text: "Tap menu (three dots)" },
                { icon: Download, text: "Select 'Install App'" },
                { icon: CheckCircle2, text: "Follow prompts" }
            ]
        },
        desktop: {
            title: "Install Desktop App",
            steps: [
                { icon: Download, text: "Click install icon in URL bar" },
                { icon: MoreVertical, text: "Or select 'Apps' > 'Install'" },
                { icon: CheckCircle2, text: "Launch from desktop" }
            ]
        }
    };

    const guide = instructions[platform] || instructions.desktop;

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-6 bg-slate-950/80 backdrop-blur-sm">
            <div className="relative w-full max-w-sm bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
                <div className="p-6 border-b border-slate-700 flex justify-between items-center">
                    <h3 className="text-lg font-bold text-white">{guide.title}</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-white"><X className="w-5 h-5" /></button>
                </div>
                <div className="p-6 space-y-4">
                    {guide.steps.map((step, i) => (
                        <div key={i} className="flex items-center gap-4">
                            <div className="w-8 h-8 bg-blue-600/20 rounded-lg flex items-center justify-center text-blue-500">
                                <step.icon className="w-4 h-4" />
                            </div>
                            <p className="text-sm font-medium text-slate-300">{step.text}</p>
                        </div>
                    ))}
                    <button
                        onClick={onClose}
                        className="w-full py-3 mt-4 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-500 transition-colors"
                    >
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};

const LandingPage = ({ onStart, onOpenInfo, onOpenDocs, onOpenPricing }) => {
    const [scrolled, setScrolled] = useState(false);
    const [deferredPrompt, setDeferredPrompt] = useState(null);
    const [showGuide, setShowGuide] = useState(false);
    const [platform, setPlatform] = useState('desktop');
    const [activeFaq, setActiveFaq] = useState(null);

    useEffect(() => {
        const userAgent = window.navigator.userAgent.toLowerCase();
        if (/iphone|ipad|ipod/.test(userAgent)) setPlatform('ios');
        else if (/android/.test(userAgent)) setPlatform('android');

        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            setDeferredPrompt(e);
        });

        const handleScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const handleInstallClick = async () => {
        if (!deferredPrompt) {
            setShowGuide(true);
            return;
        }
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        setDeferredPrompt(null);
    };

    return (
        <div className="min-h-screen bg-[#050B14] text-white font-sans selection:bg-emerald-500/30">
            <InstallationGuideModal isOpen={showGuide} onClose={() => setShowGuide(false)} platform={platform} />

            <main>
                {/* Announcement Banner */}
                <AnnouncementBanner />

                {/* Institutional Hero Section */}
                <section className="relative pt-32 pb-40 overflow-hidden bg-[#020617]">
                    {/* Architectural Grid Background */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>
                    <div className="absolute inset-0 bg-gradient-to-b from-[#020617] via-transparent to-[#020617] pointer-events-none"></div>

                    <div className="max-w-7xl mx-auto px-6 lg:px-8 relative z-10">
                        <div className="flex flex-col lg:flex-row items-center gap-16 lg:gap-24">

                            {/* Copy Column */}
                            <div className="flex-1 text-center lg:text-left space-y-8">
                                <motion.div
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="inline-flex items-center gap-3 px-4 py-1.5 bg-slate-900 border border-slate-800 rounded-full"
                                >
                                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_10px_#10b981]"></div>
                                    <span className="text-[11px] font-bold text-slate-300 uppercase tracking-widest">System Operational v5.0</span>
                                </motion.div>

                                <motion.h1
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.1 }}
                                    className="text-5xl lg:text-7xl font-bold tracking-tight text-white leading-[1.1]"
                                >
                                    Institutional <br />
                                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-slate-200 to-slate-500">Market Intelligence.</span>
                                </motion.h1>

                                <motion.p
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.2 }}
                                    className="text-lg text-slate-400 max-w-xl mx-auto lg:mx-0 leading-relaxed font-light"
                                >
                                    Diver AI provides hedge-fund grade optical analysis and probabilistic forecasting for the modern independent trader. Deploy systematic alpha strategies with precision.
                                </motion.p>

                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 }}
                                    className="flex flex-wrap items-center justify-center lg:justify-start gap-4"
                                >
                                    <button onClick={onStart} className="px-8 py-4 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-all shadow-lg shadow-emerald-900/20 flex items-center gap-2 text-sm uppercase tracking-wider">
                                        Launch Terminal <ArrowRight className="w-4 h-4" />
                                    </button>
                                    <button onClick={onOpenDocs} className="px-8 py-4 bg-transparent border border-slate-700 text-slate-300 font-semibold rounded-lg hover:bg-slate-900/50 transition-colors text-sm uppercase tracking-wider">
                                        View Documentation
                                    </button>
                                    <a
                                        href="/diverai-quant-whitepaper.pdf"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-8 py-4 bg-slate-900 border border-slate-800 text-emerald-500 font-bold rounded-lg hover:border-emerald-500/50 transition-all text-sm uppercase tracking-widest flex items-center gap-2"
                                    >
                                        <FileText className="w-4 h-4" /> White Paper
                                    </a>
                                </motion.div>

                                <div className="pt-8 flex items-center justify-center lg:justify-start gap-8 opacity-60 grayscale hover:grayscale-0 transition-all duration-500">
                                    <div className="text-xs font-mono text-slate-500 flex items-center gap-2"><Lock className="w-3 h-3" /> SOC2 Compliant Arch</div>
                                    <div className="text-xs font-mono text-slate-500 flex items-center gap-2"><ShieldCheck className="w-3 h-3" /> 256-bit Encryption</div>
                                </div>
                            </div>

                            {/* Visual Column (Mock Terminal) */}
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: 0.4 }}
                                className="flex-1 w-full max-w-2xl"
                            >
                                <div className="rounded-xl bg-[#0B1121] border border-slate-800 shadow-2xl overflow-hidden relative group">
                                    <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-1000"></div>

                                    {/* Glass Header */}
                                    <div className="h-9 bg-slate-900/80 backdrop-blur border-b border-slate-800 flex items-center justify-between px-4">
                                        <div className="flex gap-1.5 opacity-50">
                                            <div className="w-2.5 h-2.5 rounded-full bg-slate-600"></div>
                                            <div className="w-2.5 h-2.5 rounded-full bg-slate-600"></div>
                                            <div className="w-2.5 h-2.5 rounded-full bg-slate-600"></div>
                                        </div>
                                        <div className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">Diver AI Quantitative Core</div>
                                    </div>

                                    {/* Minimalist Data Grid */}
                                    <div className="p-1">
                                        <div className="grid grid-cols-3 gap-px bg-slate-900/50 border border-slate-800 rounded-lg overflow-hidden">
                                            {/* Live Ticker Block */}
                                            <div className="col-span-3 p-6 bg-[#0B1121] border-b border-slate-800 flex justify-between items-end">
                                                <div>
                                                    <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider mb-1">Asset Class</div>
                                                    <div className="text-3xl font-bold text-white tracking-tight">BTC/USD</div>
                                                </div>
                                                <div className="text-right">
                                                    <div className="text-2xl font-mono text-emerald-400">$64,241.50</div>
                                                    <div className="text-xs font-bold text-emerald-600">+1.24% (24h)</div>
                                                </div>
                                            </div>

                                            {/* Metric Cells */}
                                            <div className="p-4 bg-[#0B1121] hover:bg-slate-900/50 transition-colors">
                                                <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Neural Confidence</div>
                                                <div className="text-lg font-bold text-emerald-400">94.2%</div>
                                            </div>
                                            <div className="p-4 bg-[#0B1121] hover:bg-slate-900/50 transition-colors">
                                                <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Volatility (ATR)</div>
                                                <div className="text-lg font-bold text-white">1.45%</div>
                                            </div>
                                            <div className="p-4 bg-[#0B1121] hover:bg-slate-900/50 transition-colors">
                                                <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Sharpe Ratio</div>
                                                <div className="text-lg font-bold text-blue-400">2.81</div>
                                            </div>
                                        </div>

                                        {/* Abstract Chart */}
                                        <div className="h-48 mt-1 bg-[#0B1121] border border-slate-800 rounded-lg relative overflow-hidden">
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <div className="w-full h-px bg-slate-800/30"></div>
                                            </div>
                                            <svg className="w-full h-full" preserveAspectRatio="none">
                                                <path d="M0 120 C 100 110, 200 60, 300 80 S 500 40, 600 50" stroke="#10b981" strokeWidth="1.5" fill="none" className="drop-shadow-[0_0_5px_rgba(16,185,129,0.5)]" />
                                                <path d="M0 120 L 600 50 L 600 200 L 0 200 Z" fill="url(#grad)" opacity="0.1" />
                                                <defs>
                                                    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="#10b981" />
                                                        <stop offset="100%" stopColor="transparent" />
                                                    </linearGradient>
                                                </defs>
                                            </svg>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </div>
                </section>

                {/* Universal Compatibility Section */}
                <section className="py-20 border-y border-slate-900 bg-[#020617]">
                    <div className="max-w-7xl mx-auto px-6 text-center space-y-12">
                        <div className="space-y-4">
                            <h2 className="text-sm font-bold text-emerald-500 uppercase tracking-[0.2em]">Universal Integration</h2>
                            <p className="text-slate-400 font-medium text-lg max-w-2xl mx-auto">
                                The Diver AI Neural Extension is available on the Chrome Web Store and is architected to overlay seamlessly on any institutional charting platform.
                            </p>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12 opacity-60 grayscale hover:grayscale-0 transition-all duration-500">
                            {/* Yahoo Finance */}
                            <div className="flex items-center justify-center gap-3 group">
                                <span className="text-2xl font-black text-slate-300 group-hover:text-purple-400 transition-colors tracking-tighter">yahoo<span className="text-purple-500">!</span>finance</span>
                            </div>
                            {/* TradingView */}
                            <div className="flex items-center justify-center gap-3 group">
                                <svg className="w-8 h-8 text-slate-300 group-hover:text-blue-500 transition-colors" viewBox="0 0 24 24" fill="currentColor"><path d="M21 12a9 9 0 1 1-9-9 9 9 0 0 1 9 9z" opacity="0.2" /><path d="M16.5 13.5a1.5 1.5 0 1 0-.001-3.001A1.5 1.5 0 0 0 16.5 13.5zm-5 4a1.5 1.5 0 1 0-.001-3.001A1.5 1.5 0 0 0 11.5 17.5zm-5-8a1.5 1.5 0 1 0-.001-3.001A1.5 1.5 0 0 0 6.5 9.5z" /></svg>
                                <span className="text-lg font-bold text-slate-300 group-hover:text-white transition-colors">TradingView</span>
                            </div>
                            {/* CoinGecko */}
                            <div className="flex items-center justify-center gap-3 group">
                                <span className="text-xl font-black text-slate-300 group-hover:text-emerald-400 transition-colors tracking-tight">CoinGecko</span>
                            </div>
                            {/* Robinhood */}
                            <div className="flex items-center justify-center gap-3 group">
                                <span className="text-xl font-black text-slate-300 group-hover:text-emerald-500 transition-colors tracking-tight">Robinhood</span>
                            </div>

                        </div>

                        <div className="text-[10px] font-mono text-slate-600 uppercase tracking-widest pt-8">
                            * Institutional Core v5.0 Precision Certified
                        </div>
                    </div>
                </section>

                {/* Features Grid */}
                <section className="py-32 bg-slate-950 border-t border-slate-900">
                    <div className="max-w-7xl mx-auto px-6 lg:px-8">
                        <div className="grid md:grid-cols-3 gap-12">
                            <div className="space-y-4">
                                <div className="w-12 h-12 bg-blue-500/10 rounded-xl border border-blue-500/20 flex items-center justify-center">
                                    <Activity className="w-6 h-6 text-blue-400" />
                                </div>
                                <h3 className="text-xl font-bold text-white">Pattern Recognition</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    Our computer vision algorithms scan chart structures in real-time, identifying complex setups invisible to standard indicators.
                                </p>
                            </div>
                            <div className="space-y-4">
                                <div className="w-12 h-12 bg-emerald-500/10 rounded-xl border border-emerald-500/20 flex items-center justify-center">
                                    <Cpu className="w-6 h-6 text-emerald-400" />
                                </div>
                                <h3 className="text-xl font-bold text-white">Probabilistic Forecasting</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    We don't predict prices; we calculate probabilities. Every signal comes with a confidence score derived from historical backtesting.
                                </p>
                            </div>
                            <div className="space-y-4">
                                <div className="w-12 h-12 bg-purple-500/10 rounded-xl border border-purple-500/20 flex items-center justify-center">
                                    <Globe className="w-6 h-6 text-purple-400" />
                                </div>
                                <h3 className="text-xl font-bold text-white">Macro Synthesis</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    Global macro data is ingested alongside technicals to filter out false positives during high-volatility events.
                                </p>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Chrome Extension Promo */}
                <section className="py-32 bg-[#020617] relative overflow-hidden">
                    {/* Background Glow */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/10 blur-[120px] rounded-full pointer-events-none"></div>

                    <div className="max-w-7xl mx-auto px-6 lg:px-8 relative z-10 text-center space-y-12">
                        <div className="space-y-6 max-w-3xl mx-auto">
                            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full">
                                <Chrome className="w-4 h-4 text-blue-400" />
                                <span className="text-[10px] font-black uppercase tracking-widest text-blue-300">Browser Integration</span>
                            </div>
                            <h2 className="text-4xl md:text-6xl font-black text-white tracking-tighter">
                                Seamless <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">Workflow Overlay.</span>
                            </h2>
                            <p className="text-slate-400 text-lg leading-relaxed">
                                Don't switch tabs. The Diver AI Chrome Extension overlays institutional-grade analysis directly onto your existing charts. Compatible with TradingView, Yahoo Finance, CoinGecko, and more.
                            </p>
                        </div>

                        <div className="flex justify-center">
                            <a
                                href="https://chromewebstore.google.com/detail/gjmicgddmbghplbdolnmiecjkdfipand?utm_source=item-share-cb"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="group relative inline-flex items-center justify-center gap-3 px-10 py-5 bg-white text-slate-950 font-black rounded-xl hover:bg-slate-200 transition-all shadow-xl shadow-white/10 overflow-hidden"
                            >
                                <Chrome className="w-6 h-6 text-blue-600" />
                                <span className="text-sm uppercase tracking-widest relative z-10">Add to Chrome</span>
                                <ArrowRight className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform" />
                            </a>
                        </div>

                        {/* Preview Image / Mockup could go here, for now using a clean layout */}
                    </div>
                </section>

                {/* Testimonials */}
                <section className="py-24 max-w-7xl mx-auto border-t border-slate-900 px-6">
                    <div className="text-center mb-16"><h2 className="text-3xl md:text-5xl font-black tracking-tighter">Verified Alpha.</h2></div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {testimonials.map((t, i) => (
                            <div key={i} className="p-10 bg-slate-900/30 border border-slate-800/50 rounded-2xl space-y-6">
                                <div className="flex text-emerald-500 gap-1">{[...Array(5)].map((_, j) => <Star key={j} className="w-4 h-4 fill-current" />)}</div>
                                <p className="text-slate-300 font-medium text-sm leading-relaxed font-mono">"{t.text}"</p>
                                <div className="flex items-center gap-4 pt-4 border-t border-slate-800/50">
                                    <div className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center font-black text-emerald-400 text-xs">{t.name[0]}</div>
                                    <div><h4 className="text-white font-bold text-sm uppercase tracking-tight">{t.name}</h4><p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">{t.role}</p></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* FAQ */}
                <section id="faq" className="py-24 max-w-3xl mx-auto px-6">
                    <div className="text-center mb-16 space-y-4">
                        <h2 className="text-4xl md:text-5xl font-black tracking-tighter">System Logic.</h2>
                        <p className="text-slate-500 font-medium font-mono text-sm">Protocol documentation and operation details.</p>
                    </div>
                    <div className="space-y-4">
                        {faqs.map((faq, i) => (
                            <div key={i} className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/20">
                                <button
                                    onClick={() => setActiveFaq(activeFaq === i ? null : i)}
                                    className="w-full px-8 py-6 flex items-center justify-between text-left hover:bg-slate-900 transition-colors group"
                                >
                                    <span className="font-bold uppercase tracking-tight text-xs md:text-sm group-hover:text-emerald-400 transition-colors text-slate-300">{faq.q}</span>
                                    <div className={`p-2 rounded-lg bg-slate-950 border border-slate-800 transition-transform duration-300 ${activeFaq === i ? 'rotate-180' : ''}`}>
                                        <ChevronDown className="w-4 h-4 text-slate-500" />
                                    </div>
                                </button>
                                <AnimatePresence>
                                    {activeFaq === i && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            className="px-8 pb-8"
                                        >
                                            <p className="text-slate-400 text-sm font-medium leading-relaxed pt-2 border-t border-slate-800/50">{faq.a}</p>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        ))}
                    </div>
                </section>

                <ReviewSection />

                {/* Mobile Command Center */}
                <section className="py-24 bg-[#0B1121] border-t border-b border-slate-900 overflow-hidden relative">
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:40px_40px] opacity-10 pointer-events-none"></div>
                    <div className="max-w-7xl mx-auto px-6 relative z-10">
                        <div className="flex flex-col lg:flex-row items-center justify-between gap-16">
                            <div className="flex-1 space-y-8 text-center lg:text-left">
                                <div className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-full">
                                    <Wifi className="w-4 h-4 text-emerald-500 animate-pulse" />
                                    <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Mobile Uplink Active</span>
                                </div>
                                <h2 className="text-4xl md:text-5xl font-black text-white tracking-tighter">
                                    Command Center <br />
                                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-500">In Your Pocket.</span>
                                </h2>
                                <p className="text-slate-400 text-lg max-w-xl mx-auto lg:mx-0 leading-relaxed">
                                    Monitor your positions and receive real-time neural alerts directly on your device. The Diver AI PWA installs as a native application for zero-latency performance.
                                </p>
                                <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4">
                                    <button
                                        onClick={() => { setPlatform('ios'); setShowGuide(true); }}
                                        className="px-8 py-4 bg-slate-900 border border-slate-700 hover:border-emerald-500/50 text-white font-bold rounded-xl transition-all flex items-center gap-3 min-w-[200px] justify-center group"
                                    >
                                        <Smartphone className="w-5 h-5 text-slate-400 group-hover:text-emerald-400 transition-colors" />
                                        <span>Install on iOS</span>
                                    </button>
                                    <button
                                        onClick={() => { setPlatform('android'); setShowGuide(true); }}
                                        className="px-8 py-4 bg-slate-900 border border-slate-700 hover:border-emerald-500/50 text-white font-bold rounded-xl transition-all flex items-center gap-3 min-w-[200px] justify-center group"
                                    >
                                        <Monitor className="w-5 h-5 text-slate-400 group-hover:text-emerald-400 transition-colors" />
                                        <span>Install on Android</span>
                                    </button>
                                </div>
                            </div>

                            {/* Mobile Mockup */}
                            <div className="relative w-[300px] mx-auto lg:mx-0">
                                <div className="absolute inset-0 bg-emerald-500/20 blur-[100px] rounded-full"></div>
                                <div className="relative border-slate-800 bg-slate-950 border-[8px] rounded-[3rem] h-[600px] w-[300px] shadow-2xl flex flex-col overflow-hidden">
                                    <div className="h-[32px] w-[3px] bg-slate-800 absolute -left-[10px] top-[72px] rounded-l-lg"></div>
                                    <div className="h-[46px] w-[3px] bg-slate-800 absolute -left-[10px] top-[124px] rounded-l-lg"></div>
                                    <div className="h-[46px] w-[3px] bg-slate-800 absolute -left-[10px] top-[178px] rounded-l-lg"></div>
                                    <div className="h-[64px] w-[3px] bg-slate-800 absolute -right-[10px] top-[142px] rounded-r-lg"></div>

                                    {/* Screen Content */}
                                    <div className="flex-1 bg-[#050B14] w-full overflow-hidden relative">
                                        <div className="absolute top-0 left-0 right-0 h-6 bg-slate-950 z-20 flex justify-between px-6 items-center">
                                            <div className="text-[10px] font-bold text-white">9:41</div>
                                            <div className="flex gap-1">
                                                <div className="w-3 h-3 bg-white rounded-full"></div>
                                            </div>
                                        </div>
                                        <div className="pt-12 px-6 space-y-6">
                                            <div className="space-y-2">
                                                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">Portfolio Value</div>
                                                <div className="text-3xl font-black text-white">$42,891.20</div>
                                                <div className="text-xs font-bold text-emerald-500">+8.4% (All Time)</div>
                                            </div>
                                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800 space-y-4">
                                                <div className="flex justify-between items-center">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-8 h-8 rounded-full bg-orange-500/20 flex items-center justify-center text-orange-500 font-bold text-xs">₿</div>
                                                        <div>
                                                            <div className="text-sm font-bold text-white">Bitcoin</div>
                                                            <div className="text-[10px] text-slate-500">BTC/USD</div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className="text-sm font-bold text-white">$64,241</div>
                                                        <div className="text-[10px] text-emerald-500">+1.24%</div>
                                                    </div>
                                                </div>
                                                <div className="h-16 w-full bg-slate-900 rounded-lg overflow-hidden relative">
                                                    <svg className="w-full h-full" preserveAspectRatio="none">
                                                        <path d="M0 64 C 20 60, 40 30, 60 40 S 100 20, 120 25" stroke="#10b981" strokeWidth="2" fill="none" />
                                                        <path d="M0 64 L 120 25 L 120 64 Z" fill="url(#grad2)" opacity="0.2" />
                                                        <defs>
                                                            <linearGradient id="grad2" x1="0" y1="0" x2="0" y2="1">
                                                                <stop offset="0%" stopColor="#10b981" />
                                                                <stop offset="100%" stopColor="transparent" />
                                                            </linearGradient>
                                                        </defs>
                                                    </svg>
                                                </div>
                                            </div>
                                            <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                                                <div className="flex justify-between items-center mb-4">
                                                    <div className="text-[10px] text-slate-500 uppercase tracking-widest">Recent Alert</div>
                                                    <div className="text-[10px] text-slate-500">2m ago</div>
                                                </div>
                                                <div className="flex gap-3">
                                                    <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-500"><Zap className="w-4 h-4" /></div>
                                                    <div>
                                                        <div className="text-xs font-bold text-white">Bullish Divergence Detected</div>
                                                        <div className="text-[10px] text-slate-500">RSI divergence on 1H timeframe</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Industry Capabilities Section */}
                <section className="py-24 bg-slate-950 border-t border-slate-900 overflow-hidden">
                    <div className="max-w-7xl mx-auto px-6 lg:px-8">
                        <div className="flex flex-col lg:flex-row items-center gap-16">
                            <div className="flex-1 space-y-8">
                                <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                                    <BarChart3 className="w-4 h-4 text-emerald-400" />
                                    <span className="text-[10px] font-black uppercase tracking-widest text-emerald-300">Market Benchmark</span>
                                </div>
                                <h2 className="text-4xl md:text-5xl font-black text-white tracking-tighter">
                                    The Intelligent <br />
                                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-blue-500">Industry Standard.</span>
                                </h2>
                                <p className="text-slate-400 text-lg leading-relaxed">
                                    Diver AI provides a high-fidelity alternative to legacy charting software. By combining Optical Pattern Recognition with a proprietary Bayesian Neural Core, we offer institutional-grade insights that traditional technical analysis misses.
                                </p>
                                <ul className="space-y-4">
                                    <li className="flex items-center gap-3 text-slate-300 font-medium">
                                        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                                        <span>Optical AI vs Standard Trendlines</span>
                                    </li>
                                    <li className="flex items-center gap-3 text-slate-300 font-medium">
                                        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                                        <span>Probabilistic vs Lagging Indicators</span>
                                    </li>
                                    <li className="flex items-center gap-3 text-slate-300 font-medium">
                                        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                                        <span>Transparent Alpha vs Black-box Signals</span>
                                    </li>
                                </ul>
                            </div>
                            <div className="flex-1 relative">
                                <div className="absolute inset-0 bg-blue-500/10 blur-[100px] rounded-full"></div>
                                <div className="relative rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-2xl overflow-hidden group">
                                    <div className="aspect-video bg-slate-950 rounded-lg flex items-center justify-center border border-slate-800 relative overflow-hidden">
                                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.1)_0%,transparent_70%)]"></div>
                                        <Activity className="w-16 h-16 text-emerald-500/20" />
                                        <div className="absolute bottom-4 left-4 right-4 h-1 bg-slate-800 rounded-full overflow-hidden">
                                            <div className="h-full bg-emerald-500 w-2/3 animate-pulse"></div>
                                        </div>
                                    </div>
                                    <div className="mt-6 space-y-2">
                                        <div className="h-2 bg-slate-800 rounded w-1/2"></div>
                                        <div className="h-2 bg-slate-800 rounded w-full opacity-50"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Final CTA */}
                <section className="py-24 max-w-7xl mx-auto px-6">
                    <div className="p-12 md:p-24 bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800 rounded-3xl relative overflow-hidden text-center space-y-8 group">
                        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay"></div>

                        <motion.h2
                            initial={{ opacity: 0, scale: 0.9 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            className="text-4xl md:text-6xl font-black text-white tracking-tighter relative z-10"
                        >
                            Initialize Sequence.
                        </motion.h2>
                        <p className="text-slate-400 max-w-xl mx-auto text-base font-medium relative z-10 leading-relaxed">
                            Join the elite tier of traders using high-fidelity Optical Pattern Recognition.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center relative z-10 pt-4">
                            <button
                                onClick={onStart}
                                className="px-10 py-5 bg-emerald-500 text-slate-950 font-black rounded-xl hover:bg-emerald-400 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)] flex items-center gap-3 text-sm uppercase tracking-widest active:scale-95"
                            >
                                Get Started <ArrowRight className="w-5 h-5" />
                            </button>
                            <p className="text-slate-500 text-[10px] font-mono uppercase">Free tier available</p>
                        </div>
                    </div>
                </section>
            </main>

            <AnimatePresence>
                {showGuide && (
                    <InstallationGuideModal
                        isOpen={showGuide}
                        onClose={() => setShowGuide(false)}
                        platform={platform}
                    />
                )}
            </AnimatePresence>

            <footer className="py-24 border-t border-slate-900 bg-[#020617]">
                <div className="max-w-7xl mx-auto px-6 lg:px-8">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-12 lg:gap-24 mb-16">

                        {/* Column 1: Brand */}
                        <div className="col-span-2 md:col-span-1 space-y-8">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-slate-900 border border-slate-800 rounded-lg flex items-center justify-center">
                                    <Activity className="text-emerald-500 w-5 h-5" />
                                </div>
                                <span className="text-xl font-black tracking-tighter text-white">Diver<span className="text-emerald-500">AI</span></span>
                            </div>
                            <p className="text-sm text-slate-500 leading-relaxed font-medium max-w-xs">
                                Institutional-grade optical pattern recognition for the independent trader.
                            </p>
                            <div className="flex gap-4">
                                <a href="#" className="w-10 h-10 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-white hover:border-slate-600 transition-all">
                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path></svg>
                                </a>
                                <a href="#" className="w-10 h-10 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-white hover:border-slate-600 transition-all">
                                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd"></path></svg>
                                </a>
                            </div>
                        </div>

                        {/* Column 2: Product */}
                        <div className="space-y-6">
                            <h4 className="text-white font-bold text-sm uppercase tracking-widest">Product</h4>
                            <ul className="space-y-4">
                                <li><button onClick={onStart} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Terminal</button></li>
                                <li>
                                    <a href="https://chromewebstore.google.com/detail/gjmicgddmbghplbdolnmiecjkdfipand?utm_source=item-share-cb" target="_blank" rel="noreferrer" className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Chrome Extension</a>
                                </li>
                                <li><button onClick={() => { setPlatform('ios'); setShowGuide(true); }} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Mobile App</button></li>
                                <li><button onClick={onOpenPricing} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Pricing</button></li>
                            </ul>
                        </div>

                        {/* Column 3: Resources */}
                        <div className="space-y-6">
                            <h4 className="text-white font-bold text-sm uppercase tracking-widest">Resources</h4>
                            <ul className="space-y-4">
                                <li><button onClick={onOpenDocs} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Documentation</button></li>
                                <li><a href="/diverai-quant-whitepaper.pdf" target="_blank" rel="noreferrer" className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">White Paper</a></li>
                                <li><a href="#" className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">API Reference</a></li>
                                <li><a href="#" className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">System Status</a></li>
                            </ul>
                        </div>

                        {/* Column 4: Legal */}
                        <div className="space-y-6">
                            <h4 className="text-white font-bold text-sm uppercase tracking-widest">Legal</h4>
                            <ul className="space-y-4">
                                <li><button onClick={() => onOpenInfo('privacy')} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Privacy Policy</button></li>
                                <li><button onClick={() => onOpenInfo('terms')} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Terms of Service</button></li>
                                <li><button onClick={() => onOpenInfo('risk')} className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Risk Disclosure</button></li>
                                <li><a href="#" className="text-sm text-slate-400 hover:text-emerald-400 transition-colors font-medium">Security</a></li>
                            </ul>
                        </div>
                    </div>

                    <div className="pt-8 border-t border-slate-900 flex flex-col md:flex-row justify-between items-center gap-6">
                        <div className="text-xs text-slate-600 font-mono flex flex-col md:flex-row items-center gap-1 md:gap-4">
                            <span>© 2026 Diver AI. All rights reserved.</span>
                            <span className="hidden md:block text-slate-800">|</span>
                            <a href="https://flisoft.agency" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-500 transition-colors">
                                Built & Published by Fli SOFT
                            </a>
                        </div>
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center gap-6">
                                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                <span className="text-xs text-slate-500 font-mono uppercase tracking-widest">System Nominal • v2.5</span>
                            </div>
                            <div className="flex flex-wrap gap-x-6 gap-y-2 text-[10px] text-slate-700 font-mono uppercase">
                                <span>Advanced Market Scanning</span>
                                <span>Neural Forecasts</span>
                                <span>AI Stock Analysis</span>
                                <span>Optical Pattern Engine</span>
                                <span>Neural Trading Signals</span>
                            </div>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
