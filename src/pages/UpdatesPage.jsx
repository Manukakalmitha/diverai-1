import React from 'react';
import { motion } from 'framer-motion';
import {
    ArrowLeft,
    Activity,
    Sparkles,
    Zap,
    Cpu,
    Eye,
    Shield,
    Rocket,
    Calendar,
    Tag
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

const updates = [
    {
        version: 'v5.1.0',
        date: 'January 14, 2026',
        title: 'Native Android Experience',
        isLatest: true,
        changes: [
            {
                type: 'feature',
                icon: Rocket,
                title: 'Signed TWA Application',
                description: 'Official signed APK release, fully compliant with Google Play Store requirements for a trusted native experience.'
            },
            {
                type: 'feature',
                icon: Zap,
                title: 'Native Integration',
                description: 'Seamless deep linking and asset link verification for a true app-like feel on Android devices.'
            },
            {
                type: 'improvement',
                icon: Sparkles,
                title: 'Streamlined Mobile UX',
                description: 'Optimizations for the Android status bar, splash screen, and navigation for a cohesive mobile environment.'
            }
        ]
    },
    {
        version: 'v5.0.0',
        date: 'January 14, 2026',
        title: 'Institutional Grade: V5 Precision',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Cpu,
                title: 'Multi-Timeframe Alignment',
                description: 'Neural core now cross-verifies signals against Daily trend bias to ensure high-probability macro alignment.'
            },
            {
                type: 'feature',
                icon: Activity,
                title: 'Institutional VWAP Integration',
                description: 'Volume-Weighted Average Price analysis identifies institutional value zones for more precise trade entries.'
            },
            {
                type: 'improvement',
                icon: Zap,
                title: 'V5 Hybrid Fusion',
                description: 'Revised Bayesian synthesis logic factorizing MTF bias and volume profile for a significant confidence boost.'
            }
        ]
    },
    {
        version: 'v4.0.0',
        date: 'January 13, 2026',
        title: 'Precision Calibration: RMSE V4',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Cpu,
                title: 'Bayesian Neural Core V4',
                description: 'Upgraded LSTM architecture with a 45-period temporal window and multivariate ATR fusion for extreme predictive precision.'
            },
            {
                type: 'feature',
                icon: Zap,
                title: 'Volatility-Adjusted Multiplier',
                description: 'Hybrid fusion logic now dynamically scales neural probability based on market volatility, reducing false signals.'
            },
            {
                type: 'improvement',
                icon: Sparkles,
                title: 'Refined OCR Preprocessing',
                description: 'New multi-stage denoising and sharpening kernels for near-perfect ticker extraction from complex chart backgrounds.'
            }
        ]
    },
    {
        version: 'v2.5.0',
        date: 'January 12, 2026',
        title: 'New Identity & PWA Evolution',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Sparkles,
                title: 'Brand Identity Refresh',
                description: 'Complete visual overhaul of Diver AI branding including a new modern logo and a professionally crafted OG image.'
            },
            {
                type: 'feature',
                icon: Rocket,
                title: 'Enhanced PWA Support',
                description: 'Full suite of high-resolution icons provided for iOS, Android, and Desktop, ensuring a premium native app experience.'
            },
            {
                type: 'improvement',
                icon: Zap,
                title: 'Asset Optimization',
                description: 'All branding assets have been precisely resized and optimized for performance and visual clarity across all devices.'
            }
        ]
    },
    {
        version: 'v2.4.0',
        date: 'January 11, 2026',
        title: 'Optical Engine Enhancement',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Eye,
                title: 'Enhanced Pattern Recognition',
                description: 'Improved accuracy in detecting complex chart patterns including double tops, head & shoulders, and wedge formations.'
            },
            {
                type: 'feature',
                icon: Cpu,
                title: 'Neural Core Optimization',
                description: 'Faster analysis processing with reduced latency for real-time market scanning.'
            },
            {
                type: 'improvement',
                icon: Zap,
                title: 'Performance Boost',
                description: 'Up to 40% faster chart rendering and smoother animations across all devices.'
            },
            {
                type: 'improvement',
                icon: Shield,
                title: 'Security Update',
                description: 'Enhanced encryption protocols and improved session management for better account security.'
            }
        ]
    },
    {
        version: 'v2.3.0',
        date: 'December 28, 2025',
        title: 'Chrome Extension Launch',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Sparkles,
                title: 'Chrome Extension Released',
                description: 'Overlay institutional analysis directly on TradingView, Yahoo Finance, and other charting platforms.'
            },
            {
                type: 'feature',
                icon: Rocket,
                title: '10-Year Price History',
                description: 'Pro users can now view comprehensive 10-year price fluctuation charts for deeper market context.'
            }
        ]
    },
    {
        version: 'v2.2.0',
        date: 'December 15, 2025',
        title: 'Pro Tier & PDF Reports',
        isLatest: false,
        changes: [
            {
                type: 'feature',
                icon: Sparkles,
                title: 'Pro Subscription Tier',
                description: 'Unlimited scans, priority analysis, and advanced features for professional traders.'
            },
            {
                type: 'feature',
                icon: Sparkles,
                title: 'PDF Export',
                description: 'Export your analysis results as professional PDF reports with full branding.'
            }
        ]
    }
];

const UpdatesPage = () => {

    return (
        <div className="min-h-screen bg-black text-white">
            <Helmet>
                <title>Product Updates & Changelog | Diver AI</title>
                <meta name="description" content="Stay up to date with the latest Diver AI features, improvements, and fixes. See what's new in the AI trading platform." />
                <meta property="og:title" content="Diver AI Updates | What's New" />
                <meta property="og:description" content="Latest features and improvements to the Diver AI trading analysis platform." />
            </Helmet>
            {/* Header */}
            <div className="border-b border-slate-800 bg-black">
                <div className="max-w-4xl mx-auto px-6 py-12">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-brand transition-colors mb-6"
                    >
                        <ArrowLeft className="w-4 h-4" /> Back to Home
                    </Link>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-4"
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-brand/10 border border-brand/20 rounded-full">
                            <Sparkles className="w-3 h-3 text-brand" />
                            <span className="text-[10px] font-bold uppercase tracking-widest text-brand">What's New</span>
                        </div>
                        <h1 className="text-4xl md:text-5xl font-black tracking-tight">
                            Product Updates
                        </h1>
                        <p className="text-slate-400 text-lg max-w-2xl">
                            Stay up to date with the latest features, improvements, and fixes to the Diver AI platform.
                        </p>
                    </motion.div>
                </div>
            </div>

            {/* Updates List */}
            <div className="max-w-4xl mx-auto px-6 py-16 space-y-16">
                {updates.map((update, index) => (
                    <motion.div
                        key={update.version}
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="relative"
                    >
                        {/* Version Header */}
                        <div className="flex flex-wrap items-center gap-4 mb-8">
                            <div className="flex items-center gap-3">
                                <div className={`px-3 py-1.5 rounded-lg font-mono text-sm font-bold ${update.isLatest
                                    ? 'bg-brand/20 text-brand border border-brand/30'
                                    : 'bg-slate-800 text-slate-300 border border-slate-700'
                                    }`}>
                                    {update.version}
                                </div>
                                {update.isLatest && (
                                    <span className="px-2 py-0.5 bg-brand text-slate-950 text-[10px] font-black uppercase tracking-wider rounded-full">
                                        Latest
                                    </span>
                                )}
                            </div>
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                                <Calendar className="w-4 h-4" />
                                {update.date}
                            </div>
                        </div>

                        <h2 className="text-2xl font-bold text-white mb-6">{update.title}</h2>

                        {/* Changes Grid */}
                        <div className="grid gap-4">
                            {update.changes.map((change, changeIndex) => (
                                <div
                                    key={changeIndex}
                                    className="p-6 bg-black-ash/50 border border-slate-800 rounded-xl hover:border-slate-700 transition-colors group"
                                >
                                    <div className="flex gap-4">
                                        <div className={`shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${change.type === 'feature'
                                            ? 'bg-brand/10 text-brand'
                                            : 'bg-brand-dark/10 text-brand-dark'
                                            }`}>
                                            <change.icon className="w-5 h-5" />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex items-center gap-2">
                                                <h3 className="font-bold text-white group-hover:text-brand transition-colors">
                                                    {change.title}
                                                </h3>
                                                <span className={`px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded ${change.type === 'feature'
                                                    ? 'bg-brand/10 text-brand'
                                                    : 'bg-brand-dark/10 text-brand-dark'
                                                    }`}>
                                                    {change.type}
                                                </span>
                                            </div>
                                            <p className="text-slate-400 text-sm leading-relaxed">
                                                {change.description}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Divider */}
                        {index < updates.length - 1 && (
                            <div className="absolute -bottom-8 left-0 right-0 h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />
                        )}
                    </motion.div>
                ))}
            </div>

            {/* CTA Section */}
            <div className="border-t border-slate-800 bg-black">
                <div className="max-w-4xl mx-auto px-6 py-16 text-center space-y-6">
                    <h3 className="text-2xl font-bold text-white">Ready to try the latest features?</h3>
                    <p className="text-slate-400">Experience institutional-grade market analysis with the enhanced Optical Engine.</p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <Link
                            to="/analysis"
                            className="btn-flame px-8 !py-4"
                        >
                            Launch Terminal <Rocket className="w-4 h-4" />
                        </Link>
                        <Link
                            to="/pricing"
                            className="px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white font-semibold rounded-lg transition-all"
                        >
                            View Plans
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UpdatesPage;
