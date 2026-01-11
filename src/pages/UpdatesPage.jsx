import React from 'react';
import { motion } from 'framer-motion';
import {
    ArrowLeft,
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

const updates = [
    {
        version: 'v2.4.0',
        date: 'January 11, 2026',
        title: 'Optical Engine Enhancement',
        isLatest: true,
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
        <div className="min-h-screen bg-[#050B14] text-white">
            {/* Header */}
            <div className="border-b border-slate-800 bg-[#020617]">
                <div className="max-w-4xl mx-auto px-6 py-12">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-emerald-400 transition-colors mb-6"
                    >
                        <ArrowLeft className="w-4 h-4" /> Back to Home
                    </Link>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-4"
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                            <Sparkles className="w-3 h-3 text-emerald-400" />
                            <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-300">What's New</span>
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
                                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                                    }`}>
                                    {update.version}
                                </div>
                                {update.isLatest && (
                                    <span className="px-2 py-0.5 bg-emerald-500 text-slate-950 text-[10px] font-black uppercase tracking-wider rounded-full">
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
                                    className="p-6 bg-slate-900/50 border border-slate-800 rounded-xl hover:border-slate-700 transition-colors group"
                                >
                                    <div className="flex gap-4">
                                        <div className={`shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${change.type === 'feature'
                                                ? 'bg-emerald-500/10 text-emerald-400'
                                                : 'bg-blue-500/10 text-blue-400'
                                            }`}>
                                            <change.icon className="w-5 h-5" />
                                        </div>
                                        <div className="space-y-1">
                                            <div className="flex items-center gap-2">
                                                <h3 className="font-bold text-white group-hover:text-emerald-400 transition-colors">
                                                    {change.title}
                                                </h3>
                                                <span className={`px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded ${change.type === 'feature'
                                                        ? 'bg-emerald-500/10 text-emerald-400'
                                                        : 'bg-blue-500/10 text-blue-400'
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
            <div className="border-t border-slate-800 bg-[#020617]">
                <div className="max-w-4xl mx-auto px-6 py-16 text-center space-y-6">
                    <h3 className="text-2xl font-bold text-white">Ready to try the latest features?</h3>
                    <p className="text-slate-400">Experience institutional-grade market analysis with the enhanced Optical Engine.</p>
                    <div className="flex flex-wrap justify-center gap-4">
                        <Link
                            to="/analysis"
                            className="px-8 py-4 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-all flex items-center gap-2"
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
