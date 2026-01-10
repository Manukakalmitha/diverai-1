import React from 'react';
import { Shield, Zap, Target, Search, HelpCircle, Activity, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const WhatIsDiverAI = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans selection:bg-blue-500/30 pt-20 pb-20">
            <div className="max-w-4xl mx-auto px-6">

                {/* Header */}
                <header className="mb-16 text-center space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400 text-xs font-semibold uppercase tracking-widest">
                        <Activity className="w-4 h-4" /> Official Definition
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
                        What Is <span className="text-blue-500">Diver AI</span>?
                    </h1>
                    <p className="text-xl text-slate-400 font-medium max-w-2xl mx-auto leading-relaxed">
                        AI Trading & Optical Pattern Recognition Explained
                    </p>
                </header>

                <main className="space-y-20">

                    {/* What It Is */}
                    <section className="bg-slate-900 border border-slate-800 rounded-2xl p-8 md:p-12 relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-20 bg-blue-500/5 rounded-full blur-[80px]"></div>
                        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                            <Zap className="w-6 h-6 text-blue-500" />
                            What Diver AI Is
                        </h2>
                        <p className="text-lg text-slate-300 leading-relaxed mb-6">
                            <strong>Diver AI</strong> is an institutional-grade <strong>trading intelligence platform</strong>. Unlike traditional bots that rely on simple moving averages, Diver AI uses <strong>Optical Pattern Recognition (OPR)</strong> to "see" charts like a human analyst.
                        </p>
                        <p className="text-lg text-slate-300 leading-relaxed">
                            It combines this visual data with a <strong>Bayesian Neural Network</strong> to calculate the statistical probability of future price movements in stocks, crypto, and forex markets.
                        </p>
                    </section>

                    {/* What It Is NOT */}
                    <section className="bg-slate-900 border border-slate-800 rounded-2xl p-8 md:p-12 relative overflow-hidden">
                        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                            <X className="w-6 h-6 text-rose-500" />
                            What It Is NOT
                        </h2>
                        <ul className="space-y-4">
                            <li className="flex items-start gap-4">
                                <div className="p-1.5 bg-rose-500/10 rounded-lg mt-1"><X className="w-3 h-3 text-rose-500" /></div>
                                <p className="text-slate-300 text-lg"><strong>Not a Scuba Tool:</strong> We have no association with diving equipment, underwater exploration, or deep-sea diving software.</p>
                            </li>
                            <li className="flex items-start gap-4">
                                <div className="p-1.5 bg-rose-500/10 rounded-lg mt-1"><X className="w-3 h-3 text-rose-500" /></div>
                                <p className="text-slate-300 text-lg"><strong>Not a Magic Button:</strong> It is a decision-support system, not an "auto-money" generator. It enhances skill, it doesn't replace risk management.</p>
                            </li>
                        </ul>
                    </section>

                    {/* How It Works */}
                    <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 hover:border-blue-500/30 transition-colors">
                            <Search className="w-8 h-8 text-blue-500 mb-6" />
                            <h3 className="text-xl font-bold text-white mb-4">1. Optical Scan</h3>
                            <p className="text-slate-400 leading-relaxed">
                                The engine takes a snapshot of your chart and uses Computer Vision to identify geometric structures (Head & Shoulders, Flags, Wedges) and key support/resistance zones.
                            </p>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 hover:border-blue-500/30 transition-colors">
                            <Target className="w-8 h-8 text-blue-500 mb-6" />
                            <h3 className="text-xl font-bold text-white mb-4">2. Probability Fusion</h3>
                            <p className="text-slate-400 leading-relaxed">
                                It compares the current pattern against 10 years of historical data (1M+ scenarios) to output a <strong>probabilistic forecast</strong> (e.g., "78% chance of upward breakout").
                            </p>
                        </div>
                    </section>

                    {/* Who It's For */}
                    <section className="bg-slate-900 border border-slate-800 rounded-2xl p-8 md:p-12 text-center">
                        <h2 className="text-2xl font-bold text-white mb-6">Who Is Diver AI For?</h2>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
                            <div className="p-6 bg-slate-950 rounded-xl border border-slate-800">
                                <h4 className="font-bold text-white mb-2">Day Traders</h4>
                                <p className="text-sm text-slate-400">For rapid confirmation of scalping setups.</p>
                            </div>
                            <div className="p-6 bg-slate-950 rounded-xl border border-slate-800">
                                <h4 className="font-bold text-white mb-2">Swing Traders</h4>
                                <p className="text-sm text-slate-400">To find high-timeframe structural biases.</p>
                            </div>
                            <div className="p-6 bg-slate-950 rounded-xl border border-slate-800">
                                <h4 className="font-bold text-white mb-2">Crypto Investors</h4>
                                <p className="text-sm text-slate-400">To navigate volatile altcoin markets with data.</p>
                            </div>
                        </div>
                    </section>

                    {/* FAQs */}
                    <section className="space-y-8">
                        <div className="flex items-center gap-3 mb-6">
                            <HelpCircle className="w-5 h-5 text-slate-500" />
                            <h2 className="text-xl font-bold text-white uppercase tracking-tight">Quick FAQs</h2>
                        </div>
                        <div className="space-y-4">
                            <div className="p-6 bg-slate-900 border border-slate-800 rounded-xl">
                                <h4 className="font-bold text-white mb-2">Is Diver AI free?</h4>
                                <p className="text-slate-400">We offer a free tier with daily limits using our core OCR engine. Pro Quant plans unlock unlimited scans and institutional data feeds.</p>
                            </div>
                            <div className="p-6 bg-slate-900 border border-slate-800 rounded-xl">
                                <h4 className="font-bold text-white mb-2">Does it place trades for me?</h4>
                                <p className="text-slate-400">No. Diver AI is a <strong>non-custodial intelligence tool</strong>. We provide the map; you drive the car.</p>
                            </div>
                        </div>
                    </section>

                </main>

                <footer className="mt-24 text-center">
                    <button
                        onClick={() => navigate('/')}
                        className="px-8 py-3.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg transition-colors shadow-lg shadow-blue-500/20"
                    >
                        Explore The Platform
                    </button>
                </footer>

            </div>
        </div>
    );
};

export default WhatIsDiverAI;
