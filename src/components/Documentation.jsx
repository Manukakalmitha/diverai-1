import React, { useState } from 'react';
import { BookOpen, AlertTriangle, CheckCircle, BarChart3, Scan, Cpu, X, Search, ChevronRight, Hash, Menu, FileText, Download } from 'lucide-react';

const Documentation = ({ onClose }) => {
    const [activeSection, setActiveSection] = useState('getting-started');

    const scrollToSection = (id) => {
        setActiveSection(id);
        const element = document.getElementById(id);
        if (element) element.scrollIntoView({ behavior: 'smooth' });
    };

    const sections = [
        { id: 'getting-started', title: 'Getting Started' },
        { id: 'supported-assets', title: 'Supported Assets' },
        { id: 'prediction-validity', title: 'Prediction Validity' },
        { id: 'interpreting-signals', title: 'Interpreting Signals' },
    ];

    return (
        <div className="fixed inset-0 z-40 top-16 bg-[#050B14] text-white flex overflow-hidden font-sans">
            {/* Left Sidebar */}
            <aside className="w-64 border-r border-slate-800 bg-[#050B14] hidden md:flex flex-col">
                <div className="p-4 border-b border-slate-800 flex items-center gap-3">
                    <BookOpen className="w-5 h-5 text-emerald-500" />
                    <span className="font-bold tracking-tight">Diver Docs</span>
                </div>

                <div className="p-4">
                    <div className="relative">
                        <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search..."
                            className="w-full bg-slate-900 border border-slate-800 rounded-lg py-2 pl-9 pr-4 text-xs focus:outline-none focus:border-emerald-500 text-slate-300"
                        />
                        <span className="absolute right-3 top-2.5 text-[10px] text-slate-600 font-mono">Ctrl K</span>
                    </div>
                </div>

                <nav className="flex-1 overflow-y-auto px-4 space-y-6">
                    <div>
                        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 px-2">Core Concepts</h3>
                        <div className="space-y-1">
                            {sections.map(section => (
                                <button
                                    key={section.id}
                                    onClick={() => scrollToSection(section.id)}
                                    className={`w-full text-left px-2 py-1.5 rounded-md text-sm transition-colors ${activeSection === section.id ? 'bg-emerald-500/10 text-emerald-400 font-medium' : 'text-slate-400 hover:text-white hover:bg-slate-900'}`}
                                >
                                    {section.title}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div>
                        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3 px-2">Reference</h3>
                        <div className="space-y-1">
                            {['API Reference', 'Calculations', 'Changelog'].map(item => (
                                <button key={item} className="w-full text-left px-2 py-1.5 rounded-md text-sm text-slate-400 hover:text-white hover:bg-slate-900 transition-colors">
                                    {item}
                                </button>
                            ))}
                        </div>
                    </div>
                </nav>

                <div className="p-4 border-t border-slate-800">
                    <button onClick={onClose} className="flex items-center gap-2 text-xs text-slate-500 hover:text-white transition-colors">
                        <X className="w-4 h-4" /> Exit Documentation
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto relative scroll-smooth bg-[#050B14]">
                {/* Mobile Header */}
                <div className="md:hidden p-4 border-b border-slate-800 flex justify-between items-center sticky top-0 bg-[#050B14]/90 backdrop-blur z-20">
                    <div className="flex items-center gap-2">
                        <BookOpen className="w-5 h-5 text-emerald-500" />
                        <span className="font-bold">Docs</span>
                    </div>
                    <button onClick={onClose}><X className="w-5 h-5 text-slate-400" /></button>
                </div>

                <div className="max-w-4xl mx-auto px-8 py-12 md:py-16">
                    <div className="mb-12">
                        <div className="flex items-center gap-2 text-emerald-500 text-xs font-bold uppercase tracking-widest mb-4">
                            Introduction
                        </div>
                        <h1 className="text-4xl md:text-5xl font-black text-white mb-6 tracking-tighter">Quantitative Manual</h1>
                        <p className="text-xl text-slate-400 leading-relaxed font-light">
                            Standard Operating Procedures (v3.2.1-STABLE) for the Diver AI forecasting terminal.
                        </p>
                    </div>

                    <div className="space-y-24">
                        {/* Section 1: Introduction */}
                        <section id="getting-started" className="scroll-mt-24 space-y-8">
                            <div className="pb-4 border-b border-slate-800">
                                <h2 className="text-2xl font-bold text-white mb-2">Getting Started</h2>
                                <p className="text-slate-500 text-sm">Understanding the core architecture.</p>
                            </div>

                            <p className="text-slate-300 leading-relaxed">
                                Diver AI is an institutional-grade forecasting terminal utilizing a <strong>Bayesian-LSTM Hybrid Architecture</strong>. Our engine fuses <strong>Optical Pattern Recognition (OPR)</strong> with multi-layer neural networks to identify high-probability price structures with statistical precision.
                            </p>

                            <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-500"></div>
                                <div className="p-8">
                                    <h3 className="text-white font-bold mb-6 uppercase tracking-wider text-xs flex items-center gap-2">
                                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" /> Execution Pipeline
                                    </h3>
                                    <ol className="relative border-l border-slate-800 ml-3 space-y-8">
                                        <li className="ml-6">
                                            <span className="absolute -left-1.5 flex h-3 w-3 items-center justify-center rounded-full bg-slate-800 ring-4 ring-[#050B14]">
                                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                                            </span>
                                            <h4 className="flex items-center mb-1 text-sm font-bold text-white">Ingestion</h4>
                                            <p className="text-sm text-slate-400">Optical capture of raw chart data via singleton OCR engine. Baseline price and ticker extraction.</p>
                                        </li>
                                        <li className="ml-6">
                                            <span className="absolute -left-1.5 flex h-3 w-3 items-center justify-center rounded-full bg-slate-800 ring-4 ring-[#050B14]">
                                                <div className="w-1.5 h-1.5 rounded-full bg-slate-600"></div>
                                            </span>
                                            <h4 className="flex items-center mb-1 text-sm font-bold text-white">Cross-Verification</h4>
                                            <p className="text-sm text-slate-400">Parallel data synchronization across Yahoo Finance, Finnhub, and CoinGecko to eliminate data divergence.</p>
                                        </li>
                                        <li className="ml-6">
                                            <span className="absolute -left-1.5 flex h-3 w-3 items-center justify-center rounded-full bg-slate-800 ring-4 ring-[#050B14]">
                                                <div className="w-1.5 h-1.5 rounded-full bg-slate-600"></div>
                                            </span>
                                            <h4 className="flex items-center mb-1 text-sm font-bold text-white">Neural Fusion</h4>
                                            <p className="text-sm text-slate-400">Bayesian inference computes the weighted mean of LSTM neural predictions, geometric pattern analysis, and technical oscillator signatures.</p>
                                        </li>
                                    </ol>
                                </div>
                            </div>
                        </section>

                        {/* Section 2: Supported Assets */}
                        <section id="supported-assets" className="scroll-mt-24 space-y-8">
                            <div className="pb-4 border-b border-slate-800">
                                <h2 className="text-2xl font-bold text-white mb-2">Supported Assets</h2>
                                <p className="text-slate-500 text-sm">Where the engine performs best.</p>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div className="p-6 bg-slate-900/30 border border-slate-800 rounded-xl hover:border-emerald-500/30 transition-colors">
                                    <div className="w-8 h-8 bg-emerald-500/10 rounded-lg flex items-center justify-center mb-4">
                                        <CheckCircle className="w-4 h-4 text-emerald-500" />
                                    </div>
                                    <h3 className="text-white font-bold mb-2">Optimal Performance</h3>
                                    <p className="text-slate-400 text-sm mb-4 leading-relaxed">The engine is tuned for high-volume, high-liquidity assets where technical patterns are statistically significant.</p>
                                    <div className="flex flex-wrap gap-2">
                                        {['BTC', 'ETH', 'SOL', 'AAPL', 'TSLA', 'NVDA'].map(t => (
                                            <span key={t} className="px-2 py-1 bg-slate-800 rounded text-[10px] font-mono text-emerald-400 font-bold border border-slate-700">{t}</span>
                                        ))}
                                    </div>
                                </div>
                                <div className="p-6 bg-slate-900/30 border border-slate-800 rounded-xl hover:border-amber-500/30 transition-colors">
                                    <div className="w-8 h-8 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4">
                                        <AlertTriangle className="w-4 h-4 text-amber-500" />
                                    </div>
                                    <h3 className="text-white font-bold mb-2">System Limitations</h3>
                                    <ul className="text-slate-400 text-sm space-y-2 list-disc pl-4 marker:text-amber-500">
                                        <li>Blurry or low-resolution images may fail OCR.</li>
                                        <li>Low volume altcoins (&lt;$10M) show erratic variance.</li>
                                        <li>Heavily cluttered charts reduce vision accuracy.</li>
                                    </ul>
                                </div>
                            </div>
                        </section>

                        {/* Section 3: Prediction Validity */}
                        <section id="prediction-validity" className="scroll-mt-24 space-y-8">
                            <div className="pb-4 border-b border-slate-800">
                                <h2 className="text-2xl font-bold text-white mb-2">Prediction Validity</h2>
                                <p className="text-slate-500 text-sm">Understanding the lifespan of a forecast.</p>
                            </div>

                            <p className="text-slate-300 leading-relaxed">
                                Forensics generated by Diver AI are <strong>timeframe-sensitive snapshots</strong>. To maintain the highest level of capital safety, traders must adhere to the following validity protocols:
                            </p>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl">
                                    <div className="text-emerald-500 font-bold text-xs uppercase mb-3 tracking-tighter italic">Next-Candle Rule</div>
                                    <h4 className="text-white font-bold text-sm mb-2">Temporal Window</h4>
                                    <p className="text-xs text-slate-400 leading-relaxed">The Neural Core (LSTM) minimizes loss by targeting the immediate next candle of your current timeframe.</p>
                                </div>
                                <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl">
                                    <div className="text-emerald-500 font-bold text-xs uppercase mb-3 tracking-tighter italic">Setup Decay</div>
                                    <h4 className="text-white font-bold text-sm mb-2">ATR Logic</h4>
                                    <p className="text-xs text-slate-400 leading-relaxed">Trade protocols are valid for 3-5 candles. If targets aren't reached within this window, the setup is considered neutral.</p>
                                </div>
                                <div className="p-5 bg-slate-900/40 border border-slate-800 rounded-xl">
                                    <div className="text-emerald-500 font-bold text-xs uppercase mb-3 tracking-tighter italic">Alpha Drift</div>
                                    <h4 className="text-white font-bold text-sm mb-2">Dynamic Refresh</h4>
                                    <p className="text-xs text-slate-400 leading-relaxed">If market price diverges &gt;1 ATR from the Entry Zone shown in the report, a full system rescan is mandatory.</p>
                                </div>
                            </div>
                        </section>

                        {/* Section 4: Interpreting Results */}
                        <section id="interpreting-signals" className="scroll-mt-24 space-y-8">
                            <div className="pb-4 border-b border-slate-800">
                                <h2 className="text-2xl font-bold text-white mb-2">Interpreting Signals</h2>
                                <p className="text-slate-500 text-sm">Decoding the Directional Bias Matrix.</p>
                            </div>

                            <div className="prose prose-invert max-w-none text-slate-300">
                                <p>The <strong>Directional Bias Matrix</strong> is derived from four critical weight layers:</p>
                                <div className="grid sm:grid-cols-2 gap-4 mt-6 not-prose">
                                    <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg">
                                        <div className="text-xs font-mono text-emerald-500 mb-1">Layer 1 (ω)</div>
                                        <div className="font-bold text-white text-sm">Neural LSTM</div>
                                        <div className="text-xs text-slate-500 mt-1">Temporal feature extraction</div>
                                    </div>
                                    <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg">
                                        <div className="text-xs font-mono text-emerald-500 mb-1">Layer 2 (α)</div>
                                        <div className="font-bold text-white text-sm">Geometric Pattern</div>
                                        <div className="text-xs text-slate-500 mt-1">Triangle/Flag confluence</div>
                                    </div>
                                    <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg">
                                        <div className="text-xs font-mono text-emerald-500 mb-1">Layer 3 (γ)</div>
                                        <div className="font-bold text-white text-sm">Technical</div>
                                        <div className="text-xs text-slate-500 mt-1">RSI/MACD convergence</div>
                                    </div>
                                    <div className="p-4 bg-slate-900/30 border border-slate-800 rounded-lg">
                                        <div className="text-xs font-mono text-emerald-500 mb-1">Layer 4 (M)</div>
                                        <div className="font-bold text-white text-sm">Macro</div>
                                        <div className="text-xs text-slate-500 mt-1">Institutional trend alignment</div>
                                    </div>
                                </div>
                                <div className="mt-8 p-6 bg-slate-900 border-l-2 border-emerald-500 rounded-r-xl">
                                    <div className="text-emerald-400 font-bold uppercase tracking-widest text-xs mb-2 flex items-center gap-2">
                                        <Cpu className="w-4 h-4" /> Operational Note
                                    </div>
                                    <p className="text-sm text-slate-400 m-0">
                                        Targets and protocols are generated using clinical-grade risk management algorithms. Do not deviate from the identified TP/SL structures regardless of market sentiment.
                                    </p>
                                </div>
                            </div>
                        </section>
                    </div>

                    <footer className="mt-32 pt-12 border-t border-slate-800 pb-12">
                        <div className="flex justify-between items-center text-xs text-slate-500">
                            <div>Last updated: Jan 07, 2026</div>
                            <div>© 2026 Diver AI by <a href="https://flisoft.agency" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-500 transition-colors">Fli SOFT</a></div>
                        </div>
                    </footer>
                </div>
            </main>

            {/* Right TOC */}
            <aside className="w-64 border-l border-slate-800 bg-[#050B14] hidden xl:block p-6">
                <div className="fixed w-52">
                    <h5 className="text-xs font-bold text-white mb-4 flex items-center gap-2">
                        <Menu className="w-3 h-3 text-emerald-500" /> On this page
                    </h5>
                    <div className="space-y-1 relative border-l border-slate-800 ml-1.5">
                        {sections.map(section => (
                            <button
                                key={section.id}
                                onClick={() => scrollToSection(section.id)}
                                className={`block w-full text-left pl-4 py-1 text-xs border-l -ml-[1px] transition-colors ${activeSection === section.id ? 'text-emerald-400 border-emerald-500 font-medium' : 'text-slate-500 border-transparent hover:text-slate-300'}`}
                            >
                                {section.title}
                            </button>
                        ))}
                    </div>

                    <div className="mt-8 pt-8 border-t border-slate-800">
                        <h5 className="text-xs font-bold text-white mb-4">Other Resources</h5>
                        <ul className="space-y-3 text-xs text-slate-500">
                            <li className="hover:text-emerald-400 cursor-pointer transition-colors">Risk Disclaimer</li>
                            <li>
                                <a
                                    href="/diverai-quant-whitepaper.pdf"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="hover:text-emerald-400 transition-colors flex items-center gap-2"
                                >
                                    <FileText className="w-3 h-3" /> Technical White Paper
                                </a>
                            </li>
                            <li className="hover:text-emerald-400 cursor-pointer transition-colors">API Status</li>
                            <li className="hover:text-emerald-400 cursor-pointer transition-colors">Contact Support</li>
                        </ul>
                    </div>
                </div>
            </aside>
        </div>
    );
};

export default Documentation;
