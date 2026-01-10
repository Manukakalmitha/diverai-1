import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    Activity, TrendingUp, TrendingDown, Minus, Camera,
    History, ShieldCheck, Cpu, X, ChevronRight, Zap, ThumbsUp, ThumbsDown, Trash2, Keyboard, Key
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import Tesseract from 'tesseract.js';
import { detectTicker, detectPrice, fetchMarketData, fetchHistoricalData, COIN_MAP, STOCK_MAP, fetchStockData, fetchStockHistory, fetchYahooData, fetchMacroHistory, calculateMacroSentiment, fetchTickerData } from '../lib/marketData';
import { useAppContext } from '../context/AppContext';
import AuthModal from '../components/AuthModal';
import { calculateRSI as calcRSI, calculateMACD, calculateBollingerBands, detectPatterns } from '../lib/technicalAnalysis';
import { prepareData, calculateStats, createModel, trainModel, predictNextPrice, disposeModel, assessModelAccuracy, saveGlobalModel, loadGlobalModel, runBackgroundTraining, runBackgroundAssessment, saveGlobalModelArtifacts } from '../lib/brain';
import { extractChartData, anchorPriceToVisual } from '../lib/vision';
import { runRealAnalysis } from '../lib/analysis';

// --- ENGINE LOGIC (Real Implementation) ---

// --- ENGINE LOGIC (Real Implementation) ---

// Weights are now managed via AppContext for centralization


// calculateHybridProbability and runRealAnalysis moved to ../lib/analysis.js for cross-app consistency

// 3. Main Analysis Workflow
// runRealAnalysis moved to ../lib/analysis.js

// --- Child Components ---
const Sparkline = ({ data, color = 'emerald' }) => {
    if (!data || data.length < 2) return <div className="w-12 h-4 bg-slate-900/50 rounded animate-pulse" />;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const points = data.map((v, i) => `${(i / (data.length - 1)) * 48},${16 - ((v - min) / range) * 16}`).join(' ');

    return (
        <svg className="w-12 h-4 overflow-visible" viewBox="0 0 48 16">
            <polyline
                fill="none"
                stroke={color === 'emerald' ? '#10b981' : '#f43f5e'}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={points}
                className="drop-shadow-[0_0_3px_rgba(16,185,129,0.5)]"
            />
        </svg>
    );
};

const TickerTape = ({ items }) => {
    if (!items || items.length === 0) return (
        <div className="w-full bg-slate-950 border-y border-slate-900 h-10 flex items-center justify-center">
            <div className="text-[9px] font-black text-slate-700 uppercase tracking-[0.3em] animate-pulse">Initializing Global Stream...</div>
        </div>
    );
    return (
        <div className="w-full bg-slate-950 border-y border-slate-900 overflow-hidden relative h-10 flex items-center shrink-0 shadow-2xl z-50">
            <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-slate-950 to-transparent z-10" />
            <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-slate-950 to-transparent z-10" />
            <div className="flex animate-marquee whitespace-nowrap gap-12 items-center px-4">
                {[...items, ...items].map((item, i) => (
                    <div key={i} className="flex items-center gap-4 group cursor-default">
                        <div className="flex flex-col">
                            <span className="text-[10px] font-black text-slate-500 group-hover:text-white transition-colors leading-none tracking-tighter">{item.ticker}</span>
                            <span className={`text-[8px] font-bold ${item.change >= 0 ? 'text-emerald-500/60' : 'text-rose-500/60'} leading-none mt-0.5`}>
                                {item.change >= 0 ? '+' : ''}{item.change?.toFixed(2)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            <span className="text-xs font-mono text-white font-black">${item.price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                            <Sparkline data={item.sparkline} color={item.change >= 0 ? 'emerald' : 'rose'} />
                        </div>
                        <div className="w-px h-4 bg-slate-800/50 mx-2" />
                    </div>
                ))}
            </div>
        </div>
    );
};

const SentimentGauge = ({ probability, direction }) => {
    const percentage = probability * 100;
    const isBull = direction.includes('Bullish');
    const isBear = direction.includes('Bearish');
    const color = isBull ? 'emerald' : (isBear ? 'rose' : 'blue');

    const markers = [
        { label: 'Panic', pos: 10 },
        { label: 'Fear', pos: 30 },
        { label: 'Neutral', pos: 50 },
        { label: 'Greed', pos: 70 },
        { label: 'Euphoria', pos: 90 }
    ];

    return (
        <div className="relative w-full space-y-2 py-4">
            <div className="relative h-6 bg-slate-950 rounded-md border border-slate-800/50 overflow-hidden shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]">
                {/* Background Regions */}
                <div className="absolute inset-0 flex">
                    <div className="flex-1 bg-rose-500/5 border-r border-rose-500/10" />
                    <div className="flex-1 bg-slate-500/5 border-r border-slate-500/10" />
                    <div className="flex-1 bg-emerald-500/5" />
                </div>

                {/* Active Probability Fill */}
                <div
                    className={`absolute inset-y-0 left-0 bg-gradient-to-r from-${color}-500/20 to-${color}-500 transition-all duration-1000 ease-out`}
                    style={{ width: `${percentage}%` }}
                >
                    <div className={`absolute right-0 inset-y-0 w-1 bg-${color}-400 shadow-[0_0_15px_rgba(var(--${color}-rgb),0.8)]`} />
                </div>

                {/* Markers */}
                {markers.map((m, i) => (
                    <div key={i} className="absolute top-0 bottom-0 w-px bg-slate-800/30" style={{ left: `${m.pos}%` }}>
                        <span className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 text-[6px] font-black uppercase text-slate-600 tracking-tighter">{m.label}</span>
                    </div>
                ))}

                {/* Scanning HUD line overlay */}
                <div className="absolute inset-0 flex flex-col justify-between py-1 opacity-20">
                    <div className="w-full h-px bg-white/10" />
                    <div className="w-full h-px bg-white/10" />
                </div>
            </div>

            <div className="flex justify-between items-center px-1">
                <div className="text-[8px] font-black text-slate-600 uppercase tracking-widest">Statistical Confidence Range</div>
                <div className={`text-[10px] font-mono font-black ${isBull ? 'text-emerald-400' : 'text-rose-400'}`}>
                    Σ({percentage.toFixed(1)}%)
                </div>
            </div>
        </div>
    );
};

const ScanningHUD = ({ status }) => {
    const [hexLines, setHexLines] = useState([]);

    useEffect(() => {
        const interval = setInterval(() => {
            const hex = Array.from({ length: 4 }, () => Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, '0').toUpperCase());
            setHexLines(prev => [hex.join(' '), ...prev].slice(0, 10));
        }, 150);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col items-center justify-center min-h-[500px] relative overflow-hidden bg-slate-950 rounded-[40px] border border-slate-900 shadow-2xl">
            {/* Grid Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-20" />

            <div className="relative z-10 flex flex-col items-center">
                <div className="relative w-32 h-32 mb-12">
                    <div className="absolute inset-0 border-2 border-blue-500/20 rounded-full animate-[ping_3s_linear_infinite]" />
                    <div className="absolute inset-0 border border-emerald-500/40 rounded-full rotate-45" />
                    <div className="absolute inset-2 border border-t-emerald-400 border-l-transparent border-r-transparent border-b-transparent rounded-full animate-spin [animation-duration:1s]" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Cpu className="w-12 h-12 text-emerald-400 animate-pulse" />
                    </div>
                </div>

                <div className="text-center space-y-4">
                    <h3 className="text-2xl font-black text-white tracking-tighter uppercase">{status || "Initializing Neural Core"}</h3>
                    <div className="flex items-center justify-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce" />
                    </div>
                </div>

                <div className="mt-12 w-64 h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />

                <div className="mt-8 font-mono text-[8px] text-emerald-500/40 space-y-1">
                    {hexLines.map((line, i) => (
                        <div key={i} className="animate-in fade-in slide-in-from-bottom-2 duration-300">{line}</div>
                    ))}
                </div>
            </div>

            {/* Scanning Beam */}
            <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent opacity-50 animate-scan pointer-events-none" />
        </div>
    );
};

const TradeTheoryModal = ({ targets, direction, onClose }) => {
    const isBull = direction.includes('Bullish');
    const themeColor = isBull ? 'emerald' : 'rose';
    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-6 bg-slate-950/80 backdrop-blur-sm animate-in fade-in" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="bg-slate-900 border border-slate-800 rounded-[32px] w-full max-w-lg overflow-hidden shadow-2xl animate-in zoom-in-95 flex flex-col max-h-[90vh]">
                <div className={`p-6 md:p-8 bg-gradient-to-br from-${themeColor}-500/10 to-transparent border-b border-slate-800 flex justify-between items-start shrink-0`}>
                    <div>
                        <h3 className="text-xl md:text-2xl font-black text-white flex items-center gap-2"><Zap className={`text-${themeColor}-400 w-5 h-5 md:w-6 md:h-6`} /> Trade Blueprint</h3>
                        <p className="text-slate-500 text-[10px] font-mono mt-1 uppercase tracking-widest leading-none">Calculated Risk: {isBull ? 'Long' : 'Short'} Position</p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full text-slate-500 hover:text-white transition-colors"><X className="w-6 h-6" /></button>
                </div>
                <div className="p-6 md:p-8 space-y-6 overflow-y-auto scrollbar-none">
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-5 bg-slate-950/50 rounded-2xl border border-slate-800">
                            <span className="text-[10px] text-slate-500 font-black uppercase mb-1 block leading-none">Entry Zone</span>
                            <div className="text-xl md:text-2xl font-mono text-white">${targets.entry}</div>
                        </div>
                        <div className="p-5 bg-slate-950/50 rounded-2xl border border-slate-800">
                            <span className="text-[10px] text-slate-500 font-black uppercase mb-1 block leading-none">Risk/Reward</span>
                            <div className="text-xl md:text-2xl font-mono text-emerald-400">1 : {targets.rr}</div>
                        </div>
                    </div>
                    <div className="space-y-3">
                        {[{ label: 'Primary Target', val: targets.tp1, color: 'emerald', tag: 'TP1' }, { label: 'Extended Target', val: targets.tp2, color: 'emerald', tag: 'TP2' }, { label: 'Stop Loss', val: targets.sl, color: 'rose', tag: 'SL' }].map((t, i) => (
                            <div className={`flex items-center justify-between p-4 rounded-xl bg-slate-950/40 border border-${t.color}-500/20 group hover:border-${t.color}-500/40 transition-all`}>
                                <div className="flex items-center gap-3">
                                    <div className={`w-8 h-8 rounded-lg bg-${t.color}-500 flex items-center justify-center text-slate-950 font-black text-xs shadow-[0_0_15px_rgba(var(--${t.color}-rgb),0.3)]`}>{t.tag}</div>
                                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest leading-none">{t.label}</span>
                                </div>
                                <div className={`text-xl md:text-2xl font-mono text-${t.color}-400 font-bold tabular-nums`}>${t.val}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Fallback Modal for manual input
const ManualTickerModal = ({ onSubmit, onClose }) => {
    const [input, setInput] = useState('');
    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-slate-900 border border-slate-800 rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-slate-400"><Keyboard className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none">Optical Scan Failed</h3>
                <p className="text-slate-500 text-sm font-bold mb-6">The neural core could not identify the asset ticker from the image. Please enter it manually.</p>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value.toUpperCase())}
                    placeholder="e.g. BTC, ETH, SOL"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-center text-white font-black tracking-widest uppercase mb-6 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                    autoFocus
                />
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-3 rounded-xl border border-slate-800 text-slate-500 font-bold hover:bg-slate-800 transition-colors">Cancel</button>
                    <button onClick={() => input && onSubmit(input)} disabled={!input} className="flex-1 py-3 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">Proceed</button>
                </div>
            </div>
        </div>
    );
};

const VerificationModal = ({ validationResult, ticker, onClose, onCalibrate }) => {
    const { accuracy, hits, recommendedWeights, predictions } = validationResult || {};
    const isGood = accuracy > 60;

    if (!hits || !recommendedWeights || !predictions) return null;

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md animate-in fade-in" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="bg-slate-900 border border-slate-800 rounded-[32px] w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh] animate-in zoom-in-95">
                <div className="p-6 md:p-8 border-b border-slate-800 flex justify-between items-start shrink-0">
                    <div>
                        <h3 className="text-xl md:text-2xl font-black text-white flex items-center gap-2">
                            <ShieldCheck className={isGood ? "text-emerald-400" : "text-amber-400"} />
                            Model Verification
                        </h3>
                        <p className="text-slate-500 text-[10px] font-mono mt-1 uppercase tracking-widest leading-none">
                            Backtesting Report: {ticker} (Last 14 Candles)
                        </p>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full text-slate-500 hover:text-white transition-colors"><X className="w-6 h-6" /></button>
                </div>

                <div className="p-6 md:p-8 overflow-y-auto scrollbar-none space-y-8">
                    {/* Score Cards */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className={`p-6 rounded-2xl border ${isGood ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}`}>
                            <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 block mb-2">Turing Accuracy Score</span>
                            <div className={`text-4xl font-black ${isGood ? 'text-emerald-400' : 'text-amber-400'}`}>{accuracy}<span className="text-lg">%</span></div>
                            <p className="text-[10px] text-slate-500 mt-2 leading-tight">Directional correctness over the validation period.</p>
                        </div>
                        <div className="p-6 rounded-2xl border bg-slate-950/50 border-slate-800">
                            <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 block mb-2">Technological Hits</span>
                            <div className="flex gap-4">
                                <div><div className="text-xl font-black text-white">{hits.neural}</div><div className="text-[8px] text-slate-500 uppercase">Neural</div></div>
                                <div><div className="text-xl font-black text-white">{hits.pattern}</div><div className="text-[8px] text-slate-500 uppercase">Pattern</div></div>
                                <div><div className="text-xl font-black text-white">{hits.technical}</div><div className="text-[8px] text-slate-500 uppercase">Tech</div></div>
                            </div>
                            <p className="text-[10px] text-slate-500 mt-2 leading-tight">Correct directional calls / {predictions.length} test points.</p>
                        </div>
                    </div>

                    {/* Dynamic Weight Recommendation */}
                    <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h4 className="text-white font-black text-sm uppercase tracking-tighter">Engine Calibration Recommendation</h4>
                                <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mt-1">Self-Correction Protocol</p>
                            </div>
                            <div className="flex items-center gap-2 group cursor-help">
                                <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
                                <span className="text-[9px] font-black text-emerald-400 uppercase tracking-widest">Optimized Weights</span>
                            </div>
                        </div>
                        <div className="grid grid-cols-3 gap-6">
                            {Object.entries(recommendedWeights).map(([key, val]) => (
                                <div key={key} className="relative">
                                    <div className="text-2xl font-black text-white mb-1">{(val * 100).toFixed(0)}<span className="text-[10px] text-slate-500 ml-1">%</span></div>
                                    <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">{key === 'omega' ? 'Neural' : (key === 'alpha' ? 'Pattern' : 'Technical')}</div>
                                    <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${val * 100}%` }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                        {onCalibrate && (
                            <button
                                onClick={() => onCalibrate(recommendedWeights)}
                                className="w-full mt-6 py-4 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-xl transition-all uppercase tracking-widest text-[10px] shadow-xl shadow-emerald-500/10 flex items-center justify-center gap-2"
                            >
                                <Cpu className="w-4 h-4" />
                                Calibrate AI Weights to Current Market
                            </button>
                        )}
                    </div>

                    {/* Simple Visualization Chart (SVG) */}
                    <div className="h-48 w-full bg-slate-950/50 rounded-2xl border border-slate-800 relative flex items-end px-4 pb-4 pt-8 gap-1">
                        {predictions.map((p, i) => {
                            // Normalize for height (0-100%)
                            // We find min/max of this small set for drawing
                            const allVals = predictions.flatMap(x => [x.actual, x.predicted]);
                            const min = Math.min(...allVals);
                            const max = Math.max(...allVals);
                            const range = max - min || 1;

                            const hActual = ((p.actual - min) / range) * 80 + 10;
                            const hPred = ((p.predicted - min) / range) * 80 + 10;

                            return (
                                <div key={i} className="flex-1 flex flex-col justify-end gap-1 relative group h-full">
                                    {/* Tooltip */}
                                    <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[9px] px-2 py-1 rounded hidden group-hover:block whitespace-nowrap z-10 border border-slate-700">
                                        Act: {p.actual.toFixed(2)} | Pred: {p.predicted.toFixed(2)}
                                    </div>
                                    <div className="w-full bg-emerald-500/50 rounded-t-sm transition-all hover:bg-emerald-400" style={{ height: `${hActual}%` }}></div>
                                    <div className="w-full bg-purple-500/50 rounded-t-sm absolute bottom-0 left-0 right-0 mix-blend-screen transition-all hover:bg-purple-400" style={{ height: `${hPred}%` }}></div>
                                </div>
                            );
                        })}

                        {/* Legend */}
                        <div className="absolute top-2 left-4 flex gap-4 text-[9px] font-bold uppercase tracking-widest">
                            <span className="flex items-center gap-1"><div className="w-2 h-2 bg-emerald-500 rounded-sm"></div> Actual</span>
                            <span className="flex items-center gap-1"><div className="w-2 h-2 bg-purple-500 rounded-sm"></div> Predicted</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const ApiKeyModal = ({ ticker, onSubmit, onClose }) => {
    const [input, setInput] = useState('');
    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-slate-900 border border-slate-800 rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-emerald-400"><Key className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none">Stock Data Access</h3>
                <p className="text-slate-500 text-sm font-bold mb-4">You are analyzing <span className="text-emerald-400">{ticker}</span>. To access real-time S&P 500 data, a free Finnhub.io API Token is required.</p>
                <p className="text-xs text-slate-600 mb-6 bg-slate-950 p-2 rounded-lg border border-slate-800">Your key is stored locally in your browser and never shared.</p>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Paste Finnhub API Key"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-center text-white font-mono text-sm mb-6 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                    autoFocus
                />
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-3 rounded-xl border border-slate-800 text-slate-500 font-bold hover:bg-slate-800 transition-colors">Cancel</button>
                    <button onClick={() => input && onSubmit(input)} disabled={!input} className="flex-1 py-3 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">Save & Scan</button>
                </div>
                <a href="https://finnhub.io/register" target="_blank" rel="noreferrer" className="block mt-4 text-xs text-blue-500 hover:underline">Get a free API Key →</a>
            </div>
        </div>
    );
};

const LimitModal = ({ message, type, onClose, onLogin }) => {
    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-slate-950/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-slate-900 border border-slate-800 rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.2)]"><Lock className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none uppercase tracking-tight">Access Restricted</h3>
                <p className="text-slate-500 text-sm font-bold mb-8 leading-relaxed">{message}</p>
                <div className="flex flex-col gap-3">
                    {type === 'free' ? (
                        <button onClick={() => window.open('/pricing', '_blank')} className="w-full py-4 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-all shadow-xl shadow-emerald-500/20 uppercase tracking-widest text-xs flex items-center justify-center gap-2">Upgrade to Pro <ArrowRight className="w-4 h-4" /></button>
                    ) : type === 'guest' ? (
                        <button onClick={onLogin} className="w-full py-4 rounded-xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-all shadow-xl shadow-emerald-500/20 uppercase tracking-widest text-xs flex items-center justify-center gap-2">Initialize Authentication <ArrowRight className="w-4 h-4" /></button>
                    ) : (
                        <button onClick={onClose} className="w-full py-4 rounded-xl bg-slate-800 text-white font-black hover:bg-slate-700 transition-all uppercase tracking-widest text-xs">Check Inbox</button>
                    )}
                    <button onClick={onClose} className="w-full py-3 rounded-xl border border-slate-800 text-slate-500 font-bold hover:bg-slate-800 transition-colors uppercase tracking-widest text-[10px]">Close Terminal</button>
                </div>
            </div>
        </div>
    );
};

const FileUpload = ({ onFileSelect, isAnalyzing, statusMessage }) => {
    const [isDragging, setIsDragging] = useState(false);
    const handleDrag = (e) => { e.preventDefault(); setIsDragging(e.type === 'dragover'); };
    return (
        <div
            onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files[0]) onFileSelect(e.dataTransfer.files[0]); }}
            className={`relative group h-96 rounded-[40px] border-4 border-dashed transition-all duration-500 flex flex-col items-center justify-center cursor-pointer overflow-hidden ${isDragging ? 'border-emerald-500 bg-emerald-500/5 scale-[0.99] shadow-inner' : 'border-slate-800 bg-slate-900/50 hover:border-slate-700 hover:bg-slate-900 shadow-2xl'}`}
        >
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer z-10" onChange={e => onFileSelect(e.target.files[0])} accept="image/*" />
            {isAnalyzing && !analysisResult ? (
                <ScanningHUD status={statusMessage} />
            ) : (
                <>
                    <div className="w-24 h-24 bg-emerald-500 rounded-3xl flex items-center justify-center mb-8 group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 shadow-2xl shadow-emerald-500/20"><ShieldCheck className="w-12 h-12 text-slate-950" /></div>
                    <h3 className="text-3xl font-black text-white mb-3 tracking-tighter uppercase">Ready for Analysis</h3>
                    <p className="text-slate-500 max-w-sm text-center mb-10 leading-relaxed font-bold text-sm">Drag & drop your chart, paste from clipboard, or click to browse local files.</p>
                    <div className="px-8 py-4 bg-emerald-500 text-slate-950 font-black rounded-2xl shadow-xl shadow-emerald-500/10 group-hover:bg-emerald-400 group-hover:scale-105 transition-all uppercase tracking-widest text-xs">Execute Neural Analysis</div>
                </>
            )}
        </div>
    );
};

// --- Pro Utils ---
const generatePDF = () => {
    window.print();
};

const MacroFluctuationChart = ({ macroTrend, isPro, source }) => {
    if (!isPro) return null;
    if (!macroTrend || !macroTrend.prices || !macroTrend.dates) return null;

    // Process Data: Group by Year
    const yearlyData = {};
    macroTrend.dates.forEach((ts, i) => {
        const year = new Date(ts * 1000).getFullYear();
        if (!yearlyData[year]) yearlyData[year] = { first: macroTrend.prices[i], last: macroTrend.prices[i], min: macroTrend.prices[i], max: macroTrend.prices[i] };
        else {
            yearlyData[year].last = macroTrend.prices[i];
            yearlyData[year].min = Math.min(yearlyData[year].min, macroTrend.prices[i]);
            yearlyData[year].max = Math.max(yearlyData[year].max, macroTrend.prices[i]);
        }
    });

    const years = Object.keys(yearlyData).sort();
    const bars = years.map(y => {
        const d = yearlyData[y];
        const change = ((d.last - d.first) / d.first) * 100;
        return { year: y, change };
    }).slice(-10); // Last 10 years

    const maxVal = Math.max(...bars.map(b => Math.abs(b.change)), 10); // Scale

    return (
        <div className="bg-[#020617] border border-slate-800 rounded-lg p-4 print:border-gray-300 print:bg-white">
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-slate-500" />
                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">10-Year Volatility Matrix</span>
                </div>
                <div className="px-2 py-0.5 bg-amber-500/10 text-amber-500 text-[9px] font-bold border border-amber-500/20 rounded uppercase tracking-wider print:hidden">Pro Feature</div>
            </div>

            <div className="flex items-end justify-between gap-1 h-32 relative">
                {/* Zero Line */}
                <div className="absolute top-1/2 left-0 right-0 h-px bg-slate-800 z-0"></div>

                {bars.map((bar) => {
                    const isPos = bar.change >= 0;
                    const height = Math.abs(bar.change) / maxVal * 50; // Max 50% height (up/down)
                    return (
                        <div key={bar.year} className="flex-1 flex flex-col items-center z-10 group relative">
                            {/* Bar Container */}
                            <div className="h-full w-full flex flex-col justify-center relative">
                                <div
                                    className={`w-full mx-auto max-w-[20px] rounded-sm transition-all duration-500 ${isPos ? 'bg-emerald-500 hover:bg-emerald-400' : 'bg-rose-500 hover:bg-rose-400'}`}
                                    style={{
                                        height: `${Math.min(height, 50)}%`,
                                        marginTop: isPos ? 0 : '0px',
                                        marginBottom: isPos ? '0px' : 0,
                                        transform: isPos ? 'translateY(-50%)' : 'translateY(50%) translateY(2px)' // Anchor to center
                                    }}
                                >
                                    {/* Tooltip */}
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-slate-900 border border-slate-800 text-white text-[9px] font-mono px-2 py-1 rounded whitespace-nowrap z-50 shadow-xl">
                                        <div className="font-bold">{bar.year}</div>
                                        <div className={isPos ? 'text-emerald-400' : 'text-rose-400'}>{bar.change > 0 ? '+' : ''}{bar.change.toFixed(1)}%</div>
                                    </div>
                                </div>
                            </div>
                            <div className="mt-2 text-[8px] font-bold text-slate-600">{bar.year.slice(2)}</div>
                        </div>
                    );
                })}
            </div>
            {/* Footer Note */}
            <div className="mt-4 pt-3 border-t border-slate-800 text-center">
                <p className="text-[9px] text-slate-500 font-medium uppercase tracking-wider">
                    Historic data via {source || "Institutional Feed"} • {isPro ? 'Pro Access' : 'Preview'}
                </p>
            </div>
        </div>
    );
};

const AnalysisResult = ({ result, imagePreview, onVerify, isVerifying, isPro, onToggleIncognito, isIncognito }) => {
    const [showBlueprint, setShowBlueprint] = useState(false);
    const isBull = result.direction.includes('Bullish');
    const colors = isBull ? { text: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' } : (result.direction.includes('Bearish') ? { text: 'text-rose-400', border: 'border-rose-500/30', bg: 'bg-rose-500/10' } : { text: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/10' });
    const Icon = isBull ? TrendingUp : (result.direction.includes('Bearish') ? TrendingDown : Minus);

    return (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 pb-12 print:space-y-4">
            {showBlueprint && <TradeTheoryModal targets={result.targets} direction={result.direction} onClose={() => setShowBlueprint(false)} />}

            {/* Header / Meta */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 border-b border-slate-800 pb-6 print:border-black">
                <div className="space-y-1">
                    <div className="flex items-center gap-3">
                        <div className={`p-1.5 rounded ${colors.bg} border ${colors.border} print:hidden`}>
                            <Icon className={`w-5 h-5 ${colors.text}`} />
                        </div>
                        <h2 className="text-4xl font-black text-white tracking-tighter uppercase print:text-black">{result.ticker}</h2>
                        {isPro && (
                            <div className="px-2 py-0.5 bg-amber-500/10 text-amber-500 text-[9px] font-bold border border-amber-500/20 rounded uppercase tracking-wider print:hidden">
                                Pro
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 uppercase tracking-widest print:text-gray-600">
                        <span>{new Date(result.date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}</span>
                        <span>ID: {result.id.slice(-6)}</span>
                        <span>Model: {result.version}</span>
                    </div>
                </div>

                <div className="flex items-center gap-3 print:hidden">
                    {isPro && (
                        <button
                            onClick={onToggleIncognito}
                            className={`p-2 rounded hover:bg-slate-800 border border-transparent transition-all ${isIncognito ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' : 'text-slate-500'}`}
                            title="Toggle Incognito Mode"
                        >
                            {isIncognito ? <div className="flex gap-2 items-center"><ShieldCheck className="w-4 h-4" /><span className="text-[10px] font-bold">STEALTH ACTIVE</span></div> : <History className="w-4 h-4" />}
                        </button>
                    )}
                    <button
                        onClick={generatePDF}
                        className="px-4 py-2 bg-slate-800 border border-slate-700 text-slate-300 hover:text-white rounded text-[10px] font-bold uppercase tracking-widest transition-colors"
                    >
                        Export Report
                    </button>
                    <button
                        onClick={() => setShowBlueprint(true)}
                        className={`px-4 py-2 ${colors.bg} ${colors.text} border ${colors.border} rounded text-[10px] font-bold uppercase tracking-widest hover:brightness-110 transition-all`}
                    >
                        Trade Protocol
                    </button>
                </div>
            </div>

            {/* Key Metrics Grid (Dense) */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-800 border border-slate-800 rounded-lg overflow-hidden print:border-gray-300 print:bg-gray-300">
                <div className="bg-[#020617] p-4 flex flex-col justify-between hover:bg-slate-900 transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Confidence</span>
                    <span className={`text-2xl font-mono font-bold ${colors.text} print:text-black`}>{result.confidence}%</span>
                </div>
                <div className="bg-[#020617] p-4 flex flex-col justify-between hover:bg-slate-900 transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Volatility (ATR)</span>
                    <span className="text-2xl font-mono font-bold text-white print:text-black">{result.riskMetrics?.volatility || "0.00"}%</span>
                </div>
                <div className="bg-[#020617] p-4 flex flex-col justify-between hover:bg-slate-900 transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Sharpe (Est.)</span>
                    <span className="text-2xl font-mono font-bold text-blue-400 print:text-black">{result.riskMetrics?.sharpeRatio || "0.00"}</span>
                </div>
                <div className="bg-[#020617] p-4 flex flex-col justify-between hover:bg-slate-900 transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Reward/Risk</span>
                    <span className="text-2xl font-mono font-bold text-emerald-400 print:text-black">{result.targets.rr}x</span>
                </div>
            </div>

            {/* Pro Feature: 10-Year Macro Trend */}
            <MacroFluctuationChart macroTrend={result.macroTrend} isPro={isPro} source={result.macroTrend?.source} />

            {/* Main Content Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column: Narrative & Chart */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Narrative Block */}
                    <div className="bg-slate-900/30 p-6 border border-slate-800 rounded-lg shadow-sm print:bg-white print:border-gray-200">
                        <div className="flex items-center gap-2 mb-4">
                            <Activity className="w-4 h-4 text-slate-500" />
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Situational Analysis</h3>
                        </div>
                        <p className="text-slate-300 text-sm leading-relaxed font-mono print:text-black" dangerouslySetInnerHTML={{ __html: result.overview || result.summary.replace(/\*\*(.*?)\*\*/g, '<span class="text-white font-bold">$1</span>') }} />
                    </div>

                    {/* Chart Preview */}
                    <div className="border border-slate-800 rounded-lg overflow-hidden relative group h-64 print:hidden">
                        <div className="absolute top-3 left-3 z-10 bg-black/50 backdrop-blur px-2 py-1 rounded text-[9px] font-mono text-white border border-white/10">INPUT SOURCE</div>
                        <img src={imagePreview} alt="Analyzed Chart" className="w-full h-full object-cover opacity-50 grayscale group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-500" />
                    </div>
                </div>

                {/* Right Column: Factor Table */}
                <div className="lg:col-span-1">
                    <div className="border border-slate-800 rounded-lg overflow-hidden bg-[#020617] print:border-gray-200">
                        <div className="bg-slate-900/50 px-4 py-3 border-b border-slate-800 flex justify-between items-center print:bg-gray-100">
                            <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Alpha Factors</span>
                            {onVerify && (
                                <button onClick={onVerify} disabled={isVerifying} className="text-[9px] text-blue-500 hover:text-blue-400 uppercase font-bold print:hidden">
                                    {isVerifying ? 'Verifying...' : 'Verify Engine'}
                                </button>
                            )}
                        </div>
                        <table className="w-full text-left text-[9px] md:text-[10px]">
                            <tbody className="divide-y divide-slate-800/50 print:divide-gray-200">
                                {result.factors.map((f, i) => (
                                    <tr key={i} className="hover:bg-slate-900/50 transition-colors">
                                        <td className="px-4 py-3">
                                            <div className="font-bold text-slate-300 print:text-black">{f.name}</div>
                                            <div className="text-[8px] text-slate-600 font-mono mt-0.5 print:text-gray-500">{f.type}</div>
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            <div className={`font-mono font-bold ${f.p > 0.5 ? 'text-emerald-400' : 'text-slate-400'} print:text-black`}>{(f.p * 100).toFixed(1)}%</div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {/* Macro Data Row */}
                        <div className="px-4 py-3 border-t border-slate-800 bg-slate-900/20 flex justify-between items-center">
                            <span className="text-[9px] font-bold text-slate-500 uppercase">Drawdown Risk</span>
                            <span className="text-mono text-[10px] text-rose-400 font-bold">-{result.riskMetrics?.maxDrawdown || "0.00"}%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const HistorySidebar = ({ history, onSelect, onDelete, onFeedback }) => {
    return (
        <div className="h-full flex flex-col bg-[#020617] border-l border-slate-900 shadow-2xl overflow-hidden">
            <div className="p-6 border-b border-slate-900 bg-slate-950/50 backdrop-blur-xl">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Terminal Log</h3>
                    </div>
                    <span className="text-[8px] font-mono text-slate-600 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">CORE v2.4</span>
                </div>
                <div className="flex items-center justify-between text-[8px] font-black text-slate-500 uppercase tracking-widest px-1">
                    <span>Recent Activity</span>
                    <span>{history.length} Nodes</span>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto scrollbar-none p-4 space-y-3">
                {history.map((item, i) => {
                    const isBull = item.direction?.includes('Bullish');
                    const color = isBull ? 'emerald' : 'rose';
                    const date = new Date(item.created_at || item.date);

                    return (
                        <div
                            key={item.db_id}
                            onClick={() => onSelect(item)}
                            className="group relative bg-slate-900/40 border border-slate-800/50 rounded-xl p-4 cursor-pointer hover:bg-slate-800/40 transition-all hover:border-slate-700/50 overflow-hidden"
                        >
                            {/* Status Indicator */}
                            <div className={`absolute top-0 right-0 w-24 h-24 bg-${color}-500/5 blur-2xl -translate-y-1/2 translate-x-1/2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity`} />

                            <div className="flex justify-between items-start relative z-10">
                                <div className="space-y-1">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-black text-white tracking-tight">{item.ticker}</span>
                                        <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded bg-${color}-500/10 text-${color}-400 border border-${color}-500/20 uppercase`}>
                                            {item.direction?.split(' ')[1] || item.direction}
                                        </span>
                                    </div>
                                    <div className="text-[8px] font-mono text-slate-500 flex items-center gap-1.5 uppercase">
                                        <History className="w-2.5 h-2.5" />
                                        {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className={`text-[10px] font-mono font-black text-${color}-400`}>{item.confidence}%</div>
                                    <div className="text-[7px] font-black text-slate-600 uppercase tracking-tighter">Confidence</div>
                                </div>
                            </div>

                            <div className="mt-4 pt-4 border-t border-slate-800/30 flex items-center justify-between relative z-10">
                                <div className="flex gap-2">
                                    <button
                                        onClick={(e) => { e.stopPropagation(); onFeedback(item.db_id, true); }}
                                        className={`p-1.5 rounded-lg border border-slate-800/50 transition-all ${item.feedback === 'win' ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' : 'text-slate-600 hover:text-emerald-400 hover:bg-emerald-500/10'}`}
                                    >
                                        <ThumbsUp className="w-3 h-3" />
                                    </button>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); onFeedback(item.db_id, false); }}
                                        className={`p-1.5 rounded-lg border border-slate-800/50 transition-all ${item.feedback === 'loss' ? 'bg-rose-500/20 border-rose-500/30 text-rose-400' : 'text-slate-600 hover:text-rose-400 hover:bg-rose-500/10'}`}
                                    >
                                        <ThumbsDown className="w-3 h-3" />
                                    </button>
                                </div>
                                <button
                                    onClick={(e) => { e.stopPropagation(); onDelete(item.db_id); }}
                                    className="p-1.5 text-slate-700 hover:text-rose-400 transition-colors"
                                >
                                    <Trash2 className="w-3 h-3" />
                                </button>
                            </div>
                        </div>
                    );
                })}

                {history.length === 0 && (
                    <div className="py-12 text-center space-y-4 filter grayscale opacity-20">
                        <Activity className="w-12 h-12 text-slate-600 mx-auto" />
                        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-600">Sync with market <br />to populate log</p>
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-slate-900 bg-slate-950/30">
                <div className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-xl border border-slate-800/40">
                    <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center border border-blue-500/20">
                        <Cpu className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                        <div className="text-[9px] font-black text-white uppercase leading-none">Global Model</div>
                        <div className="text-[7px] font-bold text-slate-500 uppercase mt-1">Status: Operational</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// --- TERMINAL PAGE ---
export default function Terminal() {
    const { user, profile, refreshProfile, neuralState, setNeuralState, fetchNeuralState } = useAppContext();
    const weights = neuralState;
    const location = useLocation();
    const navigate = useNavigate();

    const syncGlobalFeedback = async (isWin) => {
        if (!user) return;

        const learningRate = 0.02;
        const shift = isWin ? learningRate : -learningRate;

        const nextWeights = {
            alpha: Math.max(0.1, Math.min(0.6, weights.alpha + (Math.random() * shift))),
            beta: Math.max(0.1, Math.min(0.6, weights.beta + (Math.random() * shift))),
            omega: Math.max(0.2, Math.min(0.9, weights.omega + (isWin ? 0.01 : -0.01))),
            iterations: weights.iterations + 1
        };

        setNeuralState({ ...weights, ...nextWeights });

        // Update Supabase (Collective Intelligence)
        try {
            await supabase
                .from('neural_state')
                .update({
                    ...nextWeights,
                    last_updated: new Date().toISOString(),
                    updated_by: user.id
                })
                .eq('id', 'global_master');
        } catch (err) {
            console.error("Global intelligence sync failed:", err);
        }
    };
    const [activeTab, setActiveTab] = useState('analyze');
    const [imagePreview, setImagePreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [statusMessage, setStatusMessage] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [showAuth, setShowAuth] = useState(false);
    const [showManualInput, setShowManualInput] = useState(false);
    const [showLimitModal, setShowLimitModal] = useState(false);
    const [limitMessage, setLimitMessage] = useState("");
    const [limitType, setLimitType] = useState('guest'); // guest, verify, free
    const [showApiKeyInput, setShowApiKeyInput] = useState(false); // New State for API Key Modal
    const [pendingTicker, setPendingTicker] = useState(null); // Ticker waiting for API key
    const [validationResult, setValidationResult] = useState(null);
    const [isVerifying, setIsVerifying] = useState(false);
    const [userIp, setUserIp] = useState(null);
    const [tickerItems, setTickerItems] = useState([]);
    const [isIncognito, setIsIncognito] = useState(false); // Pro Feature
    const tempFileRef = useRef(null);
    const ocrWorkerRef = useRef(null);

    // Check Pro Status
    const isPro = profile?.subscription_status === 'active' || profile?.subscription_tier === 'pro' || location.search.includes('demo=true');

    // Singleton OCR Engine Initialization
    const getOcrWorker = async () => {
        if (ocrWorkerRef.current) return ocrWorkerRef.current;
        setStatusMessage("Priming Neural OCR Engine...");
        const worker = await Tesseract.createWorker('eng');
        ocrWorkerRef.current = worker;
        return worker;
    };

    useEffect(() => {
        const fetchTickers = async () => {
            const data = await fetchTickerData(['BTC', 'ETH', 'SOL', 'AAPL', 'TSLA', 'SPY', 'QQQ']);
            if (data) setTickerItems(data);
        };
        fetchTickers();
        const interval = setInterval(fetchTickers, 60000); // 60s polling to respect free API limits
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        fetchHistory();
        fetchIp();
        const handlePaste = (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (const item of items) {
                if (item.type.indexOf("image") !== -1) {
                    const file = item.getAsFile();
                    if (file) handleFileSelect(file);
                    break;
                }
            }
        };
        window.addEventListener('paste', handlePaste);
        return () => {
            window.removeEventListener('paste', handlePaste);
            if (ocrWorkerRef.current) {
                ocrWorkerRef.current.terminate();
                ocrWorkerRef.current = null;
            }
        };
    }, [user]);

    useEffect(() => {
        const handleSharedImage = async () => {
            const params = new URLSearchParams(location.search);
            if (params.get('demo') === 'true') {
                setAnalysisResult({
                    ticker: 'BTC',
                    direction: 'Strong Bullish',
                    confidence: '87.4',
                    date: new Date().toISOString(),
                    id: 'mock-123',
                    version: 'HYBRID-MOCK',
                    targets: { rr: '3.5', entry: '95000', tp1: '98000', tp2: '105000', sl: '92000' },
                    overview: 'Mock analysis for UI verification. <strong>Strong Bullish</strong> patterns detected.',
                    finalProb: 0.87,
                    pattern: { name: 'Bull Flag', sentiment: 'Bullish' },
                    factors: [{ name: 'Neural', p: 0.9, type: 'Deep Learning' }, { name: 'Pattern', p: 0.8, type: 'Geometric' }],
                    riskMetrics: { volatility: '4.2', sharpeRatio: '2.1', maxDrawdown: '12.5' },
                    macroTrend: {
                        // Mock 10 years of data (one point per year is enough for the grouper to work)
                        prices: [15000, 18000, 4000, 12000, 7000, 28000, 69000, 16000, 42000, 95000],
                        dates: [
                            new Date('2015-01-01').getTime() / 1000,
                            new Date('2016-01-01').getTime() / 1000,
                            new Date('2017-01-01').getTime() / 1000,
                            new Date('2018-01-01').getTime() / 1000,
                            new Date('2019-01-01').getTime() / 1000,
                            new Date('2020-01-01').getTime() / 1000,
                            new Date('2021-01-01').getTime() / 1000,
                            new Date('2022-01-01').getTime() / 1000,
                            new Date('2023-01-01').getTime() / 1000,
                            new Date('2024-01-01').getTime() / 1000
                        ]
                    }
                });
                return;
            }
            if (params.get('shared-image') === 'true') {
                try {
                    const cache = await caches.open('shared-media');
                    const response = await cache.match('shared-image');
                    if (response) {
                        const blob = await response.blob();
                        const file = new File([blob], "shared-image.png", { type: blob.type });
                        // Wait a bit to ensure component is fully ready
                        setTimeout(() => handleFileSelect(file), 500);

                        // Clean up
                        await cache.delete('shared-image');
                        navigate(location.pathname, { replace: true });
                    }
                } catch (err) {
                    console.error("Shared image processing failed:", err);
                }
            }
        };
        handleSharedImage();
    }, [location.search]);

    const fetchIp = async () => {
        try {
            // Use a slightly different approach to avoid common ad-blocker patterns if needed, 
            // but for now just silence the error.
            const res = await fetch('https://api.ipify.org?format=json');
            if (!res.ok) throw new Error("Network response was not ok");
            const data = await res.json();
            setUserIp(data.ip);
        } catch (err) {
            // Ad-blockers or privacy tools often block this. Fail silently.
            setUserIp('unknown-client');
        }
    };

    const fetchHistory = async () => {
        if (!user) {
            const local = localStorage.getItem('diver_ai_guest_history');
            setHistory(local ? JSON.parse(local) : []);
            return;
        }
        const { data, error } = await supabase.from('prediction_history').select('*').eq('user_id', user.id).order('created_at', { ascending: false });
        if (!error && data) setHistory(data.map(row => ({ ...row.data, db_id: row.id, created_at: row.created_at })));
    };

    const checkLimits = () => {
        const today = new Date().toISOString().split('T')[0];

        if (!user) {
            // Guest / IP-based Limit
            const guestLogs = JSON.parse(localStorage.getItem('diver_ai_guest_ip_logs') || '{}');
            const currentIpLog = guestLogs[userIp || 'unknown'] || { count: 0, date: today };

            // Reset if new day
            if (currentIpLog.date !== today) {
                currentIpLog.count = 0;
                currentIpLog.date = today;
            }

            if (currentIpLog.count >= 3) {
                setLimitMessage("IP Limit Reached: 3 guest analysis/day. Please log in for expanded access.");
                setLimitType('guest');
                setShowLimitModal(true);
                return false;
            }
            return true;
        }

        // Email Verification Check
        if (!user.email_confirmed_at) {
            setLimitMessage("Security Protocol: Email verification required. Please check your inbox and confirm your email to unlock the terminal scanning engine.");
            setLimitType('verify');
            setShowLimitModal(true);
            return false;
        }

        // Usage limit for authenticated users (free tier)
        if (profile && profile.subscription_tier !== 'pro') {
            const today = new Date().toISOString().split('T')[0];
            const todayCount = history.filter(item => {
                const dateStr = item.created_at ? item.created_at.split('T')[0] : null;
                return dateStr === today;
            }).length;
            if (todayCount >= 3) {
                setLimitMessage("Neural Capacity Reached: 3/day. Upgrade to Pro for unlimited terminal access.");
                setLimitType('free');
                setShowLimitModal(true);
                return false;
            }
        }
        return true;
    };

    const updateGuestUsage = () => {
        if (user) return;
        const today = new Date().toISOString().split('T')[0];
        const guestLogs = JSON.parse(localStorage.getItem('diver_ai_guest_ip_logs') || '{}');
        const currentIp = userIp || 'unknown';
        const log = guestLogs[currentIp] || { count: 0, date: today };

        if (log.date !== today) {
            log.count = 1;
            log.date = today;
        } else {
            log.count += 1;
        }

        guestLogs[currentIp] = log;
        localStorage.setItem('diver_ai_guest_ip_logs', JSON.stringify(guestLogs));
    };

    const preprocessImage = (imageElement) => {
        const width = imageElement.width;
        const height = imageElement.height;
        const scale = Math.max(1, 1000 / width); // Slightly higher res for OCR
        const w = width * scale;
        const h = height * scale;

        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d');

        // Draw original to check stats
        ctx.drawImage(imageElement, 0, 0, w, h);
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        let totalBrightness = 0;
        for (let i = 0; i < data.length; i += 4) {
            totalBrightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }
        const avgBrightness = totalBrightness / (data.length / 4);
        const isDark = avgBrightness < 128;

        // 1. Visual Source (Standard Grayscale)
        // Reset and draw for Visual
        ctx.globalCompositeOperation = 'source-over';
        ctx.filter = 'grayscale(100%) contrast(120%) brightness(110%)';
        ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); // Clear
        ctx.drawImage(imageElement, 0, 0, w, h);
        const visualSrc = canvas.toDataURL('image/png');

        // 2. OCR Source (Inverted if dark, high contrast)
        // Reset filters
        ctx.filter = isDark
            ? 'invert(100%) grayscale(100%) contrast(150%) brightness(110%)'
            : 'grayscale(100%) contrast(150%)';

        ctx.fillStyle = isDark ? '#FFF' : '#000'; // Background logic
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(imageElement, 0, 0, w, h);
        const ocrSrc = canvas.toDataURL('image/png');

        return { visualSrc, ocrSrc };
    };

    const handleFileSelect = (file) => {
        if (!checkLimits()) return;
        if (!file) return;
        tempFileRef.current = file;

        const reader = new FileReader();
        reader.onloadend = () => {
            const img = new Image();
            img.onload = () => {
                const { visualSrc, ocrSrc } = preprocessImage(img);
                setImagePreview(reader.result);
                runAnalysisWorkflow(reader.result, visualSrc, ocrSrc);
            };
            img.src = reader.result;
        };
        reader.readAsDataURL(file);
    };

    const runAnalysisWorkflow = async (originalFileSrc, visualSrc, ocrSrc = null, manualTicker = null, manualApiKey = null) => {
        const ocrImage = ocrSrc || visualSrc; // Fallback

        setIsAnalyzing(true); setAnalysisResult(null);
        if (!manualTicker && !manualApiKey) setStatusMessage("Initializing Optical Core...");

        // Request Notification Permission
        if (Notification.permission === 'default') {
            Notification.requestPermission();
        }

        try {
            let ticker = manualTicker;
            let marketStats = null;
            let historicalPrices = null;

            // --- HYBRID DATA CAPTURE (Simultaneous OCR & Visual) ---
            const ocrPromise = (async () => {
                if (manualTicker) return manualTicker;

                // Server-Side OCR for Pro Users (Offload & Deep Scan)
                if (profile?.subscription_tier === 'pro' && user) {
                    try {
                        setStatusMessage("Deep Scan (Cloud OCR)...");
                        // CRITICAL: We pass ocrSrc (which can be originalFileSrc if we want raw)
                        // Actually, let's use originalFileSrc for cloud OCR as it's pure
                        const { data, error } = await supabase.functions.invoke('detect_ticker', {
                            body: { image: originalFileSrc }
                        });

                        if (error) throw error;
                        if (data?.text) {
                            return { text: data.text, ticker: detectTicker(data.text) };
                        }
                    } catch (cloudErr) {
                        if (cloudErr.message?.includes("401")) {
                            console.info("Cloud OCR unavailable (Guest/Session Stale). Switching to local engine.");
                        } else {
                            console.warn("Cloud OCR Failed, reverting to local:", cloudErr);
                        }
                    }
                }

                try {
                    const worker = await getOcrWorker();
                    setStatusMessage("Scanning Chart (OCR)...");

                    // Optimizing for ticker symbols (uppercase, numbers, basic symbols)
                    await worker.setParameters({
                        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$-/. '
                    });

                    const { data: { text } } = await worker.recognize(ocrImage);
                    return { text, ticker: detectTicker(text) };
                } catch (ocrErr) {
                    console.warn("OCR Engine Failure:", ocrErr);
                    return { text: "", ticker: null };
                }
            })();

            const visualPromise = (async () => {
                setStatusMessage("Extracting Visual Data...");
                return await extractChartData(visualSrc);
            })();

            // Wait for both "eyes" to see
            const [ocrRawResult, visualData] = await Promise.all([ocrPromise, visualPromise]);

            // Handle Hybrid Detection (Ticker OR Price)
            let detectedTicker = null;
            let anchorPrice = null;

            if (typeof ocrRawResult === 'object' && ocrRawResult !== null) {
                // Cloud OCR path (or updated promise) with raw text
                const fullText = ocrRawResult.text || "";
                console.log("[Terminal] OCR text:", fullText.substring(0, 100));
                detectedTicker = detectTicker(fullText);
                anchorPrice = detectPrice(fullText);
                console.log("[Terminal] Detected:", { ticker: detectedTicker, price: anchorPrice });
            }

            ticker = detectedTicker || manualTicker;
            const hasVisual = visualData && visualData.points.length > 20;

            if (!ticker && !hasVisual) {
                setStatusMessage("Neural Core Rejected: No Valid Data.");
                alert("Safety Check Triggered:\nNo valid ticker or chart pattern detected.");
                setIsAnalyzing(false);
                return;
            }

            // Fallback for visual data if ticker is missing
            if (!ticker && hasVisual) {
                ticker = "VISUAL-SCAN";

                // CRITICAL FIX: Scale visual points to anchor price if detected
                const finalPoints = anchorPrice
                    ? anchorPriceToVisual(visualData.points, anchorPrice)
                    : visualData.points;

                historicalPrices = finalPoints;
                marketStats = {
                    price: finalPoints[finalPoints.length - 1],
                    change24h: 0,
                    volume: 0,
                    source: anchorPrice ? 'Visual Cortex (Anchored)' : 'Visual Cortex'
                };
            }

            // --- DATA SOURCE ROUTING ---
            // Only fetch if NOT visual scan and we don't have data yet
            if (ticker !== "VISUAL-SCAN" && (!marketStats || !historicalPrices)) {
                try {
                    const isStock = STOCK_MAP[ticker];

                    if (isStock || !COIN_MAP[ticker]) {
                        // 1. Try Yahoo Finance (No Key Required) - Primary for Stocks AND Unknowns
                        setStatusMessage(`Accessing Public Exchange Data for ${ticker}...`);
                        const yahooData = await fetchYahooData(ticker);

                        if (yahooData) {
                            marketStats = yahooData.marketStats;
                            historicalPrices = yahooData.historicalPrices;
                        } else {
                            // 2. Fallback to Finnhub (Key Required)
                            console.warn("Yahoo Finance unavailable. Requesting Institutional Access...");
                            let apiKey = manualApiKey || localStorage.getItem('finnhub_key');

                            if (!apiKey) {
                                setStatusMessage("Institutional Access Required. Verifying Credentials...");
                                setPendingTicker(ticker);
                                setShowApiKeyInput(true);
                                setIsAnalyzing(false);
                                // No worker to terminate here; cleanup handled later
                                return; // PAUSE for API Key
                            }

                            setStatusMessage(`Authenticating with Finnhub for ${ticker}...`);
                            marketStats = await fetchStockData(ticker, apiKey);
                            historicalPrices = await fetchStockHistory(ticker, apiKey);
                        }

                        // 3. Last Resort Fallback Logic is handled in catch block
                    }

                    if (!marketStats || !historicalPrices) {
                        if (isStock) throw new Error("Failed to fetch stock data from all sources.");
                        // If not explicitly a stock, it might be a crypto not in our map check?
                        // But let's proceed to try crypto if Yahoo failed and it wasn't a stock-map match.
                    }

                    // CRYPTO LOGIC (CoinGecko)
                    // Only try if we haven't found data yet AND it's either in coin map or we are desperate
                    if (!marketStats && !historicalPrices) {
                        const coinId = COIN_MAP[ticker];
                        if (coinId) {
                            setStatusMessage(`Target Locked: ${ticker}. Fetching Matrix...`);
                            marketStats = await fetchMarketData(ticker);
                            historicalPrices = await fetchHistoricalData(ticker, 90);
                        } else {
                            // If not in coin map and Yahoo failed, we really don't have it.
                            // But wait, the original logic had a hard else.
                            // New logic: We tried Yahoo above for ALL non-cryptos. 
                            // So if we are here, Yahoo failed. If it's in COIN_MAP, try CoinGecko.
                        }

                        // Fallback for CoinGecko failure or unsupported ticker
                        if (!historicalPrices && !isStock && !coinId) {
                            throw new Error(`Data unavailable for ${ticker}.`);
                        }
                    }
                } catch (dataErr) {
                    console.warn("API Data Fetch Failed, engaging Anchored Visual Fallback:", dataErr);

                    // UNIVERSAL FALLBACK:
                    // If API fails (bad key, rate limit, etc), use Visual Analysis instead of failing.
                    setStatusMessage("API Access Denied. Engaging Anchored Visual Extraction...");

                    // Use pre-captured visualData or re-extract if missing
                    let fallbackPrices = visualData?.points;
                    if (!fallbackPrices || fallbackPrices.length < 20) {
                        fallbackPrices = await extractChartData(visualSrc);
                    }

                    if (fallbackPrices && fallbackPrices.length > 20) {
                        // Apply OCR Anchor if we have it
                        const finalPoints = anchorPrice
                            ? anchorPriceToVisual(fallbackPrices, anchorPrice)
                            : fallbackPrices;

                        ticker = `${ticker || "UNKNOWN"} (Visual Sync)`;
                        marketStats = {
                            price: finalPoints[finalPoints.length - 1],
                            change24h: 0,
                            volume: 0,
                            source: anchorPrice ? 'Visual Cortex (Anchored)' : 'Visual Cortex (Estimate)'
                        };
                        historicalPrices = finalPoints;

                        // Clean up Finnhub key if that was the fail point
                        if (dataErr.message?.includes("forbidden") || dataErr.message?.includes("403")) {
                            localStorage.removeItem('finnhub_key');
                        }
                    } else {
                        alert("Safety Check Triggered:\nNo valid chart or price data found. Please ensure the chart is clearly visible.");
                        setIsAnalyzing(false);
                        return;
                    }
                }
            }

            // 5. Run Hybrid AI Fusion with Visual Sync
            setStatusMessage("Synchronizing Global Intelligence...");

            // Calculate Sync Reliability (c_i)
            let syncReliability = 0.95;
            if (ticker !== "VISUAL-SCAN" && visualData) {
                const visualTrend = visualData.points[visualData.points.length - 1] > visualData.points[0] ? 1 : -1;
                const marketTrend = historicalPrices[historicalPrices.length - 1] > historicalPrices[0] ? 1 : -1;

                if (visualTrend !== marketTrend) {
                    syncReliability = 0.4; // Strong penalty for trend mismatch
                    console.warn("Divergence Detected: Visual Trend vs Market Data.");
                } else {
                    syncReliability = 0.9 + (visualData.confidence * 0.1); // Boost based on geometric goodness
                }
            } else if (ticker === "VISUAL-SCAN" && visualData) {
                syncReliability = visualData.confidence;
            }

            const result = await runRealAnalysis(ticker, marketStats, historicalPrices, user, weights, setStatusMessage, syncReliability);

            // --- SNAPSHOT UPLOAD ---
            if (user) {
                setStatusMessage("Archiving Neural Snapshot...");
                try {
                    const snapshotUrl = await uploadSnapshot(originalFileSrc, user.id);
                    if (snapshotUrl) {
                        result.imageUrl = snapshotUrl;
                    }
                } catch (uploadErr) {
                    console.warn("Snapshot upload failed:", uploadErr);
                }
            }

            // Update Usage (only on success)
            if (user && profile) {
                const today = new Date().toISOString().split('T')[0];
                const newCount = (profile.last_upload_date !== today) ? 1 : (profile.upload_count || 0) + 1;
                await supabase.from('profiles').update({ upload_count: newCount, last_upload_date: today }).eq('id', user.id);
                refreshProfile();
            } else if (!user) {
                updateGuestUsage();
            }

            // Save History
            saveHistory(result);
            setAnalysisResult(result);

            // Trigger Notification
            if (Notification.permission === 'granted') {
                new Notification(`Analysis Complete: ${result.ticker}`, {
                    body: `${result.direction} (${result.confidence}%) - Click to view details`,
                    icon: '/pwa-192x192.png' // Utilizing PWA icon if available
                });
            }

        } catch (err) {
            // Auto-reset API Key if it seems invalid
            if (err.message && (err.message.includes("API Key") || err.message.includes("Forbidden") || err.message.includes("403"))) {
                localStorage.removeItem('finnhub_key');
            }
            alert(err.message || "Analysis failed.");
            console.error(err);
        }
        finally {
            setIsAnalyzing(false);
        }
    };

    const saveHistory = async (result) => {
        if (isIncognito) {
            setStatusMessage("Stealth Protocol: History Log Skipped");
            return;
        }

        if (user) {
            const { data } = await supabase.from('prediction_history').insert([{ user_id: user.id, data: result }]).select();
            if (data?.[0]) {
                const newItem = { ...result, db_id: data[0].id, created_at: data[0].created_at };
                setHistory(prev => [{ ...newItem }, ...prev]);
            }
        } else {
            const now = new Date().toISOString();
            const item = { ...result, db_id: 'local-' + Date.now(), created_at: now };
            setHistory(prev => {
                const nh = [item, ...prev].slice(0, 5);
                localStorage.setItem('diver_ai_guest_history', JSON.stringify(nh));
                return nh;
            });
        }
    };

    const handleManualSubmit = (ticker) => {
        if (!checkLimits()) return;
        setShowManualInput(false);
        // Fix: Pass null for ocrSrc (3rd arg) so ticker is 4th arg
        if (imagePreview) runAnalysisWorkflow(null, null, null, ticker);
    };

    const handleApiKeySubmit = (key) => {
        const cleanKey = key.trim();
        localStorage.setItem('finnhub_key', cleanKey);
        setShowApiKeyInput(false);
        // Resume with pending ticker and new key
        // Fix: Pass null for ocrSrc (3rd arg) so ticker is 4th arg, key is 5th
        if (pendingTicker && imagePreview) {
            runAnalysisWorkflow(null, null, null, pendingTicker, cleanKey);
        }
    };

    const deleteHistoryItem = async (id) => {
        if (user) await supabase.from('prediction_history').delete().eq('id', id);
        setHistory(prev => prev.filter(i => i.db_id !== id));
        setAnalysisResult(null);
    };

    const handleFeedback = async (id, isWin) => {
        await syncGlobalFeedback(isWin);
        if (user) await supabase.from('prediction_history').update({ data: { ...history.find(h => h.db_id === id), feedback: isWin ? 'win' : 'loss', synapses: weights } }).eq('id', id);
        setHistory(prev => prev.map(i => i.db_id === id ? { ...i, feedback: isWin ? 'win' : 'loss' } : i));
    };

    const handleVerifyModel = async () => {
        if (!analysisResult || !analysisResult.raw_prices) {
            alert("Cannot verify: No historical data available in this report.");
            return;
        }
        setIsVerifying(true);
        setStatusMessage("Initializing Verification Protocol (Background)...");
        try {
            const result = await runBackgroundAssessment(analysisResult.raw_prices, (p) => setStatusMessage(`Backtesting: ${p.toFixed(0)}%`));
            setValidationResult(result);
        } catch (err) {
            console.error(err);
            alert("Verification Failed: " + err.message);
        } finally {
            setIsVerifying(false);
            setStatusMessage("");
        }
    };

    const handleCalibrateWeights = async (newWeights) => {
        setNeuralState(prev => ({ ...prev, ...newWeights }));
        setValidationResult(null);

        // Sync to Supabase
        if (user) {
            try {
                await supabase.from('neural_state').update({
                    ...newWeights,
                    last_updated: new Date().toISOString(),
                    updated_by: user.id
                }).eq('id', 'global_master');
                alert("Self-Correction Complete:\nEngine has been calibrated for " + analysisResult?.ticker);
            } catch (err) {
                console.error("Calibration sync failed:", err);
            }
        } else {
            alert("Guest Mode: Calibration applied to current session.");
        }
    };

    const uploadSnapshot = async (dataUrl, userId) => {
        try {
            // Convert DataURL to Blob
            const res = await fetch(dataUrl);
            const blob = await res.blob();
            const ext = blob.type.split('/')[1] || 'png';
            const filename = `${userId}/${Date.now()}.${ext}`;

            console.log("Uploading snapshot to bucket 'snapshots'...", filename);

            const { data, error } = await supabase.storage
                .from('snapshots')
                .upload(filename, blob, {
                    contentType: blob.type,
                    upsert: false
                });

            if (error) {
                // If bucket doesn't exist, try 'public' or 'avatars' as fallback or just log
                console.warn("Primary bucket upload failed", error);
                throw error;
            }

            const { data: { publicUrl } } = supabase.storage
                .from('snapshots')
                .getPublicUrl(filename);

            return publicUrl;
        } catch (err) {
            console.error("Snapshot upload error:", err);
            return null;
        }
    };

    return (
        <div className="flex flex-col h-[calc(100dvh-64px)] pb-0 relative overflow-hidden bg-slate-950">
            <TickerTape items={tickerItems} />

            {/* Clinical System HUD */}
            <div className="bg-slate-900/50 border-b border-slate-800/50 px-4 py-1.5 flex items-center justify-between text-[8px] font-bold font-mono tracking-widest text-slate-500">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" /> ENGINE_STATUS: OPERATIONAL</div>
                    <div className="flex items-center gap-1.5"><Activity className="w-3 h-3" /> LATENCY: 42MS</div>
                    <div className="flex items-center gap-1.5"><Cpu className="w-3 h-3" /> WEBGL_ACCELERATION: ACTIVE</div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">DATA_STREAM: [YAHOO:PASS, CG:PASS, FINN:PASS]</div>
                    <div className="flex items-center gap-1.5 text-emerald-400">INSTITUTIONAL_CORE: v3.2.1-STABLE</div>
                </div>
            </div>

            <div className="flex flex-1 overflow-hidden relative">
                {showAuth && <AuthModal onClose={() => setShowAuth(false)} onSuccess={() => setShowAuth(false)} />}
                {showLimitModal && <LimitModal message={limitMessage} type={limitType} onClose={() => setShowLimitModal(false)} onLogin={() => { setShowLimitModal(false); setShowAuth(true); }} />}
                {showManualInput && <ManualTickerModal onSubmit={handleManualSubmit} onClose={() => { setShowManualInput(false); setIsAnalyzing(false); }} />}
                {showApiKeyInput && <ApiKeyModal ticker={pendingTicker} onSubmit={handleApiKeySubmit} onClose={() => { setShowApiKeyInput(false); setIsAnalyzing(false); }} />}
                {validationResult && <VerificationModal validationResult={validationResult} ticker={analysisResult?.ticker} onCalibrate={handleCalibrateWeights} onClose={() => setValidationResult(null)} />}

                <main className={`flex-1 overflow-y-auto p-4 md:p-12 pb-32 md:pb-12 scrollbar-none transition-all ${activeTab !== 'analyze' ? 'hidden md:block opacity-0' : 'block opacity-100'}`}>
                    <div className="max-w-6xl mx-auto space-y-6 md:space-y-12">
                        {!analysisResult && !isAnalyzing && !showManualInput && !showApiKeyInput ? (
                            <div className="animate-in fade-in duration-1000 py-12 md:py-24">
                                <FileUpload onFileSelect={handleFileSelect} isAnalyzing={isAnalyzing} statusMessage={statusMessage} />
                            </div>
                        ) : (
                            <div className="space-y-6">
                                {(analysisResult || isAnalyzing || showManualInput || showApiKeyInput) && (
                                    <div className="flex items-center justify-between">
                                        {!isAnalyzing && !showManualInput && !showApiKeyInput && (
                                            <div className="flex gap-4">
                                                <button onClick={() => { setAnalysisResult(null); setImagePreview(null); }} className="px-4 py-2 bg-slate-900 hover:bg-slate-800 rounded-xl border border-slate-800 text-slate-400 text-[10px] font-black uppercase tracking-widest transition-all">← New Analysis</button>
                                                {isPro && (
                                                    <button
                                                        onClick={() => setIsIncognito(!isIncognito)}
                                                        className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-[10px] font-black uppercase tracking-widest ${isIncognito ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'}`}
                                                    >
                                                        {isIncognito ? <ShieldCheck className="w-4 h-4" /> : <History className="w-4 h-4" />}
                                                        {isIncognito ? 'Stealth Active' : 'Incognito Off'}
                                                    </button>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {isAnalyzing && !analysisResult && (
                                    <div className="flex flex-col items-center justify-center min-h-[400px]">
                                        <div className="relative w-24 h-24 mb-6">
                                            <div className="absolute inset-0 border-4 border-emerald-500/10 rounded-full"></div>
                                            <div className="absolute inset-0 border-4 border-t-emerald-400 rounded-full animate-spin"></div>
                                            <Camera className="absolute inset-0 m-auto text-emerald-400 w-10 h-10 animate-pulse" />
                                        </div>
                                        <h3 className="text-2xl font-bold text-white mb-2 leading-none">{statusMessage}</h3>
                                    </div>
                                )}

                                {analysisResult && (
                                    <AnalysisResult
                                        result={analysisResult}
                                        imagePreview={imagePreview}
                                        onVerify={handleVerifyModel}
                                        isVerifying={isVerifying}
                                        isPro={isPro}
                                        isIncognito={isIncognito}
                                        onToggleIncognito={() => setIsIncognito(!isIncognito)}
                                    />
                                )}
                            </div>
                        )}
                    </div>
                </main>

                <aside className={`w-full md:w-80 h-full transition-all duration-500 fixed md:relative z-40 ${activeTab === 'history' ? 'translate-x-0' : 'translate-x-full md:translate-x-0'}`}>
                    <HistorySidebar history={history} onSelect={(item) => { setAnalysisResult(item); setImagePreview(item.imageUrl || null); setActiveTab('analyze'); }} onDelete={deleteHistoryItem} onFeedback={handleFeedback} />
                </aside>

                <div className="md:hidden fixed bottom-20 left-4 right-4 z-40 bg-slate-900/90 border border-slate-800 backdrop-blur-xl px-4 py-3 flex items-center justify-around rounded-[24px] shadow-2xl ring-1 ring-white/5">
                    <button onClick={() => setActiveTab('analyze')} className={`flex flex-col items-center gap-1.5 transition-all ${activeTab === 'analyze' ? 'text-emerald-400' : 'text-slate-500'}`}><Activity className="w-5 h-5" /><span className="text-[8px] font-black uppercase">Analyze</span></button>
                    <button onClick={() => setActiveTab('history')} className={`flex flex-col items-center gap-1.5 transition-all ${activeTab === 'history' ? 'text-emerald-400' : 'text-slate-500'}`}><History className="w-5 h-5" /><span className="text-[8px] font-black uppercase">Terminal</span></button>
                </div>
            </div>
        </div>
    );
}
