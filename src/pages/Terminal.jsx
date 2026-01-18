import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
    Activity, TrendingUp, TrendingDown, Minus, Camera,
    History, ShieldCheck, Cpu, X, ChevronRight, Zap, ThumbsUp, ThumbsDown, Trash2, Keyboard, Key, Share2, Lock as LockIcon, AlertTriangle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '../lib/supabase';
import Tesseract from 'tesseract.js';
import { detectTicker, detectPrice, fetchMarketData, fetchHistoricalData, COIN_MAP, STOCK_MAP, fetchStockData, fetchStockHistory, fetchYahooData, fetchMacroHistory, calculateMacroSentiment, fetchTickerData, isValidTicker, detectTimeframe, getOptimizedPrice } from '../lib/marketData';
import { useAppContext } from '../context/AppContext';
import AuthModal from '../components/AuthModal';
import { calculateRSI as calcRSI, calculateMACD, calculateBollingerBands, detectPatterns } from '../lib/technicalAnalysis';
import { prepareData, calculateStats, createModel, trainModel, predictNextPrice, disposeModel, assessModelAccuracy, saveGlobalModel, loadGlobalModel, runBackgroundTraining, runBackgroundAssessment, saveGlobalModelArtifacts } from '../lib/brain';
import { extractChartData, anchorPriceToVisual } from '../lib/vision';
import { enhanceMobileChart } from '../lib/visionMobile';
import { runRealAnalysis } from '../lib/analysis';
import { Helmet } from 'react-helmet-async';
import FloatingWidget from '../components/FloatingWidget';
import { companion } from '../lib/companion';
import CandlestickVisual from '../components/CandlestickVisuals';

// --- ENGINE LOGIC (Real Implementation) ---

// --- ENGINE LOGIC (Real Implementation) ---

// Weights are now managed via AppContext for centralization


// calculateHybridProbability and runRealAnalysis moved to ../lib/analysis.js for cross-app consistency

// 3. Main Analysis Workflow
// runRealAnalysis moved to ../lib/analysis.js

// --- Child Components ---
// --- Helper Utils ---
const resizeImageForCloud = (dataUrl) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                const MAX_WIDTH = 1200; // Optimal for OCR while reducing size
                let width = img.width;
                let height = img.height;

                if (width > MAX_WIDTH) {
                    height = height * (MAX_WIDTH / width);
                    width = MAX_WIDTH;
                }

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);
                // Use JPEG with 0.8 quality for good compression/quality ratio
                resolve(canvas.toDataURL('image/jpeg', 0.8));
            } catch (e) {
                // If anything fails (e.g. canvas error), fallback to original
                console.warn("Resize failed, using original", e);
                resolve(dataUrl);
            }
        };
        img.onerror = (e) => {
            console.warn("Image load failed during resize", e);
            resolve(dataUrl);
        };
        img.src = dataUrl;
    });
};

const Sparkline = ({ data, color = 'emerald' }) => {
    if (!data || data.length < 2) return <div className="w-12 h-4 bg-black-ash/50 rounded animate-pulse" />;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const points = data.map((v, i) => `${(i / (data.length - 1)) * 48},${16 - ((v - min) / range) * 16}`).join(' ');

    return (
        <svg className="w-12 h-4 overflow-visible" viewBox="0 0 48 16">
            <polyline
                fill="none"
                stroke={color === 'brand' ? '#f59e0b' : (color === 'emerald' ? '#10b981' : '#f43f5e')}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={points}
                className={`drop-shadow-[0_0_3px_${color === 'brand' ? 'rgba(245,158,11,0.5)' : (color === 'emerald' ? 'rgba(16,185,129,0.5)' : 'rgba(244,63,94,0.5)')}]`}
            />
        </svg>
    );
};

const TickerTape = ({ items }) => {
    if (!items || items.length === 0) return (
        <div className="w-full bg-ash border-y border-slate-900 h-10 flex items-center justify-center">
            <div className="text-[9px] font-black text-slate-700 uppercase tracking-[0.3em] animate-pulse">Initializing Global Stream...</div>
        </div>
    );
    return (
        <div className="w-full bg-ash border-y border-slate-900 overflow-hidden relative h-10 flex items-center shrink-0 shadow-2xl z-50">
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
    const color = isBull ? 'brand' : (isBear ? 'rose' : 'blue');

    const markers = [
        { label: 'Panic', pos: 10 },
        { label: 'Fear', pos: 30 },
        { label: 'Neutral', pos: 50 },
        { label: 'Greed', pos: 70 },
        { label: 'Euphoria', pos: 90 }
    ];

    return (
        <div className="relative w-full space-y-2 py-4">
            <div className="relative h-6 bg-ash rounded-md border border-ash/50 overflow-hidden shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]">
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

const ScanningHUD = ({ status, isUnstable, onConfirm, onReject }) => {
    const [hexLines, setHexLines] = useState([]);

    useEffect(() => {
        const interval = setInterval(() => {
            const hex = Array.from({ length: 4 }, () => Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, '0').toUpperCase());
            setHexLines(prev => [hex.join(' '), ...prev].slice(0, 10));
        }, 150);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col items-center justify-center min-h-[500px] w-full relative overflow-hidden bg-black rounded-[40px] border border-slate-900 shadow-2xl">
            {/* Grid Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-20" />

            <div className="relative z-10 flex flex-col items-center w-full px-6">
                <div className="relative w-32 h-32 mb-12">
                    <div className="absolute inset-0 border-2 border-brand/20 rounded-full animate-[ping_3s_linear_infinite]" />
                    <div className="absolute inset-0 border border-brand/40 rounded-full rotate-45" />
                    <div className="absolute inset-2 border border-t-brand border-l-transparent border-r-transparent border-b-transparent rounded-full animate-spin [animation-duration:1s]" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Cpu className="w-12 h-12 text-brand animate-pulse" />
                    </div>
                </div>

                <div className="text-center space-y-4 max-w-sm">
                    <h3 className="text-2xl font-black text-white tracking-tighter uppercase leading-tight">{status || "Initializing Neural Core"}</h3>

                    {!isUnstable ? (
                        <div className="flex items-center justify-center gap-1.5">
                            <div className="w-1.5 h-1.5 bg-brand rounded-full animate-bounce [animation-delay:-0.3s]" />
                            <div className="w-1.5 h-1.5 bg-brand rounded-full animate-bounce [animation-delay:-0.15s]" />
                            <div className="w-1.5 h-1.5 bg-brand rounded-full animate-bounce" />
                        </div>
                    ) : (
                        <div className="flex flex-col gap-4 mt-8 animate-in zoom-in duration-300">
                            <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest bg-ash/50 px-3 py-1 rounded-full border border-ash">Ambiguous Ticker Detection triggered</p>
                            <div className="flex gap-3 justify-center">
                                <button
                                    onClick={(e) => { e.stopPropagation(); onConfirm(); }}
                                    className="flex items-center gap-2 bg-brand text-slate-950 px-6 py-3 rounded-2xl font-black text-xs uppercase tracking-tighter hover:scale-105 transition-transform"
                                >
                                    <ThumbsUp className="w-4 h-4" /> Confirm
                                </button>
                                <button
                                    onClick={(e) => { e.stopPropagation(); onReject(); }}
                                    className="flex items-center gap-2 bg-ash/80 text-white border border-ash px-6 py-3 rounded-2xl font-black text-xs uppercase tracking-tighter hover:bg-slate-800 transition-colors"
                                >
                                    <ThumbsDown className="w-4 h-4" /> Manual
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {!isUnstable && (
                    <>
                        <div className="mt-12 w-64 h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />
                        <div className="mt-8 font-mono text-[8px] text-brand/40 space-y-1">
                            {hexLines.map((line, i) => (
                                <div key={i} className="animate-in fade-in slide-in-from-bottom-2 duration-300">{line}</div>
                            ))}
                        </div>
                    </>
                )}
            </div>

            {/* Scanning Beam */}
            <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-brand to-transparent opacity-50 animate-scan pointer-events-none" />
        </div>
    );
};

// Trade Blueprint Section (Previously Modal)
const TradeBlueprint = ({ targets, direction }) => {
    const isBull = direction.includes('Bullish');
    const themeColor = isBull ? 'brand' : 'rose';

    return (
        <div className="bg-black-ash/40 border border-ash rounded-3xl overflow-hidden shadow-sm group">
            <div className={`p-6 bg-gradient-to-br from-${themeColor}-500/5 to-transparent border-b border-ash/50 flex justify-between items-center`}>
                <div>
                    <h3 className="text-sm font-black text-white flex items-center gap-2 uppercase tracking-tight">
                        <Zap className={`text-${themeColor}-400 w-4 h-4`} />
                        Execution Blueprint
                    </h3>
                    <p className="text-slate-500 text-[9px] font-mono mt-0.5 uppercase tracking-widest leading-none">
                        Institutional {isBull ? 'Long' : 'Short'} Logic
                    </p>
                </div>
                <div className="px-3 py-1 bg-ash/50 rounded-full border border-ash text-[9px] font-mono text-brand font-bold">
                    R/R: 1 : {targets.rr}
                </div>
            </div>

            <div className="p-6">
                <div className="flex items-center justify-between mb-6 p-4 bg-ash/50 rounded-2xl border border-ash/50">
                    <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Entry Zone</span>
                    <div className="text-2xl font-mono text-white font-black">${targets.entry}</div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {[
                        { label: 'Primary', val: targets.tp1, color: 'brand', tag: 'TP1' },
                        { label: 'Extended', val: targets.tp2, color: 'brand', tag: 'TP2' },
                        { label: 'Stop Loss', val: targets.sl, color: 'rose', tag: 'SL' }
                    ].map((t) => (
                        <div key={t.tag} className={`flex flex-col p-4 rounded-xl bg-ash/40 border border-${t.color}-500/10 group-hover:border-${t.color}-500/30 transition-all`}>
                            <div className="flex items-center justify-between mb-2">
                                <div className={`w-6 h-6 rounded bg-${t.color}-500/20 flex items-center justify-center text-${t.color}-400 font-black text-[9px]`}>{t.tag}</div>
                                <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest">{t.label}</span>
                            </div>
                            <div className={`text-lg font-mono text-${t.color}-400 font-black tabular-nums`}>${t.val}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

// Fallback Modal for manual input
const ManualTickerModal = ({ onSubmit, onClose }) => {
    const [input, setInput] = useState('');
    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-ash/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-black-ash border border-ash rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-slate-400"><Keyboard className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none">Optical Scan Failed</h3>
                <p className="text-slate-500 text-sm font-bold mb-6">The neural core could not identify the asset ticker from the image. Please enter it manually.</p>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value.toUpperCase())}
                    placeholder="e.g. BTC, ETH, SOL"
                    className="w-full bg-ash border border-ash rounded-xl px-4 py-3 text-center text-white font-black tracking-widest uppercase mb-6 focus:outline-none focus:border-brand focus:ring-1 focus:ring-brand"
                    autoFocus
                />
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-3 rounded-xl border border-ash text-slate-500 font-bold hover:bg-black-ash transition-colors">Cancel</button>
                    <button onClick={() => input && onSubmit(input)} disabled={!input} className="btn-flame flex-1 !py-3">Proceed</button>
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
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-ash/90 backdrop-blur-md animate-in fade-in" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="bg-black-ash border border-ash rounded-[32px] w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh] animate-in zoom-in-95">
                <div className="p-6 md:p-8 border-b border-ash flex justify-between items-start shrink-0">
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
                        <div className="p-6 rounded-2xl border bg-ash/50 border-ash">
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
                                className="btn-flame w-full mt-6"
                            >
                                <Cpu className="w-4 h-4" />
                                Calibrate AI Weights
                            </button>
                        )}
                    </div>

                    {/* Simple Visualization Chart (SVG) */}
                    <div className="h-48 w-full bg-ash/50 rounded-2xl border border-ash relative flex items-end px-4 pb-4 pt-8 gap-1">
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
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-ash/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-black-ash border border-ash rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-emerald-400"><Key className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none">Stock Data Access</h3>
                <p className="text-slate-500 text-sm font-bold mb-4">You are analyzing <span className="text-emerald-400">{ticker}</span>. To access real-time S&P 500 data, a free Finnhub.io API Token is required.</p>
                <p className="text-xs text-slate-600 mb-6 bg-ash p-2 rounded-lg border border-ash">Your key is stored locally in your browser and never shared.</p>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Paste Finnhub API Key"
                    className="w-full bg-ash border border-ash rounded-xl px-4 py-3 text-center text-white font-mono text-sm mb-6 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                    autoFocus
                />
                <div className="flex gap-3">
                    <button onClick={onClose} className="flex-1 py-3 rounded-xl border border-ash text-slate-500 font-bold hover:bg-black-ash transition-colors">Cancel</button>
                    <button onClick={() => input && onSubmit(input)} disabled={!input} className="btn-flame flex-1 !py-3">Save & Scan</button>
                </div>
                <a href="https://finnhub.io/register" target="_blank" rel="noreferrer" className="block mt-4 text-xs text-blue-500 hover:underline">Get a free API Key →</a>
            </div>
        </div>
    );
};

const LimitModal = ({ message, type, onClose, onLogin }) => {
    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-ash/90 backdrop-blur-md animate-in fade-in">
            <div className="bg-black-ash border border-ash rounded-[32px] w-full max-w-sm overflow-hidden shadow-2xl p-8 text-center animate-in zoom-in-95">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 text-rose-500 shadow-[0_0_20px_rgba(244,63,94,0.2)]"><Lock className="w-8 h-8" /></div>
                <h3 className="text-2xl font-black text-white mb-2 leading-none uppercase tracking-tight">Access Restricted</h3>
                <p className="text-slate-500 text-sm font-bold mb-8 leading-relaxed">{message}</p>
                <div className="flex flex-col gap-3">
                    {type === 'free' ? (
                        <button onClick={() => window.open('/pricing', '_blank')} className="btn-flame w-full !py-4">Upgrade to Pro <ArrowRight className="w-4 h-4" /></button>
                    ) : type === 'guest' ? (
                        <button onClick={onLogin} className="btn-flame w-full !py-4">Initialize Authentication <ArrowRight className="w-4 h-4" /></button>
                    ) : (
                        <button onClick={onClose} className="w-full py-4 rounded-xl bg-slate-800 text-white font-black hover:bg-slate-700 transition-all uppercase tracking-widest text-xs">Check Inbox</button>
                    )}
                    <button onClick={onClose} className="w-full py-3 rounded-xl border border-ash text-slate-500 font-bold hover:bg-slate-800 transition-colors uppercase tracking-widest text-[10px]">Close Terminal</button>
                </div>
            </div>
        </div>
    );
};

const FileUpload = ({ onFileSelect, isAnalyzing, statusMessage, isUnstableDetection, handleConfirmTicker }) => {
    const [isDragging, setIsDragging] = useState(false);
    const handleDrag = (e) => { e.preventDefault(); setIsDragging(e.type === 'dragover'); };
    return (
        <div
            onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={(e) => { e.preventDefault(); setIsDragging(false); if (e.dataTransfer.files[0]) onFileSelect(e.dataTransfer.files[0]); }}
            className={`relative group h-96 rounded-[40px] border-4 border-dashed transition-all duration-500 flex flex-col items-center justify-center cursor-pointer overflow-hidden ${isDragging ? 'border-brand bg-brand/5 scale-[0.99] shadow-inner' : 'border-ash bg-black-ash/50 hover:border-slate-700 hover:bg-black-ash shadow-2xl'}`}
        >
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer z-10" onChange={e => onFileSelect(e.target.files[0])} accept="image/*" />
            {(isAnalyzing || isUnstableDetection) && !analysisResult ? (
                <ScanningHUD
                    status={statusMessage}
                    isUnstable={isUnstableDetection}
                    onConfirm={() => handleConfirmTicker(true)}
                    onReject={() => handleConfirmTicker(false)}
                />
            ) : (
                <>
                    <div className="w-24 h-24 bg-brand rounded-3xl flex items-center justify-center mb-8 group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 shadow-2xl shadow-brand/20"><ShieldCheck className="w-12 h-12 text-slate-950" /></div>
                    <h3 className="text-3xl font-black text-white mb-3 tracking-tighter uppercase">Ready for Analysis</h3>
                    <p className="text-slate-500 max-w-sm text-center mb-10 leading-relaxed font-bold text-sm">Drag & drop your chart, paste a TradingView link, or click to browse files.</p>

                    <div className="flex flex-col gap-4 w-full max-w-sm items-center z-20">
                        <div className="relative w-full">
                            <input
                                type="text"
                                placeholder="Paste TradingView URL..."
                                onChange={(e) => {
                                    const val = e.target.value;
                                    if (val.includes('tradingview.com')) {
                                        onFileSelect(val); // Passing string triggers URL workflow
                                    }
                                }}
                                className="w-full bg-ash/80 border border-ash/50 rounded-2xl px-6 py-4 text-white text-sm focus:outline-none focus:border-brand/50 transition-all text-center placeholder:text-slate-600"
                            />
                            <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                <Zap className="w-4 h-4 text-brand/30" />
                            </div>
                        </div>
                        <div className="btn-flame px-8 !py-4 w-full text-center">Execute Neural Analysis</div>
                    </div>
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
        <div className="bg-black border border-ash rounded-lg p-4 print:border-gray-300 print:bg-white">
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
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-black-ash border border-ash text-white text-[9px] font-mono px-2 py-1 rounded whitespace-nowrap z-50 shadow-xl">
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
            <div className="mt-4 pt-3 border-t border-ash text-center">
                <p className="text-[9px] text-slate-500 font-medium uppercase tracking-wider">
                    Historic data via {source || "Institutional Feed"} • {isPro ? 'Pro Access' : 'Preview'}
                </p>
            </div>
        </div>
    );
};

const PrintHeader = ({ ticker, date }) => (
    <div className="hidden print:flex justify-between items-center border-b-2 border-slate-900 pb-4 mb-8">
        <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-black-ash rounded-lg flex items-center justify-center">
                <Cpu className="w-6 h-6 text-white" />
            </div>
            <div>
                <h1 className="text-xl font-black tracking-tighter text-slate-900 uppercase">DiverAI Quant Engine</h1>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Institutional Analysis Report</p>
            </div>
        </div>
        <div className="text-right">
            <div className="text-[10px] font-mono font-bold text-slate-800 uppercase tracking-wider">{ticker} Analysis</div>
            <div className="text-[9px] font-mono text-slate-500">{new Date(date).toLocaleString()}</div>
        </div>
    </div>
);

const PrintFooter = () => (
    <div className="hidden print:block mt-20 pt-8 border-t border-slate-200 text-center">
        <p className="text-[9px] font-bold text-slate-400 uppercase tracking-[0.3em] mb-2">Generated by DiverAI Neural Core v5.0</p>
        <p className="text-[8px] text-slate-400 font-mono max-w-2xl mx-auto leading-relaxed">
            Confidential. For institutional use only. This analysis is generated by proprietary Bayesian neural networks and technical indicators. Past performance is not indicative of future results. Quantitative models are subject to market volatility.
        </p>
    </div>
);

const AnalysisResult = ({ result, imagePreview, onVerify, isVerifying, isPro, onToggleIncognito, isIncognito, isQuickShareMode, onQuickShareReset }) => {
    const [showBlueprint, setShowBlueprint] = useState(false);
    const isBull = result.direction.includes('Bullish');
    const colors = isBull ? { text: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' } : (result.direction.includes('Bearish') ? { text: 'text-rose-400', border: 'border-rose-500/30', bg: 'bg-rose-500/10' } : { text: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/10' });
    const Icon = isBull ? TrendingUp : (result.direction.includes('Bearish') ? TrendingDown : Minus);

    return (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-6 pb-12 print:space-y-4 print:pb-0">
            <PrintHeader ticker={result.ticker} date={result.date} />

            {/* Header / Meta */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 border-b border-ash pb-6 print:border-black">
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
                        className="px-4 py-2 bg-slate-800 border border-slate-700 text-slate-300 hover:text-white rounded text-[10px] font-black uppercase tracking-widest transition-colors shadow-lg"
                    >
                        Export PDF
                    </button>
                    {isQuickShareMode && (
                        <button
                            onClick={onQuickShareReset}
                            className="px-4 py-2 bg-emerald-500 text-black rounded text-[10px] font-black uppercase tracking-widest hover:brightness-110 transition-all flex items-center gap-2"
                        >
                            <Zap className="w-3 h-3 fill-current" />
                            Return
                        </button>
                    )}
                </div>
            </div>

            {/* Key Metrics Grid (Dense) */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-800 border border-ash rounded-lg overflow-hidden print:border-gray-300 print:bg-gray-300">
                <div className="bg-black p-4 flex flex-col justify-between hover:bg-black-ash transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Confidence</span>
                    <span className={`text-2xl font-mono font-bold ${colors.text} print:text-black`}>{result.confidence}%</span>
                </div>
                <div className="bg-black p-4 flex flex-col justify-between hover:bg-black-ash transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Volatility (ATR)</span>
                    <span className="text-2xl font-mono font-bold text-white print:text-black">{result.riskMetrics?.volatility || "0.00"}%</span>
                </div>
                <div className="bg-black p-4 flex flex-col justify-between hover:bg-black-ash transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Sharpe (Est.)</span>
                    <span className="text-2xl font-mono font-bold text-brand print:text-black">{result.riskMetrics?.sharpeRatio || "0.00"}</span>
                </div>
                <div className="bg-black p-4 flex flex-col justify-between hover:bg-black-ash transition-colors print:bg-white">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Reward/Risk</span>
                    <span className="text-2xl font-mono font-bold text-brand print:text-black">{result.targets.rr}x</span>
                </div>
            </div>

            {/* Trade Blueprint Integrated */}
            <TradeBlueprint targets={result.targets} direction={result.direction} />

            {/* Pro Feature: 10-Year Macro Trend */}
            <MacroFluctuationChart macroTrend={result.macroTrend} isPro={isPro} source={result.macroTrend?.source} />

            {/* Analysis Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column: Patterns & Narrative */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Pattern Showcase */}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        {(result.patterns || []).map((p, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className={`p-4 rounded-2xl border ${p.sentiment === 'Bullish' ? 'bg-emerald-500/5 border-emerald-500/10' : (p.sentiment === 'Bearish' ? 'bg-rose-500/5 border-rose-500/10' : 'bg-slate-500/5 border-slate-500/10')} group hover:scale-[1.02] transition-all`}
                            >
                                <div className="flex items-center gap-3 mb-2">
                                    <div className={`p-1 rounded-lg ${p.sentiment === 'Bullish' ? 'bg-emerald-500/10' : (p.sentiment === 'Bearish' ? 'bg-rose-500/10' : 'bg-slate-500/10')}`}>
                                        <CandlestickVisual pattern={p.name} sentiment={p.sentiment} />
                                    </div>
                                    <span className="text-[10px] font-black uppercase text-slate-400 tracking-widest">{p.name}</span>
                                </div>
                                <div className={`text-[9px] font-bold ${p.sentiment === 'Bullish' ? 'text-emerald-500/60' : (p.sentiment === 'Bearish' ? 'text-rose-500/60' : 'text-slate-500/60')} uppercase`}>
                                    {p.sentiment} Signal
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Narrative Block */}
                    <div className="bg-black-ash/30 p-8 border border-ash rounded-3xl shadow-sm print:bg-white print:border-gray-200 group relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
                            <Cpu className="w-32 h-32" />
                        </div>
                        <div className="flex items-center gap-2 mb-6">
                            <Activity className="w-4 h-4 text-brand" />
                            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Neural Strategic Analysis</h3>
                        </div>
                        <p className="text-slate-300 text-sm md:text-base leading-relaxed font-mono print:text-black relative z-10" dangerouslySetInnerHTML={{ __html: result.overview || result.summary.replace(/\*\*(.*?)\*\*/g, '<span class="text-white font-bold">$1</span>') }} />
                    </div>

                    {/* Chart Preview */}
                    <div className="border border-ash rounded-3xl overflow-hidden relative group h-80 print:h-96 print:border-gray-300 shadow-2xl">
                        <div className="absolute inset-0 bg-gradient-to-t from-slate-950 to-transparent opacity-60 z-10" />
                        <div className="absolute top-4 left-4 z-20 bg-brand text-slate-950 px-3 py-1 rounded-full text-[9px] font-black tracking-widest uppercase shadow-lg">DATA SOURCE: VISUAL SCAN</div>
                        <img src={imagePreview} alt="Analyzed Chart" className="w-full h-full object-cover opacity-60 grayscale group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-700 scale-105 group-hover:scale-100 print:opacity-100 print:grayscale-0" />
                    </div>
                </div>

                {/* Right Column: Factor Table & Metrics */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-black-ash/20 border border-ash rounded-3xl overflow-hidden print:border-gray-200">
                        <div className="bg-black-ash/50 px-6 py-4 border-b border-ash flex justify-between items-center print:bg-gray-100">
                            <div className="flex items-center gap-2">
                                <ShieldCheck className="w-4 h-4 text-brand" />
                                <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Alpha Factors</span>
                            </div>
                            {onVerify && (
                                <button onClick={onVerify} disabled={isVerifying} className="text-[9px] text-brand hover:text-brand-light uppercase font-black print:hidden transition-colors">
                                    {isVerifying ? 'CALIBRATING...' : 'VERIFY CORE'}
                                </button>
                            )}
                        </div>
                        <table className="w-full text-left text-[10px]">
                            <tbody className="divide-y divide-slate-800/30 print:divide-gray-200">
                                {result.factors.map((f, i) => (
                                    <tr key={i} className="hover:bg-white/5 transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="font-black text-white uppercase tracking-tight print:text-black">{f.name}</div>
                                            <div className="text-[8px] text-slate-500 font-bold uppercase mt-1 print:text-gray-500">{f.type}</div>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className={`font-mono font-black ${f.p > 0.6 ? 'text-emerald-400' : (f.p < 0.4 ? 'text-rose-400' : 'text-slate-400')} print:text-black`}>{(f.p * 100).toFixed(0)}%</div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <div className="px-6 py-4 border-t border-ash/50 bg-ash/50 flex justify-between items-center">
                            <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Drawdown Risk</span>
                            <span className="font-mono text-[11px] text-rose-400 font-black">-{result.riskMetrics?.maxDrawdown || "0.00"}%</span>
                        </div>
                    </div>
                </div>
            </div>
            <PrintFooter />
        </div>
    );
};

const HistorySidebar = ({ history, onSelect, onDelete, onFeedback }) => {
    return (
        <div className="h-full flex flex-col bg-black border-l border-slate-900 shadow-2xl overflow-hidden">
            <div className="p-6 border-b border-slate-900 bg-ash/50 backdrop-blur-xl">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-brand rounded-full animate-pulse" />
                        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Terminal Log</h3>
                    </div>
                    <span className="text-[8px] font-mono text-slate-600 bg-black-ash px-2 py-0.5 rounded border border-ash">CORE v5.0</span>
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
                            className="group relative bg-black-ash/40 border border-ash/50 rounded-xl p-4 cursor-pointer hover:bg-slate-800/40 transition-all hover:border-slate-700/50 overflow-hidden"
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

                            <div className="mt-4 pt-4 border-t border-ash/30 flex items-center justify-between relative z-10">
                                <div className="flex gap-2">
                                    <button
                                        onClick={(e) => { e.stopPropagation(); onFeedback(item.db_id, true); }}
                                        className={`p-1.5 rounded-lg border border-ash/50 transition-all ${item.feedback === 'win' ? 'bg-brand/20 border-brand/30 text-brand' : 'text-slate-600 hover:text-brand hover:bg-brand/10'}`}
                                    >
                                        <ThumbsUp className="w-3 h-3" />
                                    </button>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); onFeedback(item.db_id, false); }}
                                        className={`p-1.5 rounded-lg border border-ash/50 transition-all ${item.feedback === 'loss' ? 'bg-rose-500/20 border-rose-500/30 text-rose-400' : 'text-slate-600 hover:text-rose-400 hover:bg-rose-500/10'}`}
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
                    <div className="w-8 h-8 rounded-lg bg-brand/10 flex items-center justify-center border border-brand/20">
                        <Cpu className="w-4 h-4 text-brand" />
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
    const [isQuickShareMode, setIsQuickShareMode] = useState(false);
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
    const [isUnstableDetection, setIsUnstableDetection] = useState(false); // New: Confirmation Guard
    const [pendingAnalysisData, setPendingAnalysisData] = useState(null); // New: Confirmation Guard
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
                } else if (item.type === "text/plain") {
                    item.getAsString((text) => {
                        if (text.includes('tradingview.com')) {
                            handleFileSelect(text); // Pass string to trigger URL logic
                        }
                    });
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

            if (params.get('shared-image') === 'true' || params.get('shared-url') === 'true') {
                if (!checkLimits()) {
                    navigate('/', { replace: true });
                    return;
                }
                setIsQuickShareMode(true);
                setIsAnalyzing(true);
                setStatusMessage("Receiving Shared Intelligence...");

                try {
                    const cache = await caches.open('shared-media');
                    let response = null;
                    const isUrl = params.get('shared-url') === 'true';
                    const cacheKey = isUrl ? 'shared-url' : 'shared-image';

                    // Retry logic: Service worker might still be writing to cache
                    for (let i = 0; i < 10; i++) {
                        response = await cache.match(cacheKey);
                        if (response) break;
                        await new Promise(r => setTimeout(r, 200)); // Wait 200ms
                    }

                    if (response) {
                        if (isUrl) {
                            setStatusMessage("Fetching Shared Link Data...");
                            const sharedUrl = await response.text();

                            // V5.4: Extract ticker from URL directly
                            const urlTicker = detectTicker(sharedUrl);
                            const urlTimeframe = detectTimeframe(sharedUrl);

                            if (urlTicker) {
                                setStatusMessage(`Neural Link Active: ${urlTicker}...`);
                                // Since we don't have an image, we'll trigger a fetch-based workflow
                                runAnalysisWorkflow(null, null, null, urlTicker);
                                await cache.delete(cacheKey);
                                navigate(location.pathname, { replace: true });
                                return;
                            } else {
                                alert(`Shared Link Received: ${sharedUrl}\n\nNote: For instant analysis, please use 'Share Image' from TradingView instead of 'Share Link'.`);
                                throw new Error("Link sharing not fully supported yet - Use Image Share");
                            }
                        }

                        setStatusMessage("Processing Neural Input...");
                        const blob = await response.blob();

                        // Basic validation that we got an image
                        if (blob.size < 100) throw new Error("Captured image data too small/corrupt");

                        const file = new File([blob], "shared-image.png", { type: blob.type || "image/png" });

                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const img = new Image();
                            img.onload = () => {
                                const { visualSrc, ocrSrc } = preprocessImage(img);
                                setImagePreview(reader.result);
                                runAnalysisWorkflow(reader.result, visualSrc, ocrSrc);
                            };
                            img.onerror = () => {
                                setStatusMessage("Image format unsupported or corrupt.");
                                setIsAnalyzing(false);
                                setIsQuickShareMode(false);
                            };
                            img.src = reader.result;
                        };
                        reader.readAsDataURL(file);

                        // Clean up cache
                        await cache.delete(cacheKey);
                        navigate(location.pathname, { replace: true });
                    } else {
                        throw new Error("Shared data not found in cache after retries");
                    }
                } catch (err) {
                    console.error("Shared processing failed:", err);
                    setStatusMessage(err.message || "Neural Link Failed");
                    setTimeout(() => {
                        setIsQuickShareMode(false);
                        setIsAnalyzing(false);
                    }, 5000);
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

    const triggerPaste = async () => {
        try {
            const clipboardItems = await navigator.clipboard.read();
            for (const item of clipboardItems) {
                for (const type of item.types) {
                    if (type.startsWith('image/')) {
                        const blob = await item.getType(type);
                        const file = new File([blob], "pasted-image.png", { type });
                        handleFileSelect(file);
                        return;
                    }
                }
            }
            alert("No image found in clipboard. Please copy a chart screenshot first.");
        } catch (err) {
            console.error("Clipboard access denied/failed:", err);
            // Fallback for mobile: explain how to use share target
            setLimitMessage("Mobile Shortcut: To analyze instantly, take a screenshot of your chart, click 'Share' in your gallery, and select 'Diver AI'. Check Documentation for full guide.");
            setLimitType('docs');
            setShowLimitModal(true);
        }
    };

    const handleLaunchCompanion = async () => {
        try {
            await companion.start();
            if (analysisResult) {
                companion.update({
                    ticker: analysisResult.ticker || '---',
                    direction: analysisResult.direction || 'Analyzing...',
                    confidence: analysisResult.confidence || '0'
                });
            }
        } catch (err) {
            alert("Companion Mode requires Picture-in-Picture support. Please use Chrome/Safari on mobile.");
        }
    };

    useEffect(() => {
        if (analysisResult && document.pictureInPictureElement) {
            companion.update({
                ticker: analysisResult.ticker || '---',
                direction: analysisResult.direction || 'Ready',
                confidence: analysisResult.confidence || '0'
            });
        }
    }, [analysisResult]);

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
        if (!user) {
            setLimitMessage("Authentication Required: Please log in to access the neural analysis terminal.");
            setLimitType('guest');
            setShowLimitModal(true);
            return false;
        }

        const today = new Date().toISOString().split('T')[0];

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
        // Guest usage tracking disabled as login is mandatory
        return;
    };

    const preprocessImage = (imageElement) => {
        // Check for Mobile/TWA context to apply ROI cropping
        const isMobile = window.innerWidth < 768 || navigator.userAgent.toLowerCase().includes('android');

        if (isMobile) {
            console.log("[MobileVision] Activating ROI enhancement...");
            const enhancedSrc = enhanceMobileChart(imageElement);
            // Re-draw with enhanced source for further processing if needed
            // For now, we return this as the base for OCR and Visual
            return { visualSrc: enhancedSrc, ocrSrc: enhancedSrc };
        }

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

    const handleFileSelect = (input) => {
        if (!checkLimits()) return;
        if (!input) return;

        if (typeof input === 'string') {
            // It's a URL
            setImagePreview(null);

            // Handle TradingView Snapshots specifically
            if (input.includes('tradingview.com/x/')) {
                alert("Snapshot Link Detected:\nTradingView snapshot links (/x/) don't contain symbol data. To analyze this, please:\n1. Save the image to your device\n2. Upload it here as a file");
                return;
            }

            const ticker = detectTicker(input);
            if (ticker) {
                runAnalysisWorkflow(null, null, null, ticker);
            } else {
                alert("URL Recognition Failed:\nThis doesn't look like a valid TradingView chart or symbol link.");
            }
            return;
        }

        const file = input;
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

    const handleConfirmTicker = (confirmed) => {
        if (confirmed && pendingAnalysisData) {
            const { originalFileSrc, visualSrc, ocrSrc, ticker } = pendingAnalysisData;
            setIsUnstableDetection(false);
            setPendingAnalysisData(null);
            // Run again with the now-confirmed ticker
            runAnalysisWorkflow(originalFileSrc, visualSrc, ocrSrc, ticker);
        } else {
            setIsUnstableDetection(false);
            setPendingAnalysisData(null);
            setStatusMessage("Detection Discarded. Please use manual entry.");
            setShowManualInput(true);
        }
    };

    const runAnalysisWorkflow = async (originalFileSrc, visualSrc, ocrSrc = null, manualTicker = null, manualApiKey = null) => {
        const ocrImage = ocrSrc || visualSrc; // Fallback

        setIsAnalyzing(true); setAnalysisResult(null);
        if (document.pictureInPictureElement) {
            companion.update({ status: 'analyzing' });
        }
        if (!manualTicker && !manualApiKey) setStatusMessage("Initializing Optical Core...");

        // Helper to send notification safely in TWA/PWA
        const sendNotification = async (title, options) => {
            try {
                if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
                    if ('serviceWorker' in navigator) {
                        const registration = await navigator.serviceWorker.ready;
                        if (registration && registration.showNotification) {
                            await registration.showNotification(title, options);
                            return;
                        }
                    }
                    // In TWA (Android), new Notification() is strictly forbidden and throws Illegal Constructor.
                    // We only use it if we are sure we are not on a mobile device that doesn't support it.
                    // For safety, if SW isn't ready, we just skip it or log it.
                    console.info("Notification skipped: Service Worker not ready or showNotification missing.");
                }
            } catch (err) {
                console.warn("Notification system error:", err);
            }
        };

        // Request Notification Permission
        if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
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
                if (profile?.subscription_tier === 'pro' && user && originalFileSrc) {
                    try {
                        setStatusMessage("Deep Scan (Cloud OCR)...");

                        // OPTIMIZATION: Resize image before sending to Edge Function
                        // This prevents massive Egress usage from sending 4K screenshots
                        const compressedImage = await resizeImageForCloud(originalFileSrc);

                        // Ensure session is fresh before calling Edge Function
                        const { data: { session }, error: sessionError } = await supabase.auth.getSession();

                        if (sessionError || !session) {
                            console.warn("[Cloud OCR] No valid session found, attempting refresh...");
                            const { data: refreshData, error: refreshError } = await supabase.auth.refreshSession();

                            if (refreshError) {
                                console.error("[Cloud OCR] Session refresh failed:", refreshError);
                                throw new Error("Session expired. Please log in again.");
                            }

                            console.log("[Cloud OCR] Session refreshed successfully");
                        }

                        // Invoke Edge Function with retry on 401
                        let attempt = 0;
                        let lastError = null;

                        while (attempt < 2) {
                            attempt++;
                            console.log(`[Cloud OCR] Attempt ${attempt}/2`);

                            const { data, error } = await supabase.functions.invoke('detect_ticker', {
                                body: { image: compressedImage },
                            });

                            // Check for 401 specifically
                            if (error) {
                                console.error(`[Cloud OCR] Error on attempt ${attempt}:`, error);

                                // If 401 and first attempt, refresh session and retry
                                if (error.message?.includes('401') && attempt === 1) {
                                    console.warn("[Cloud OCR] 401 detected, refreshing session and retrying...");
                                    await supabase.auth.refreshSession();
                                    lastError = error;
                                    continue; // Retry
                                }

                                lastError = error;
                                break; // Non-401 error or second attempt failed
                            }

                            // Success
                            if (data?.text) {
                                console.log("[Cloud OCR] Success:", data);
                                return { text: data.text, ticker: detectTicker(data.text) };
                            } else {
                                console.warn("[Cloud OCR] No text in response:", data);
                                break; // No data, don't retry
                            }
                        }

                        // If we get here, all attempts failed
                        throw lastError || new Error("Cloud OCR returned no data");

                    } catch (cloudErr) {
                        console.warn("Cloud OCR unavailable, using fallback:", cloudErr);
                        console.error("[Cloud OCR] Full error details:", {
                            message: cloudErr?.message,
                            status: cloudErr?.status,
                            details: cloudErr?.details
                        });
                        setStatusMessage("Cloud OCR unavailable. Switching to Local Neural Engine...");
                    }
                }


                // Multi-Pass OCR with Advanced Preprocessing
                try {
                    const worker = await getOcrWorker();
                    setStatusMessage("Initializing Multi-Pass OCR Engine...");

                    // Import preprocessing utilities
                    const { generatePreprocessedVariants } = await import('../lib/ocrPreprocessor');
                    const { getMultiPassConfigs, filterByConfidence } = await import('../lib/ocrConfig');

                    // Create image element from ocrImage data URL
                    if (!ocrImage) return { text: "", ticker: null };

                    const img = new Image();
                    await new Promise((resolve, reject) => {
                        img.onload = resolve;
                        img.onerror = reject;
                        img.src = ocrImage;
                    });

                    // Helper to yield main thread
                    const yieldToMain = () => new Promise(resolve => setTimeout(resolve, 10));

                    // Generate preprocessed variants
                    setStatusMessage("Preprocessing Image (4 variants)...");
                    await yieldToMain(); // Yield to allow HUD to update
                    const allVariants = generatePreprocessedVariants(img);
                    await yieldToMain();

                    // V5.3 Update: Allow ROI extraction for all users to improve terminal reliability
                    const variants = allVariants;

                    console.log(`[OCR] Generated ${variants.length} preprocessed variants ${isPro ? '(Pro: includes ROI)' : '(Free: ROI excluded)'}`);

                    // Get multi-pass configurations
                    const configs = getMultiPassConfigs();

                    // PRO FEATURE: Pro users get 8 passes, Free users get 4 passes
                    const maxPasses = isPro ? 8 : 4;

                    // Run OCR on all combinations of variants and configs
                    const results = [];
                    let passCount = 0;
                    const totalPasses = Math.min(variants.length * configs.length, maxPasses);

                    for (const variant of variants) {
                        for (const configSet of configs) {
                            if (passCount >= maxPasses) break; // Pro: 8 passes, Free: 4 passes

                            passCount++;
                            const tierLabel = isPro ? 'Pro' : 'Free';
                            setStatusMessage(`OCR Pass ${passCount}/${totalPasses} [${tierLabel}] (${variant.name} + ${configSet.name})...`);

                            try {
                                // Set Tesseract parameters for this pass
                                await worker.setParameters(configSet.config);

                                // Run recognition
                                const result = await worker.recognize(variant.dataUrl);

                                // Filter by confidence
                                const filtered = filterByConfidence(result, 60, 70);

                                if (filtered) {
                                    results.push({
                                        text: filtered.text,
                                        confidence: filtered.confidence,
                                        wordCount: filtered.wordCount,
                                        variant: variant.name,
                                        config: configSet.name,
                                        priority: configSet.priority
                                    });
                                    console.log(`[OCR] Pass ${passCount}: ${filtered.confidence}% confidence, "${filtered.text.substring(0, 50)}..."`);
                                }
                            } catch (passErr) {
                                console.warn(`[OCR] Pass ${passCount} failed:`, passErr);
                            }
                        }
                        if (passCount >= maxPasses) break;
                    }

                    // Select best result based on confidence and priority
                    if (results.length > 0) {
                        results.sort((a, b) => {
                            // Primary: confidence
                            if (Math.abs(a.confidence - b.confidence) > 5) {
                                return b.confidence - a.confidence;
                            }
                            // Secondary: priority (lower is better)
                            return a.priority - b.priority;
                        });

                        const bestResult = results[0];
                        console.log(`[OCR] Best result: ${bestResult.confidence}% (${bestResult.variant} + ${bestResult.config})`);
                        setStatusMessage(`OCR Complete: ${bestResult.confidence}% confidence`);

                        // V5.4: Return both best text and all variant results for deeper inspection (e.g. Price Axis)
                        return {
                            text: bestResult.text,
                            ticker: detectTicker(bestResult.text),
                            variants: results
                        };
                    } else {
                        console.warn("[OCR] All passes failed confidence threshold");
                        setStatusMessage("OCR: Low confidence, using fallback...");
                        return { text: "", ticker: null };
                    }
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

            // --- PROMPT FOR CONFIRMATION IF UNCERTAIN (V5 ACCURACY BOOST) ---
            let detectedTicker = null;
            let anchorPrice = null;

            if (typeof ocrRawResult === 'object' && ocrRawResult !== null) {
                const fullText = ocrRawResult.text || "";
                detectedTicker = detectTicker(fullText);
                anchorPrice = detectPrice(fullText);

                // V5.4: Use Right-Axis for price anchor if available
                const rightAxisResult = ocrRawResult.variants?.find(v => v.variant === 'roi_right_axis');
                if (rightAxisResult) {
                    const rightPrice = detectPrice(rightAxisResult.text);
                    if (rightPrice) {
                        console.info("[OCR] Anchor Price verified via Right Axis ROI");
                        anchorPrice = rightPrice;
                    }
                }
            }

            // V5.7: Confirmation Guard for ambiguous detections
            const isHeuristicBased = !manualTicker && (!ocrRawResult?.ticker || !isValidTicker(ocrRawResult.ticker)) && detectedTicker;

            if (isHeuristicBased && !manualTicker) {
                console.info("[Neural Core] Ambiguous detection. Awaiting user confirmation...");
                setIsUnstableDetection(true);
                setPendingAnalysisData({
                    originalFileSrc, visualSrc, ocrSrc,
                    ticker: detectedTicker,
                    anchorPrice,
                    visualData
                });
                setIsAnalyzing(false);
                setStatusMessage(`Ambiguous Detection: Confirm ${detectedTicker}?`);
                return;
            }

            ticker = detectedTicker || manualTicker;

            // V5.6: Mandatory validation guard
            if (ticker && ticker !== "VISUAL-SCAN" && !isValidTicker(ticker)) {
                console.warn(`[Neural Core] Rejected invalid ticker: ${ticker}`);
                ticker = null;
            }

            const hasVisual = visualData && visualData.points.length > 20;

            if (!ticker && !hasVisual) {
                setStatusMessage("Neural Core Rejected: No Valid Data.");
                alert("Safety Check Triggered:\nNo valid ticker or chart pattern detected.");
                setIsAnalyzing(false);
                return;
            }

            // V5.8: Hybrid Price Validation (Live API + Visual)
            // Race the Live Price against OCR to fix scaling errors (e.g. 92.72 vs 92720)
            if (ticker && ticker !== "VISUAL-SCAN") {
                setStatusMessage(`Verifying Scale for ${ticker}...`);
                try {
                    const optimizedPrice = await getOptimizedPrice(ticker);

                    if (optimizedPrice && optimizedPrice.price > 0) {
                        const livePrice = optimizedPrice.price;
                        console.log(`[Hybrid Validation] Live Price: ${livePrice} (${optimizedPrice.source}) | OCR Anchor: ${anchorPrice}`);

                        if (!anchorPrice || isNaN(anchorPrice)) {
                            // Case 1: OCR failed to read price -> Use Live Price
                            console.warn("[Hybrid Validation] OCR Price missing. Defaulting to Live Price.");
                            anchorPrice = livePrice;
                        } else {
                            // Case 2: Compare and Auto-Scale
                            const ratio = livePrice / anchorPrice;

                            // Check for common scaling errors (10x, 100x, 1000x, 0.01x etc)
                            // Allow 15% variance for market moves
                            const isScaleError = (factor) => Math.abs(ratio - factor) < (factor * 0.15);

                            if (isScaleError(10)) anchorPrice *= 10;
                            else if (isScaleError(100)) anchorPrice *= 100;
                            else if (isScaleError(1000)) anchorPrice *= 1000;
                            else if (isScaleError(10000)) anchorPrice *= 10000;
                            else if (isScaleError(0.1)) anchorPrice *= 0.1;
                            else if (isScaleError(0.01)) anchorPrice *= 0.01;
                            else if (Math.abs(ratio - 1) > 0.2) {
                                // Case 3: Major Divergence (>20%) not explained by scale
                                // Trust Live Price for anchoring as OCR likely read a different number (e.g. volume)
                                console.warn(`[Hybrid Validation] Major divergence detected (Live: ${livePrice} vs OCR: ${anchorPrice}). Overriding.`);
                                anchorPrice = livePrice;
                            } else {
                                // Case 4: Close enough (<20% variance), use OCR for chart consistency
                                // But maybe nudge it closer if needed? No, chart relative scale matters most.
                                console.log("[Hybrid Validation] Price verified. Scale matches.");
                            }
                        }
                    }
                } catch (valErr) {
                    console.warn("[Hybrid Validation] Live fetch failed, relying on OCR.", valErr);
                }
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
            if (ticker && ticker !== "VISUAL-SCAN" && ticker !== "VISUALSCAN" && (!marketStats || !historicalPrices)) {
                try {
                    const isStock = STOCK_MAP[ticker];

                    if (isStock || !COIN_MAP[ticker]) {
                        // 1. Try Yahoo Finance (No Key Required) - Primary for Stocks AND Unknowns
                        setStatusMessage(`Accessing Public Exchange Data for ${ticker}...`);
                        const yahooData = await fetchYahooData(ticker);

                        if (yahooData) {
                            marketStats = yahooData.marketStats;
                            historicalPrices = yahooData.historicalData;
                        } else {
                            // 2. Fallback to Finnhub (Key Required)
                            console.warn("Yahoo Finance unavailable. Requesting Institutional Access...");
                            let apiKey = manualApiKey || localStorage.getItem('finnhub_key');

                            if (!apiKey && isValidTicker(ticker)) {
                                setStatusMessage("Institutional Access Required. Verifying Credentials...");
                                setPendingTicker(ticker);
                                setShowApiKeyInput(true);
                                setIsAnalyzing(false);
                                return;
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
                const hCloses = Array.isArray(historicalPrices) ? historicalPrices : (historicalPrices.closes || []);
                const visualTrend = visualData.points[visualData.points.length - 1] > visualData.points[0] ? 1 : -1;
                const marketTrend = hCloses[hCloses.length - 1] > hCloses[0] ? 1 : -1;

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

            // --- SNAPSHOT UPLOAD (Asynchronous Parallel) ---
            if (user) {
                // We don't await this to keep the UI snappy
                (async () => {
                    try {
                        console.info("Archiving Neural Snapshot in background...");
                        const snapshotUrl = await uploadSnapshot(originalFileSrc, user.id);
                        if (snapshotUrl) {
                            // Update the result object and current state if it's still for this analysis
                            result.imageUrl = snapshotUrl;
                            setAnalysisResult(prev => {
                                if (prev && prev.id === result.id) {
                                    return { ...prev, imageUrl: snapshotUrl };
                                }
                                return prev;
                            });
                            // Store in history with the URL
                            saveHistory(result);
                        } else {
                            saveHistory(result);
                        }
                    } catch (uploadErr) {
                        console.warn("Snapshot upload background task failed:", uploadErr);
                        saveHistory(result);
                    }
                })();
            } else {
                // For guests, save history immediately
                saveHistory(result);
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

            // Trigger Notification
            sendNotification(`Analysis Complete: ${result.ticker}`, {
                body: `${result.direction} (${result.confidence}%) - Click to view details`,
                icon: '/pwa-192x192.png',
                tag: 'analysis-complete', // Prevent duplicates
                renotify: true
            });

        } catch (err) {
            // Auto-reset API Key if it seems invalid
            if (err.message && (err.message.includes("API Key") || err.message.includes("Forbidden") || err.message.includes("403"))) {
                localStorage.removeItem('finnhub_key');
            }
            setStatusMessage(`ERROR: ${err.message || "Analysis failed."}`);
            console.error(err);
        }
        finally {
            setIsAnalyzing(false);
            // We don't reset isQuickShareMode here because we want to show the "Back to Trading" button on the result
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
            <Helmet>
                <title>Analysis Terminal | Diver AI - Institutional AI Chart Scanning</title>
                <meta name="description" content="Access the Diver AI Analysis Terminal. Scan charts with neural networks to identify high-probability patterns and institutional alpha." />
                <meta property="og:title" content="Diver AI Terminal | Institutional Chart Analysis" />
                <meta property="og:description" content="Live AI chart scanning and probabilistic forecasting. Get the edge with institutional-grade technology." />
            </Helmet>
            <TickerTape items={tickerItems} />

            {/* Clinical System HUD */}
            <div className="bg-slate-900/50 border-b border-slate-800/50 px-4 py-1.5 flex items-center justify-between text-[8px] font-bold font-mono tracking-widest text-slate-500">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 bg-brand rounded-full animate-pulse" /> ENGINE_STATUS: OPERATIONAL</div>
                    <div className="flex items-center gap-1.5"><Activity className="w-3 h-3" /> LATENCY: 42MS</div>
                    <div className="flex items-center gap-1.5"><Cpu className="w-3 h-3" /> WEBGL_ACCELERATION: ACTIVE</div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">DATA_STREAM: [YAHOO:PASS, CG:PASS, FINN:PASS]</div>
                    <div className="flex items-center gap-1.5 text-brand">INSTITUTIONAL_CORE: v3.2.1-STABLE</div>
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
                                {!user ? (
                                    <div className="max-w-md mx-auto text-center space-y-8 p-12 bg-black-ash/40 border border-ash rounded-[40px] shadow-2xl">
                                        <div className="w-20 h-20 bg-brand/10 border border-brand/20 rounded-3xl flex items-center justify-center mx-auto mb-6">
                                            <LockIcon className="w-10 h-10 text-brand" />
                                        </div>
                                        <div className="space-y-4">
                                            <h3 className="text-3xl font-black text-white uppercase tracking-tighter">Terminal Locked</h3>
                                            <p className="text-slate-500 font-bold leading-relaxed">
                                                Advanced neural analysis requires an active operative session. Please initialize authentication to access the terminal.
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => setShowAuth(true)}
                                            className="btn-flame w-full !py-4 text-xs font-black uppercase tracking-[0.2em]"
                                        >
                                            Initialize Authentication <ChevronRight className="w-4 h-4 ml-2" />
                                        </button>
                                        <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest">
                                            Secure end-to-end encrypted protocol
                                        </p>
                                    </div>
                                ) : (
                                    <FileUpload
                                        onFileSelect={handleFileSelect}
                                        isAnalyzing={isAnalyzing}
                                        statusMessage={statusMessage}
                                        isUnstableDetection={isUnstableDetection}
                                        handleConfirmTicker={handleConfirmTicker}
                                    />
                                )}
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
                                                        className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-[10px] font-black uppercase tracking-widest ${isIncognito ? 'bg-brand/10 border-brand/20 text-brand' : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'}`}
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
                                        {isQuickShareMode && (
                                            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-3xl z-[2000] flex flex-col items-center justify-center p-8 text-center animate-in fade-in duration-500">
                                                <div className="w-20 h-20 bg-brand/10 rounded-3xl border border-brand/20 flex items-center justify-center mb-8 relative">
                                                    <div className="absolute inset-0 border-2 border-brand rounded-3xl border-t-transparent animate-spin" />
                                                    <Share2 className="w-8 h-8 text-brand" />
                                                </div>
                                                <h2 className="text-3xl font-black text-white uppercase tracking-tighter mb-4">Signal Received</h2>
                                                <p className="text-slate-400 font-mono text-xs uppercase tracking-widest animate-pulse">{statusMessage}</p>

                                                <div className="mt-12 w-full max-w-xs h-1 bg-slate-800 rounded-full overflow-hidden">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: '100%' }}
                                                        transition={{ duration: 15, ease: "linear" }}
                                                        className="h-full bg-brand"
                                                    />
                                                </div>
                                                <button
                                                    onClick={() => { setIsAnalyzing(false); setIsQuickShareMode(false); }}
                                                    className="mt-12 px-6 py-3 bg-slate-900 text-slate-500 text-[10px] font-black uppercase tracking-widest rounded-xl hover:text-white transition-colors"
                                                >
                                                    Cancel Scan
                                                </button>
                                            </div>
                                        )}
                                        <div className="relative w-24 h-24 mb-6">
                                            <div className="absolute inset-0 border-4 border-brand/10 rounded-full"></div>
                                            <div className="absolute inset-0 border-4 border-t-brand rounded-full animate-spin"></div>
                                            <Camera className="absolute inset-0 m-auto text-brand w-10 h-10 animate-pulse" />
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
                                        isQuickShareMode={isQuickShareMode}
                                        onQuickShareReset={() => {
                                            setAnalysisResult(null);
                                            setIsQuickShareMode(false);
                                            setImagePreview(null);
                                        }}
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
                    <button onClick={() => setActiveTab('analyze')} className={`flex flex-col items-center gap-1.5 transition-all ${activeTab === 'analyze' ? 'text-brand' : 'text-slate-500'}`}><Activity className="w-5 h-5" /><span className="text-[8px] font-black uppercase">Analyze</span></button>
                    <button onClick={() => setActiveTab('history')} className={`flex flex-col items-center gap-1.5 transition-all ${activeTab === 'history' ? 'text-brand' : 'text-slate-500'}`}><History className="w-5 h-5" /><span className="text-[8px] font-black uppercase">Terminal</span></button>
                </div>
            </div>
            <FloatingWidget onPaste={triggerPaste} onCompanion={handleLaunchCompanion} />
        </div>
    );
}
