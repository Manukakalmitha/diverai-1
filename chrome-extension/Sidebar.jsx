
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, Search, AlertTriangle, CheckCircle2, Lock as LockIcon, User, LogOut, ChevronLeft, Cpu, TrendingUp, TrendingDown, Minus, Clock, ShieldCheck, Key, History, Trash2, BarChart3, Fingerprint, Share2, Trophy, Copy, Layers, Pencil } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAppContext } from '../src/context/AppContext';
import { runRealAnalysis } from '../src/lib/analysis';
import { detectTicker, detectPrice, fetchTickerData, fetchMarketData, fetchHistoricalData, fetchYahooData, fetchStockData, fetchStockHistory, STOCK_MAP } from '../src/lib/marketData';

const PROD_URL = "https://diverai.flisoft.agency";

// --- Shared Components for Terminal Aesthetic ---
const SentimentGauge = ({ probability, direction }) => {
    const percentage = probability * 100;
    const isBull = direction.includes('Bullish');

    return (
        <div className="w-full space-y-1.5 py-2">
            <div className="flex justify-between items-end">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                    <Fingerprint className="w-3 h-3" />
                    Confidence
                </span>
                <span className={`text-xs font-mono font-bold ${isBull ? 'text-emerald-500' : 'text-rose-500'}`}>
                    {percentage.toFixed(1)}%
                </span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 1, ease: "circOut" }}
                    className={`h-full ${isBull ? 'bg-emerald-500' : 'bg-rose-500'}`}
                />
            </div>
        </div>
    );
};

const OrderBookDepth = ({ ticker }) => {
    const [depth, setDepth] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!ticker) return;
        fetchDepth();
    }, [ticker]);

    const fetchDepth = async () => {
        setLoading(true);
        try {
            // Simplified order book fetch (Binance for Crypto, fallback for others)
            let symbol = ticker.replace('/', '').toUpperCase();
            if (symbol === 'BTC' || symbol === 'ETH' || symbol === 'SOL' || symbol === 'XRP') symbol += 'USDT';
            const res = await fetch(`https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=5`);
            if (res.ok) {
                const data = await res.json();
                setDepth(data);
            }
        } catch (err) {
            console.warn("Order book fetch skipped for non-compatible ticker.");
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="h-20 bg-slate-900 animate-pulse rounded-xl" />;
    if (!depth) return null;

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 space-y-2">
            <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase">
                <Layers className="w-3 h-3" /> Order Book Depth
            </div>
            <div className="space-y-1">
                {depth.asks.slice(0, 3).reverse().map(([price, qty], i) => (
                    <div key={`ask-${i}`} className="flex justify-between text-[10px] font-mono">
                        <span className="text-rose-500">{Number(price).toFixed(2)}</span>
                        <span className="text-slate-600">{Number(qty).toFixed(4)}</span>
                    </div>
                ))}
                <div className="h-px bg-slate-800 my-1" />
                {depth.bids.slice(0, 3).map(([price, qty], i) => (
                    <div key={`bid-${i}`} className="flex justify-between text-[10px] font-mono">
                        <span className="text-emerald-500">{Number(price).toFixed(2)}</span>
                        <span className="text-slate-600">{Number(qty).toFixed(4)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const Sidebar = () => {
    const { user, profile, refreshProfile, neuralState } = useAppContext();
    const [status, setStatus] = useState('idle'); // idle, scanning, analyzing, success, error
    const [analysisResult, setAnalysisResult] = useState(null);
    const [screenshot, setScreenshot] = useState(null);
    const [statusMessage, setStatusMessage] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const [showLimitModal, setShowLimitModal] = useState(false);
    const [limitMessage, setLimitMessage] = useState("");
    const [limitType, setLimitType] = useState('guest'); // guest, verify, free
    const [userIp, setUserIp] = useState(null);
    const [history, setHistory] = useState([]);
    const [activeTab, setActiveTab] = useState('analyze'); // analyze, history, referral
    const [referralCopied, setReferralCopied] = useState(false);

    useEffect(() => {
        fetchIp();

        // Listen for external auth synchronization
        const handleExternalMessage = async (request) => {
            if (request.type === 'AUTH_SYNC' && request.session) {
                const { access_token, refresh_token } = request.session;
                await supabase.auth.setSession({ access_token, refresh_token });
                refreshProfile();
            }
        };

        if (chrome.runtime.onMessageExternal) {
            chrome.runtime.onMessageExternal.addListener(handleExternalMessage);
        }

        return () => {
            if (chrome.runtime.onMessageExternal) {
                chrome.runtime.onMessageExternal.removeListener(handleExternalMessage);
            }
        };
    }, []);

    const fetchIp = async () => {
        try {
            const res = await fetch('https://api.ipify.org?format=json');
            if (!res.ok) throw new Error("Network response was not ok");
            const data = await res.json();
            setUserIp(data.ip);
        } catch (err) {
            console.info("IP tracking disabled/blocked. Using session fallback.");
            setUserIp('unknown-client');
        }
    };

    useEffect(() => {
        fetchHistory();
    }, [user]);

    const fetchHistory = async () => {
        if (!user) {
            const local = localStorage.getItem('diver_ai_guest_history');
            setHistory(local ? JSON.parse(local) : []);
            return;
        }
        const { data, error } = await supabase.from('prediction_history').select('*').eq('user_id', user.id).order('created_at', { ascending: false }).limit(10);
        if (!error && data) setHistory(data.map(row => ({ ...row.data, db_id: row.id, created_at: row.created_at })));
    };

    const saveHistory = async (result) => {
        if (user) {
            const { data } = await supabase.from('prediction_history').insert([{ user_id: user.id, data: result }]).select();
            if (data?.[0]) {
                const newItem = { ...result, db_id: data[0].id, created_at: data[0].created_at };
                setHistory(prev => [newItem, ...prev].slice(0, 10));
            }
        }
    };

    const updateGuestUsage = () => {
        // Guest usage tracking disabled as login is mandatory
        return;
    };

    const checkLimits = () => {
        if (!user) {
            setLimitMessage("Authentication Required: Please log in to access the neural analysis terminal.");
            setLimitType('guest');
            setShowLimitModal(true);
            return false;
        }

        const today = new Date().toISOString().split('T')[0];
        if (!user.email_confirmed_at) {
            setLimitMessage("Security Protocol: Email verification required. Check your inbox to unlock terminal scanning.");
            setLimitType('verify');
            setShowLimitModal(true);
            return false;
        }
        if (profile && profile.subscription_tier !== 'pro') {
            const todayCount = profile.upload_count || 0;
            if (todayCount >= 3 && profile.last_upload_date === today) {
                setLimitMessage("Neural Capacity Reached: 3/day. Upgrade to Pro for unlimited terminal access.");
                setLimitType('free');
                setShowLimitModal(true);
                return false;
            }
        }
        return true;
    };

    const handleScan = async () => {
        if (!checkLimits()) return;
        setStatus('scanning');
        setStatusMessage("Capturing Visual Stream...");
        setAnalysisResult(null);

        try {
            const msgResponse = await new Promise((resolve) => {
                chrome.runtime.sendMessage({ action: 'CAPTURE_SCREENSHOT' }, (response) => {
                    if (chrome.runtime.lastError) {
                        resolve({ error: "Connection to extension background lost. Please refresh the page." });
                    } else {
                        resolve(response);
                    }
                });
            });

            if (msgResponse.error) throw new Error(msgResponse.error);
            const imgUrl = msgResponse.dataUrl;
            setScreenshot(imgUrl);

            setStatus('analyzing');
            setStatusMessage("Deep Scan (Cloud OCR)...");

            const { data: { session } } = await supabase.auth.getSession();
            let currentToken = session?.access_token || import.meta.env.VITE_SUPABASE_ANON_KEY;
            let retryCount = 0;
            let response;
            let success = false;

            while (!success && retryCount < 3) {
                try {
                    const activeToken = retryCount === 2 ? import.meta.env.VITE_SUPABASE_ANON_KEY : currentToken;
                    response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/detect_ticker`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'apikey': import.meta.env.VITE_SUPABASE_ANON_KEY,
                            'Authorization': `Bearer ${activeToken}`
                        },
                        body: JSON.stringify({ image: imgUrl })
                    });

                    if (response.ok) {
                        success = true;
                    } else if (response.status === 401 && retryCount === 0 && session) {
                        console.info("[Sidebar] Auth 401. Attempting session refresh...");
                        try {
                            const { data: { session: newSession }, error: refreshError } = await supabase.auth.refreshSession();
                            if (refreshError || !newSession) {
                                console.warn("[Sidebar] Refresh failed, switching to Anon-Key fallback.");
                                retryCount = 2; // Jump to anon
                            } else {
                                currentToken = newSession.access_token;
                                retryCount++;
                            }
                        } catch (err) {
                            retryCount = 2;
                        }
                    } else if (response.status === 401 && retryCount < 2) {
                        console.info("[Sidebar] Auth 401 (No Session/Expired). Using Anon-Key fallback.");
                        retryCount = 2;
                        // Small delay to prevent tight loop
                        await new Promise(r => setTimeout(r, 500));
                    } else {
                        throw new Error(`Service Error (${response.status})`);
                    }
                } catch (err) {
                    if (retryCount >= 2) throw err;
                    retryCount++;
                }
            }

            if (!response || !response.ok) throw new Error("Visualization Service Unavailable");
            const ocrData = await response.json();
            const fullText = ocrData?.text || '';
            const ticker = detectTicker(fullText);
            const anchorPrice = detectPrice(fullText);
            console.log("[Sidebar] OCR Detected:", { ticker, anchorPrice });

            if (!ticker && !anchorPrice) throw new Error("Neural Core Rejected: No valid asset or price identified.");

            setStatusMessage(`Target Locked: ${ticker}. Syncing Data...`);
            let marketStats, historicalPrices;
            const isStock = STOCK_MAP[ticker];

            if (isStock) {
                const yahooData = await fetchYahooData(ticker);
                if (yahooData) {
                    marketStats = yahooData.marketStats;
                    historicalPrices = yahooData.historicalPrices;
                } else {
                    throw new Error("Institutional data access error for stocks.");
                }
            } else {
                marketStats = await fetchMarketData(ticker);
                historicalPrices = await fetchHistoricalData(ticker, 90);
            }

            setStatusMessage("Synchronizing Global Intelligence...");
            const result = await runRealAnalysis(ticker, marketStats, historicalPrices, user, neuralState, setStatusMessage, 0.95, true);

            if (user && profile) {
                const today = new Date().toISOString().split('T')[0];
                const newCount = (profile.last_upload_date !== today) ? 1 : (profile.upload_count || 0) + 1;
                await supabase.from('profiles').update({ upload_count: newCount, last_upload_date: today }).eq('id', user.id);
                refreshProfile();
            } else {
                updateGuestUsage();
            }

            await saveHistory(result);
            setAnalysisResult(result);
            setStatus('success');
        } catch (err) {
            console.error(err);
            setErrorMessage(err.message);
            setStatus('error');
        }
    };

    const handleDrawOverlay = () => {
        if (!analysisResult) {
            console.warn('[Overlay] No analysis result available to draw overlay.');
            return;
        }

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];

            if (!activeTab) {
                console.warn('[Overlay] No active tab found.');
                return;
            }

            if (!activeTab.url) {
                console.warn('[Overlay] Unable to determine current page URL.');
                return;
            }

            // Check if on a supported trading platform
            const supportedDomains = ['tradingview.com', 'yahoo.com', 'coingecko.com', 'coinmarketcap.com', 'binance.com', 'google.com/finance'];
            const isSupported = supportedDomains.some(domain => activeTab.url.includes(domain));

            if (!isSupported) {
                console.info(`[Overlay] Skipped - Not on a supported trading page. Current URL: ${activeTab.url}`);
                alert('Please navigate to a trading chart (e.g., TradingView) to draw the R/R overlay.');
                return;
            }

            // Send message to content script
            chrome.tabs.sendMessage(activeTab.id, {
                action: 'DRAW_RR_OVERLAY',
                targets: analysisResult.targets,
                ticker: analysisResult.ticker
            }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error('[Overlay] Failed:', chrome.runtime.lastError.message);
                    alert(`Overlay failed: Content script not ready. Try refreshing the trading page.`);
                } else if (response?.success) {
                    console.log('[Overlay] Successfully drawn on chart.');
                } else {
                    console.warn('[Overlay] Message sent, but no confirmation received.');
                }
            });
        });
    };

    const TabButton = ({ id, label, icon: Icon }) => (
        <button
            onClick={() => setActiveTab(id)}
            className={`flex-1 flex items-center justify-center gap-2 py-2 text-[11px] font-bold uppercase tracking-wider transition-all relative ${activeTab === id ? 'text-white' : 'text-slate-500 hover:text-slate-400'}`}
        >
            <Icon className="w-3.5 h-3.5" />
            {label}
            {activeTab === id && (
                <motion.div
                    layoutId="activeTab"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500"
                />
            )}
        </button>
    );

    return (
        <div className="h-screen flex flex-col bg-slate-950 font-sans text-white overflow-hidden selection:bg-blue-500/30">
            {/* Limit Modal */}
            {showLimitModal && (
                <div className="fixed inset-0 z-[100] bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-6 text-center animate-in fade-in">
                    <div className="space-y-6 max-w-xs">
                        <div className="w-16 h-16 bg-rose-500/10 rounded-2xl flex items-center justify-center mx-auto border border-rose-500/20"><LockIcon className="text-rose-500" /></div>
                        <h3 className="text-xl font-black uppercase tracking-tight">Access Restricted</h3>
                        <p className="text-slate-500 text-xs font-bold leading-relaxed">{limitMessage}</p>
                        <div className="space-y-3">
                            {limitType === 'free' ? (
                                <button onClick={() => window.open(`${PROD_URL}/pricing?source=extension&extId=${chrome.runtime.id}`, '_blank')} className="w-full py-3 bg-emerald-500 text-slate-950 font-black rounded-xl uppercase tracking-widest text-[10px]">Upgrade to Pro</button>
                            ) : limitType === 'guest' ? (
                                <button onClick={() => window.open(`${PROD_URL}/login?source=extension&extId=${chrome.runtime.id}`, '_blank')} className="w-full py-3 bg-emerald-500 text-slate-950 font-black rounded-xl uppercase tracking-widest text-[10px]">Initialize Login</button>
                            ) : (
                                <button onClick={() => setShowLimitModal(false)} className="w-full py-3 bg-slate-800 text-white font-black rounded-xl uppercase tracking-widest text-[10px]">Check Inbox</button>
                            )}
                            <button onClick={() => setShowLimitModal(false)} className="w-full py-2 text-slate-500 font-bold uppercase tracking-widest text-[9px]">Close</button>
                        </div>
                    </div>
                </div>
            )}

            {/* 1. Header & Navigation */}
            <div className="shrink-0 bg-slate-950 border-b border-slate-800/50">
                <div className="px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="bg-blue-600 rounded-md p-1">
                            <Activity className="w-3.5 h-3.5 text-white" />
                        </div>
                        <span className="font-bold text-sm tracking-tight">Diver<span className="text-blue-500">AI</span></span>
                    </div>
                    {user ? (
                        <div className="flex items-center gap-3">
                            <div className="text-right">
                                <p className="text-[9px] font-bold text-slate-500 uppercase">{profile?.subscription_tier === 'pro' ? 'PRO' : 'BASIC'}</p>
                            </div>
                            <button onClick={() => supabase.auth.signOut()} className="text-slate-500 hover:text-white transition-colors"><LogOut className="w-4 h-4" /></button>
                        </div>
                    ) : (
                        <button onClick={() => window.open(`${PROD_URL}/login?source=extension&extId=${chrome.runtime.id}`, '_blank')} className="text-[10px] font-bold text-blue-400 hover:text-blue-300 uppercase tracking-wide">Login</button>
                    )}
                </div>

                {/* Segmented Control */}
                <div className="flex px-2 border-t border-slate-900">
                    <TabButton id="analyze" label="Analysis" icon={Zap} />
                    <TabButton id="history" label="History" icon={History} />
                    {user && <TabButton id="referral" label="Rewards" icon={Trophy} />}
                </div>
            </div>

            {/* 2. Main Scrollable Content */}
            <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent p-4">
                <AnimatePresence mode="wait">
                    {/* --- HISTORY TAB --- */}
                    {activeTab === 'history' && (
                        <motion.div
                            key="history"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="space-y-3"
                        >
                            {history.length === 0 ? (
                                <div className="text-center py-12 text-slate-600">
                                    <History className="w-10 h-10 mx-auto mb-3 opacity-50" />
                                    <p className="text-xs font-medium">No past analyses found.</p>
                                </div>
                            ) : (
                                history.map((item, i) => (
                                    <div
                                        key={item.db_id || i}
                                        onClick={() => { setAnalysisResult(item); setStatus('success'); setActiveTab('analyze'); }}
                                        className="bg-slate-900/50 hover:bg-slate-900 border border-slate-800 rounded-xl p-3 cursor-pointer transition-all flex items-center justify-between group"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${item.direction.includes('Bullish') ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'}`}>
                                                {item.direction.includes('Bullish') ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                            </div>
                                            <div>
                                                <h4 className="text-sm font-bold text-white group-hover:text-blue-400 transition-colors">{item.ticker}</h4>
                                                <p className="text-[10px] text-slate-500 uppercase tracking-wider">{new Date(item.created_at).toLocaleDateString()}</p>
                                            </div>
                                        </div>
                                        <span className={`text-xs font-mono font-bold ${item.direction.includes('Bullish') ? 'text-emerald-500' : 'text-rose-500'}`}>
                                            {(item.finalProb * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                ))
                            )}
                        </motion.div>
                    )}

                    {/* --- REFERRAL TAB --- */}
                    {activeTab === 'referral' && user && (
                        <motion.div
                            key="referral"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="space-y-4"
                        >
                            <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl p-5 border border-white/10 shadow-xl relative overflow-hidden">
                                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
                                <div className="relative z-10 space-y-3">
                                    <h3 className="text-sm font-black uppercase tracking-tight">Earn Pro Status</h3>
                                    <p className="text-[10px] text-blue-100 font-medium leading-relaxed">Refer a friend. You both get 30 days of Pro analysis upon their signup.</p>

                                    <div className="pt-2">
                                        <div className="flex gap-2">
                                            <div className="flex-1 bg-slate-950/40 border border-white/10 rounded-lg px-3 py-2 text-[9px] font-mono text-blue-200 truncate">
                                                {profile?.referral_code ? `${PROD_URL}/signup?ref=${profile.referral_code}` : 'Generating...'}
                                            </div>
                                            <button
                                                onClick={() => {
                                                    navigator.clipboard.writeText(`${PROD_URL}/signup?ref=${profile?.referral_code}`);
                                                    setReferralCopied(true);
                                                    setTimeout(() => setReferralCopied(false), 2000);
                                                }}
                                                className="bg-white text-slate-950 p-2 rounded-lg hover:bg-blue-100 transition-colors"
                                            >
                                                {referralCopied ? <CheckCircle2 className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[8px] font-bold text-slate-500 uppercase mb-1">Total Referrals</p>
                                    <p className="text-xl font-black text-white">{profile?.referral_count || 0}</p>
                                </div>
                                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[8px] font-bold text-slate-500 uppercase mb-1">Days Granted</p>
                                    <p className="text-xl font-black text-emerald-400">{(profile?.referral_count || 0) * 30}</p>
                                </div>
                            </div>

                            <button
                                onClick={() => window.open(`${PROD_URL}/referral`, '_blank')}
                                className="w-full py-3 bg-slate-900 border border-slate-800 hover:border-blue-500/50 text-slate-300 hover:text-white text-[10px] font-bold uppercase tracking-widest rounded-xl transition-all flex items-center justify-center gap-2"
                            >
                                <Share2 className="w-3.5 h-3.5" /> View Full Dashboard
                            </button>
                        </motion.div>
                    )}

                    {/* --- ANALYZE TAB: IDLE --- */}
                    {activeTab === 'analyze' && status === 'idle' && (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.98 }}
                            className="flex flex-col items-center justify-center h-full text-center space-y-6 pt-12"
                        >
                            {!user ? (
                                <>
                                    <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center border border-slate-800 shadow-xl">
                                        <LockIcon className="w-8 h-8 text-rose-500" />
                                    </div>
                                    <div className="space-y-2 max-w-[200px]">
                                        <h3 className="text-lg font-bold text-white">Terminal Locked</h3>
                                        <p className="text-xs text-slate-500 leading-relaxed">
                                            Neural analysis requires an active operative session. Please login to continue.
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => window.open(`${PROD_URL}/login?source=extension&extId=${chrome.runtime.id}`, '_blank')}
                                        className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow-lg transition-all uppercase tracking-widest text-[10px]"
                                    >
                                        Initialize Login
                                    </button>
                                </>
                            ) : (
                                <>
                                    <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center border border-slate-800 shadow-xl">
                                        <Search className="w-8 h-8 text-blue-500" />
                                    </div>
                                    <div className="space-y-2 max-w-[200px]">
                                        <h3 className="text-lg font-bold text-white">Ready to Analyze</h3>
                                        <p className="text-xs text-slate-500 leading-relaxed">
                                            Navigate to any chart (TradingView, Yahoo, etc.) and click scan below.
                                        </p>
                                    </div>
                                </>
                            )}
                        </motion.div>
                    )}

                    {/* --- ANALYZE TAB: LOADING --- */}
                    {(status === 'scanning' || status === 'analyzing') && (
                        <motion.div
                            key="loading"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center justify-center h-full space-y-8 pt-12"
                        >
                            <div className="relative w-16 h-16">
                                <div className="absolute inset-0 border-4 border-slate-800 rounded-full"></div>
                                <div className="absolute inset-0 border-4 border-t-blue-500 rounded-full animate-spin"></div>
                            </div>
                            <div className="text-center space-y-2">
                                <p className="text-xs font-bold uppercase tracking-widest text-blue-400 animate-pulse">{statusMessage}</p>
                                <p className="text-[10px] text-slate-600">Please wait while we process the data.</p>
                            </div>
                        </motion.div>
                    )}

                    {/* --- ANALYZE TAB: RESULT --- */}
                    {status === 'success' && analysisResult && activeTab === 'analyze' && (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-4 pb-20"
                        >
                            {/* Main Ticker Card */}
                            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden">
                                <div className="flex justify-between items-start relative z-10">
                                    <div>
                                        <h2 className="text-3xl font-black text-white tracking-tight">{analysisResult.ticker}</h2>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${analysisResult.direction.includes('Bullish') ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'}`}>
                                                {analysisResult.direction.split(' ')[0]} Signal
                                            </span>
                                            <span className="text-[10px] text-slate-500 font-mono">{(analysisResult.finalProb * 100).toFixed(1)}% Conf.</span>
                                        </div>
                                    </div>
                                    <div className={`p-2 rounded-xl ${analysisResult.direction.includes('Bullish') ? 'bg-emerald-500/10' : 'bg-rose-500/10'}`}>
                                        {analysisResult.direction.includes('Bullish') ? <TrendingUp className="w-5 h-5 text-emerald-500" /> : <TrendingDown className="w-5 h-5 text-rose-500" />}
                                    </div>
                                </div>
                                {/* Subtle Background Gradient */}
                                <div className={`absolute top-0 right-0 w-32 h-32 bg-${analysisResult.direction.includes('Bullish') ? 'emerald' : 'rose'}-500/10 blur-3xl -translate-y-1/2 translate-x-1/2 rounded-full`} />
                            </div>

                            {/* Targets Cards */}
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Entry Zone</p>
                                    <p className="text-lg font-mono font-bold text-white">{analysisResult.targets.entry}</p>
                                </div>
                                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Risk/Reward</p>
                                    <p className="text-lg font-mono font-bold text-blue-400">1:{analysisResult.targets.rr}</p>
                                </div>
                                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Take Profit</p>
                                    <p className="text-lg font-mono font-bold text-emerald-500">{analysisResult.targets.tp1}</p>
                                </div>
                                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Stop Loss</p>
                                    <p className="text-lg font-mono font-bold text-rose-500">{analysisResult.targets.sl}</p>
                                </div>
                            </div>

                            {/* Summary Text */}
                            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-4">
                                <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                                    <Cpu className="w-3 h-3" /> AI Analysis
                                </h4>
                                <p className="text-xs text-slate-300 leading-relaxed font-medium" dangerouslySetInnerHTML={{ __html: analysisResult.overview.replace(/\*\*(.*?)\*\*/g, '<span class="text-white font-bold">$1</span>') }} />
                            </div>

                            {/* Order Book Depth */}
                            <OrderBookDepth ticker={analysisResult.ticker} />

                            {/* Overlay Controls */}
                            <button
                                onClick={handleDrawOverlay}
                                className="w-full py-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl flex items-center justify-center gap-2 text-[10px] font-bold uppercase tracking-widest transition-all"
                            >
                                <Pencil className="w-3.5 h-3.5 text-blue-400" /> Draw R/R on Chart
                            </button>

                            {/* Footer Note */}
                            <div className="pt-2 border-t border-slate-800/50 mt-4 text-center">
                                <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">
                                    Logic: {analysisResult.version.split(' ')[0]} • Data: {analysisResult.macroTrend?.source || "Institutional Feed"}
                                </p>
                            </div>

                            <button onClick={() => setStatus('idle')} className="w-full py-3 text-xs font-bold text-slate-500 hover:text-white transition-colors uppercase tracking-widest">
                                Start New Analysis
                            </button>
                        </motion.div>
                    )}

                    {/* Error State */}
                    {status === 'error' && (
                        <motion.div
                            key="error"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex flex-col items-center justify-center h-full text-center px-6"
                        >
                            <div className="w-12 h-12 bg-rose-500/10 rounded-full flex items-center justify-center mb-4">
                                <AlertTriangle className="w-6 h-6 text-rose-500" />
                            </div>
                            <h3 className="text-sm font-bold text-white mb-2">Analysis Failed</h3>
                            <p className="text-xs text-slate-500 mb-6">{errorMessage}</p>
                            <button onClick={() => setStatus('idle')} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold rounded-lg transition-colors">
                                Try Again
                            </button>
                        </motion.div>
                    )}

                </AnimatePresence>
            </div>

            {/* 3. Fixed Bottom Action (Only on Analyze Tab & Idle/Success) */}
            {activeTab === 'analyze' && status === 'idle' && (
                <div className="shrink-0 p-4 bg-slate-950 border-t border-slate-800/50">
                    <button
                        onClick={handleScan}
                        className="w-full py-3.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 active:scale-[0.98] transition-all flex items-center justify-center gap-2 text-xs uppercase tracking-wider"
                    >
                        <Zap className="w-4 h-4 fill-current" />
                        Execute Scan
                    </button>
                </div>
            )}
        </div>
    );
};

export default Sidebar;

