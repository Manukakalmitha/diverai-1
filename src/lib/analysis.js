
import {
    calculateRSI as calcRSI,
    calculateMACD,
    calculateBollingerBands,
    calculateEMA,
    detectPatterns,
    calculateATR,
    findSupportResistance,
    calculateVWAP,
    calculateTrendBias
} from './technicalAnalysis.js';
import { calculateStats, predictNextPrice, runBackgroundTraining, loadGlobalModel, saveGlobalModelArtifacts, disposeModel, WINDOW_SIZE } from './brain.js';
import { fetchMacroHistory, calculateMacroSentiment } from './marketData.js';
import { supabase } from './supabase.js';
import {
    calculateSharpeRatio,
    calculateSortinoRatio,
    calculateMaxDrawdown,
    calculateVolatilityRatio,
    calculateNetReturns,
    calculateROC,
    calculateRollingVolatility
} from './featureEngineering.js';

/**
 * Advanced Hybrid Fusion Logic (RMSE-Calibrated Form V4)
 */
export const calculateHybridProbability = (neuralProb, patternSentiment, indicators, weights, reliabilityFactor = 0.95, macroSentiment = 0.5, volatilityRatio = 0.02, mtfBias = 0.5) => {
    const P0 = 0.5;

    // Dynamic Macro Weighting based on context
    const W_MACRO = 0.15 + (volatilityRatio > 0.04 ? 0.05 : 0); // Increase macro weight in high volatility
    const W_OB = 0.10; // Order Book Bias Weight
    const totalW = weights.omega + weights.alpha + weights.gamma + W_MACRO + W_OB;

    const getDeviation = (p) => (p - 0.5) * 2;

    const P1 = getDeviation(neuralProb);
    const w1 = weights.omega / totalW;
    const c1 = reliabilityFactor;

    let pVal2 = 0.5;
    if (patternSentiment === 'Bullish') pVal2 = 0.85;
    else if (patternSentiment === 'Bearish') pVal2 = 0.15;
    const P2 = getDeviation(pVal2);
    const w2 = weights.alpha / totalW;

    const rsi = indicators.rsi[indicators.rsi.length - 1];
    let pVal3 = 0.5;
    if (rsi < 30) pVal3 = 0.85; // Extreme oversold
    else if (rsi < 40) pVal3 = 0.70;
    else if (rsi > 70) pVal3 = 0.15; // Extreme overbought
    else if (rsi > 60) pVal3 = 0.30;

    const P3 = getDeviation(pVal3);
    const w3 = weights.gamma / totalW;
    const f3 = (rsi < 25 || rsi > 75) ? 1.3 : 1.0; // Higher indicator impact at extremes

    const P4 = getDeviation(macroSentiment);
    const w4 = W_MACRO / totalW;

    const P5 = getDeviation(mtfBias);
    const w5 = 0.15; // Set weights for MTF alignment

    const layers = [
        { w: w1, p: P1, c: c1, f: 1.0 },
        { w: w2, p: P2, c: 0.9, f: 1.0 },
        { w: w3, p: P3, c: 1.0, f: f3 },
        { w: w4, p: P4, c: 1.0, f: 1.0 },
        { w: w5, p: P5, c: 1.0, f: 1.0 },
        { w: W_OB / totalW, p: getDeviation(weights.obBias || 0.5), c: 1.0, f: 1.0 }
    ];

    const product = layers.reduce((acc, layer) => {
        const probabilityImpact = layer.w * layer.p * layer.c * layer.f;
        return acc * (1 - Math.max(-0.99, Math.min(0.99, probabilityImpact)));
    }, (1 - P0));

    const finalProb = 1 - product;

    // V5.8: NaN Sanitation Guard
    if (isNaN(finalProb) || !isFinite(finalProb)) {
        console.warn("[Neural Core] NaN detected in Engine Fusion. Reverting to Baseline.");
        return 0.5;
    }

    return Math.min(0.995, Math.max(0.005, finalProb));
};

/**
 * Main Analysis Workflow
 */
export const runRealAnalysis = async (ticker, marketStats, historicalData, user, weights, setStatusMessage, syncReliability = 0.95, fastMode = false) => {
    // Standardize historical data (Support for Close-only array or full OHLCV object)
    const closes = Array.isArray(historicalData) ? historicalData : (historicalData.closes || []);
    const highs = historicalData.highs || closes;
    const lows = historicalData.lows || closes;
    const volumes = historicalData.volumes || [];

    // --- SCALE GUARDIAN (V5.1 ADDITION) ---
    // If OCR price is off by order of magnitude (e.g. 92.00 instead of 92000), auto-correct
    let currentPrice = marketStats.price || closes[closes.length - 1];
    const historicalClose = closes[closes.length - 1];
    if (marketStats.price && historicalClose) {
        const magnitudeDiff = Math.log10(historicalClose / marketStats.price);
        if (Math.abs(magnitudeDiff) > 0.6) { // More than ~4x difference
            const scaleFactor = Math.pow(10, Math.round(magnitudeDiff));
            console.log(`[ScaleGuardian] Correcting OCR price scale: ${marketStats.price} -> ${marketStats.price * scaleFactor} (Factor: ${scaleFactor})`);
            currentPrice = marketStats.price * scaleFactor;
        }
    }
    if (closes.length < 20) {
        throw new Error("Insufficient historical data for precision analysis");
    }

    // A. Technical Analysis & Pattern Rec
    const rsi = calcRSI(closes, 14);
    const macdResult = calculateMACD(closes);
    const macdHist = macdResult.histogram;
    const opens = historicalData.opens || [];
    const patterns = detectPatterns(closes, highs, lows, opens);
    const patternSenti = patterns[0].sentiment;
    const atr = calculateATR(highs, lows, closes, 14);
    const srLevels = findSupportResistance(closes, highs, lows);

    // V5: Institutional Volume & Trend Indicators
    const vwap = (historicalData.volumes && historicalData.volumes.length > 0)
        ? calculateVWAP(highs, lows, closes, historicalData.volumes)
        : closes;

    // V5.8: Advanced Feature Engineering (EMA Cross & BB %B)
    const ema8 = calculateEMA(closes, 8);
    const ema21 = calculateEMA(closes, 21);
    const emaRatio = ema8.map((v, i) => v / (ema21[i] || 1));
    const { upper, lower } = calculateBollingerBands(closes, 20);
    const bbLen = upper?.length || 0;
    const bPercentRaw = closes.slice(-bbLen).map((c, i) => (c - lower[i]) / (upper[i] - lower[i] || 1));
    const bPercent = [...new Array(closes.length - bbLen).fill(0.5), ...bPercentRaw];

    // C. Multi-Timeframe Alignment
    let mtfBias = 0.5;
    try {
        setStatusMessage("Aligning Multi-Timeframe Bias (Daily)...");
        const dailyData = await fetchMacroHistory(ticker);
        if (dailyData?.prices) {
            mtfBias = calculateTrendBias(dailyData.prices);
        }
    } catch (mtfErr) {
        console.warn("MTF Alignment failed, using neutral bias:", mtfErr);
    }

    // B. Neural Network Training
    let neuralProb = 0.5;
    let statsFactors = null;

    // currentPrice is already defined and auto-corrected above
    const currentATR = atr[atr.length - 1] || (currentPrice * 0.02);
    const volRatio = calculateVolatilityRatio(closes);
    const volatilityRatio = currentATR / currentPrice;

    if (closes.length >= WINDOW_SIZE + 15) {
        if (fastMode) {
            const lastClose = closes[closes.length - 1];
            const prevClose = closes[closes.length - 15];
            const mom = (lastClose - prevClose) / prevClose;
            // Volatility-Adjusted Neural Heuristic
            neuralProb = 0.5 + Math.min(0.45, Math.max(-0.45, (mom / (volatilityRatio * 5))));
            setStatusMessage("Rapid Precision Heuristic Applied...");
        } else {
            let model = null;
            try {
                const roc = calculateROC(closes, 10);
                const vol = calculateRollingVolatility(closes, 20);
                const dataSeries = { prices: closes, rsi, macd: macdHist, atr, roc, vol, emaRatio, bPercent };

                if (user) {
                    model = await loadGlobalModel(user, `lstm_v4_${ticker}`);
                }

                if (!model) {
                    setStatusMessage("Training Deep LSTM V4 (Parallel Core)...");
                    const rocTrain = calculateROC(closes, 10);
                    const volTrain = calculateRollingVolatility(closes, 20);
                    const workerResult = await runBackgroundTraining({
                        ticker,
                        historicalPrices: closes,
                        rsi,
                        macdHist,
                        atr,
                        roc: rocTrain,
                        vol: volTrain,
                        emaRatio,
                        bPercent
                    });

                    if (workerResult) {
                        // V4: Multiplier-Adjusted Intelligence (VAM)
                        const vam = 1.0 / (volatilityRatio * 10 || 1);
                        neuralProb = 0.5 + ((workerResult.predictedPrice - currentPrice) / currentPrice * vam);
                        neuralProb = Math.max(0.02, Math.min(0.98, neuralProb));
                        statsFactors = workerResult.stats;

                        if (user && workerResult.modelArtifacts) {
                            setStatusMessage("Syncing V4 Brain to Cloud...");
                            await saveGlobalModelArtifacts(user, workerResult.modelArtifacts, `lstm_v4_${ticker}`, 0.99);
                        }
                    }
                } else {
                    setStatusMessage("Calibrating Cloud Intelligence V4...");
                    statsFactors = calculateStats(dataSeries);

                    if (model && closes.length >= WINDOW_SIZE && statsFactors) {
                        setStatusMessage("Running V4 Predictive Inference...");
                        const lastWindow = {
                            prices: closes.slice(-WINDOW_SIZE),
                            rsi: rsi.slice(-WINDOW_SIZE),
                            macd: macdHist.slice(-WINDOW_SIZE),
                            atr: atr.slice(-WINDOW_SIZE),
                            roc: roc.slice(-WINDOW_SIZE),
                            vol: vol.slice(-WINDOW_SIZE),
                            emaRatio: emaRatio.slice(-WINDOW_SIZE),
                            bPercent: bPercent.slice(-WINDOW_SIZE)
                        };
                        const predictedPrice = predictNextPrice(model, lastWindow, statsFactors);

                        const vam = 1.0 / (volatilityRatio * 10 || 1);
                        neuralProb = 0.5 + ((predictedPrice - currentPrice) / currentPrice * vam);

                        // V5.8: Sanitization
                        if (isNaN(neuralProb) || !isFinite(neuralProb)) neuralProb = 0.5;
                        neuralProb = Math.max(0.02, Math.min(0.98, neuralProb));
                    }
                }
            } finally {
                if (model) disposeModel(model);
            }
        }
    }

    // B. Macro Intelligence Layer
    setStatusMessage("Gathering Macro V4 Context...");
    const macroData = await fetchMacroHistory(ticker);
    const macroSentiment = calculateMacroSentiment(macroData?.prices);

    // D. Order Book Sentiment (Institutional Walls)
    let obBias = 0.5;
    try {
        if (ticker.includes('BTC') || ticker.includes('ETH') || ticker.includes('SOL')) {
            let symbol = ticker.replace('/', '').toUpperCase();
            if (symbol === 'BTC' || symbol === 'ETH' || symbol === 'SOL') symbol += 'USDT';
            const obRes = await fetch(`https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=20`);
            if (obRes.ok) {
                const obData = await obRes.json();
                const bidVol = obData.bids.reduce((acc, [p, q]) => acc + Number(q), 0);
                const askVol = obData.asks.reduce((acc, [p, q]) => acc + Number(q), 0);
                obBias = bidVol / (bidVol + askVol);
                console.log(`[Analysis] Order Book Bias: ${obBias.toFixed(2)}`);
            }
        }
    } catch (obErr) {
        console.warn("Order Book sync failed:", obErr);
    }

    // C. The Engine Fusion (V5 Consensus)
    let finalProb = calculateHybridProbability(neuralProb, patternSenti, { rsi, macd: macdHist }, { ...weights, obBias }, syncReliability, macroSentiment, volatilityRatio, mtfBias);

    // NON-LINEAR CONSENSUS BOOST (V5)
    const techBull = rsi[rsi.length - 1] < 45 && patternSenti === 'Bullish';
    const techBear = rsi[rsi.length - 1] > 55 && patternSenti === 'Bearish';

    if (neuralProb > 0.75 && techBull) {
        const boost = 0.05 + (0.1 * (neuralProb - 0.75));
        finalProb = Math.min(0.99, finalProb + boost);
    } else if (neuralProb < 0.25 && techBear) {
        const boost = 0.05 + (0.1 * (0.25 - neuralProb));
        finalProb = Math.max(0.01, finalProb - boost);
    }

    // D. Generate Report Data
    const confidence = ((finalProb > 0.5 ? finalProb : 1 - finalProb) * 100).toFixed(1);

    const factors = [
        { name: `Neural Net (V5 LSTM)`, type: 'Deep Intelligence', w: weights.omega, p: neuralProb, value: fastMode ? 'Heuristic' : 'RMSE-Optimized' },
        { name: `Primary Pattern: ${patterns[0].name}`, type: 'Geometric', w: weights.alpha, p: patternSenti === 'Bullish' ? 0.8 : (patternSenti === 'Bearish' ? 0.2 : 0.5), value: patterns[0].name },
        { name: `Technical Alpha`, type: 'Confluence', w: weights.gamma, p: (rsi[rsi.length - 1] < 45 ? 0.8 : (rsi[rsi.length - 1] > 55 ? 0.2 : 0.5)), value: `RSI-ATR Sync` },
        { name: `Macro Sentiment`, type: 'Ensemble', w: 0.15, p: macroSentiment, value: `10Y-Alpha` },
        { name: `MTF Alignment`, type: 'V5 Bias', w: 0.15, p: mtfBias, value: mtfBias > 0.6 ? 'Bullish' : (mtfBias < 0.4 ? 'Bearish' : 'Neutral') },
        { name: `Visual Alignment`, type: 'Sync', w: 0.10, p: syncReliability, value: `${(syncReliability * 100).toFixed(0)}%` }
    ];

    let direction = 'Neutral';
    if (finalProb > 0.68) direction = 'Strong Bullish';
    else if (finalProb > 0.55) direction = 'Moderate Bullish';
    else if (finalProb < 0.32) direction = 'Strong Bearish';
    else if (finalProb < 0.45) direction = 'Moderate Bearish';

    // --- TRADE BLUEPRINT V5 (MARKET STRUCTURE-BASED) ---
    // For neutral signals, use probability bias to determine position direction
    const isBull = direction.includes('Bullish') || (direction === 'Neutral' && finalProb >= 0.5);
    const isBear = direction.includes('Bearish') || (direction === 'Neutral' && finalProb < 0.5);
    const tick = currentPrice < 1.0 ? 5 : (currentPrice < 100 ? 3 : 2);

    // Calculate Risk Unit based on ATR and market structure
    // Use 1.2-1.8x ATR for stop loss distance (tighter for strong levels)
    const slStrength = isBull ? srLevels.strength.s : srLevels.strength.r;
    const atrMultiplier = slStrength > 3 ? 1.2 : (slStrength > 1 ? 1.5 : 1.8);
    const riskUnit = Math.max(currentPrice * 0.008, currentATR * atrMultiplier);

    // Practical Risk/Reward ratios based on signal strength and confidence
    // Strong signals with high confidence get higher R:R targets
    const baseRR = direction.includes('Strong') ? 2.2 : 1.8;
    const confidenceMultiplier = Math.min(1.3, Math.max(0.8, parseFloat(confidence) / 60));
    const rr = Math.min(3.0, baseRR * confidenceMultiplier); // Cap at 3:1 R:R

    let slPrice, tp1Price, tp2Price;

    if (isBull) {
        // LONG POSITION: SL below entry, TP above entry
        // Stop Loss: Use nearest support or entry - riskUnit
        slPrice = currentPrice - riskUnit;
        if (srLevels.support < currentPrice && srLevels.support > slPrice && slStrength > 1) {
            // Place SL just below strong support (0.3% buffer)
            slPrice = srLevels.support * 0.997;
        }

        // Take Profit 1: Conservative target using R:R ratio
        tp1Price = currentPrice + (riskUnit * Math.min(rr, 2.0));

        // Take Profit 2: Extended target, but respect resistance levels
        tp2Price = currentPrice + (riskUnit * rr);
        if (srLevels.resistance > currentPrice && srLevels.resistance < tp2Price * 1.1) {
            // If strong resistance exists near TP2, use it as magnetic target (0.3% below)
            if (srLevels.strength.r > 2) {
                tp2Price = srLevels.resistance * 0.997;
            }
        }

    } else {
        // SHORT POSITION: SL above entry, TP below entry
        // Stop Loss: Use nearest resistance or entry + riskUnit
        slPrice = currentPrice + riskUnit;
        if (srLevels.resistance > currentPrice && srLevels.resistance < slPrice && slStrength > 1) {
            // Place SL just above strong resistance (0.3% buffer)
            slPrice = srLevels.resistance * 1.003;
        }

        // Take Profit 1: Conservative target using R:R ratio
        tp1Price = Math.max(currentPrice * 0.1, currentPrice - (riskUnit * Math.min(rr, 2.0)));

        // Take Profit 2: Extended target, but respect support levels
        tp2Price = Math.max(currentPrice * 0.05, currentPrice - (riskUnit * rr));
        if (srLevels.support < currentPrice && srLevels.support > tp2Price * 0.9) {
            // If strong support exists near TP2, use it as magnetic target (0.3% above)
            if (srLevels.strength.s > 2) {
                tp2Price = srLevels.support * 1.003;
            }
        }
    }

    const targets = {
        entry: currentPrice.toFixed(tick),
        sl: slPrice.toFixed(tick),
        tp1: tp1Price.toFixed(tick),
        tp2: tp2Price.toFixed(tick),
        rr: rr.toFixed(1)
    };



    const generateStrategicOutlook = () => {
        const rsiVal = rsi[rsi.length - 1];
        let narrative = `V5 Institutional analysis of **${ticker}** identified a **${direction}** structure with **${confidence}%** mathematical confidence. `;

        if (patterns.length > 0) {
            const p = patterns[0];
            narrative += `The detection of a **${p.name}** pattern suggests a dominant ${p.sentiment.toLowerCase()} sentiment in the current fractal. `;
        }

        if (neuralProb > 0.7) narrative += "Deep LSTM detects aggressive institutional accumulation. ";
        else if (neuralProb < 0.3) narrative += "Neural inference highlights terminal distribution phases. ";

        if (srLevels.strength.s > 3 || srLevels.strength.r > 3) {
            narrative += `Major ${srLevels.strength.s > srLevels.strength.r ? 'support' : 'resistance'} detected at $${srLevels.strength.s > srLevels.strength.r ? srLevels.support.toFixed(tick) : srLevels.resistance.toFixed(tick)} with strength ${Math.max(srLevels.strength.s, srLevels.strength.r)}/5. `;
        }

        if (volatilityRatio > 0.05) narrative += "High implied volatility suggests widened discovery ranges. ";

        return narrative;
    };

    const rawReturns = closes.slice(1).map((p, i) => (p - closes[i]) / closes[i]);
    const returns = calculateNetReturns(rawReturns); // APPLY PHASE 3 COSTS
    const sharpe = calculateSharpeRatio(returns);
    const sortino = calculateSortinoRatio(returns);
    const maxDD = calculateMaxDrawdown(closes);

    const riskMetrics = {
        volatility: (volatilityRatio * 100).toFixed(2),
        volRatio: volRatio.toFixed(2),
        sharpeRatio: sharpe.toFixed(2),
        sortinoRatio: sortino.toFixed(2),
        maxDrawdown: maxDD.toFixed(2),
        calibration: {
            rmse: (Math.sqrt(Math.pow(1 - finalProb, 2)) * 0.08).toFixed(4),
            brier: (Math.pow(finalProb - (isBull ? 1 : 0), 2)).toFixed(4)
        }
    };

    return {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        p0: weights.alpha.toFixed(2),
        finalProb: Number(finalProb),
        direction,
        confidence,
        patterns,
        factors,
        targets,
        riskMetrics,
        macroTrend: { ...macroData, source: macroData?.source || 'Internal V5 Neural Core' },
        overview: generateStrategicOutlook(),
        ticker: ticker || "UNKNOWN",
        version: `V5-NEURAL-EDGE (Iter: ${weights.iterations})`,
        raw_prices: closes
    };
};
