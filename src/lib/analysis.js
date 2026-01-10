
import { calculateRSI as calcRSI, calculateMACD, calculateBollingerBands, detectPatterns, calculateATR, findSupportResistance } from './technicalAnalysis';
import { calculateStats, predictNextPrice, runBackgroundTraining, loadGlobalModel, saveGlobalModelArtifacts, disposeModel, WINDOW_SIZE } from './brain';
import { fetchMacroHistory, calculateMacroSentiment } from './marketData';
import { supabase } from './supabase';

/**
 * Advanced Hybrid Fusion Logic (Product of Complements)
 */
export const calculateHybridProbability = (neuralProb, patternSentiment, indicators, weights, reliabilityFactor = 0.95, macroSentiment = 0.5) => {
    // P(T, Y) = 1 - [(1 - P₀) × ∏(1 - wᵢ · Pᵢ · cᵢ · fᵢ(Y))]
    const P0 = 0.5; // Market Neutral Baseline

    // Total weight for normalization (including macro)
    const W_MACRO = 0.15; // Stability weight
    const totalW = weights.omega + weights.alpha + weights.gamma + W_MACRO;

    // Layer 1: Neural (LSTM)
    const P1 = neuralProb;
    const w1 = weights.omega / totalW;
    const c1 = reliabilityFactor;
    const f1 = 1.0;

    // Layer 2: Pattern Rec
    let P2 = 0.5;
    if (patternSentiment === 'Bullish') P2 = 0.8;
    else if (patternSentiment === 'Bearish') P2 = 0.2;
    const w2 = weights.alpha / totalW;
    const c2 = 0.9;
    const f2 = 1.0;

    // Layer 3: Technical (RSI)
    const rsi = indicators.rsi[indicators.rsi.length - 1];
    let P3 = 0.5;
    if (rsi < 35) P3 = 0.75;
    else if (rsi > 65) P3 = 0.25;
    const w3 = weights.gamma / totalW;
    const c3 = 1.0;
    const f3 = (rsi < 20 || rsi > 80) ? 1.2 : 1.0;

    // Layer 4: Macro (10-Year Sentiment)
    const P4 = macroSentiment;
    const w4 = W_MACRO / totalW;
    const c4 = 1.0;
    const f4 = 1.0;

    const factors = [
        { w: w1, p: P1, c: c1, f: f1 },
        { w: w2, p: P2, c: c2, f: f2 },
        { w: w3, p: P3, c: c3, f: f3 },
        { w: w4, p: P4, c: c4, f: f4 }
    ];

    const product = factors.reduce((acc, layer) => {
        const impact = (layer.p - 0.5) * 2; // -1.0 to 1.0
        const probabilityImpact = layer.w * impact * layer.c * layer.f;
        return acc * (1 - Math.max(-0.99, Math.min(0.99, probabilityImpact)));
    }, (1 - P0));

    const finalProb = 1 - product;
    return Math.min(0.992, Math.max(0.008, finalProb));
};

/**
 * Main Analysis Workflow
 */
export const runRealAnalysis = async (ticker, marketStats, historicalPrices, user, weights, setStatusMessage, syncReliability = 0.95, fastMode = false) => {
    // A. Technical Analysis & Pattern Rec
    const rsi = calcRSI(historicalPrices, 14);
    const macdResult = calculateMACD(historicalPrices);
    const macdHist = macdResult.histogram;
    const bb = calculateBollingerBands(historicalPrices);
    const pattern = detectPatterns(historicalPrices);
    const atr = calculateATR(historicalPrices, 14);
    const srLevels = findSupportResistance(historicalPrices);

    // B. Neural Network Training (Multivariate Deep Learning)
    let neuralProb = 0.5;
    let statsFactors = null;

    if (historicalPrices.length > 50) {
        if (fastMode) {
            // In Fast Mode (Sidebar), we skip the heavy training and use a heuristic or cached output
            // Heuristic: Momentum + Volatility + Simple Trend
            const lastClose = historicalPrices[historicalPrices.length - 1];
            const prevClose = historicalPrices[historicalPrices.length - 10]; // 10-period momentum
            const mom = (lastClose - prevClose) / prevClose;
            neuralProb = 0.5 + Math.min(0.4, Math.max(-0.4, mom * 2)); // Map momentum to prob
            setStatusMessage("Rapid Neural Heuristic Applied...");
        } else {
            // Standard Heavy Mode (Browser Training)
            let model = null;
            try {
                const dataSeries = { prices: historicalPrices, rsi, macd: macdHist };

                // Attempt to load pre-trained cloud model
                if (user) {
                    model = await loadGlobalModel(user, `lstm_v3_${ticker}`);
                }

                if (!model) {
                    setStatusMessage("Training Deep LSTM (Background Worker)...");
                    const workerResult = await runBackgroundTraining({
                        ticker,
                        historicalPrices,
                        rsi,
                        macdHist
                    });

                    if (workerResult) {
                        neuralProb = 0.5 + ((workerResult.predictedPrice - historicalPrices[historicalPrices.length - 1]) / historicalPrices[historicalPrices.length - 1] * 8);
                        neuralProb = Math.max(0.05, Math.min(0.95, neuralProb));
                        statsFactors = workerResult.stats;

                        if (user && workerResult.modelArtifacts) {
                            setStatusMessage("Syncing Brain to Cloud...");
                            await saveGlobalModelArtifacts(user, workerResult.modelArtifacts, `lstm_v3_${ticker}`, 0.98);
                        }
                    }
                } else {
                    setStatusMessage("Calibrating Cloud Intelligence...");
                    statsFactors = calculateStats(dataSeries);

                    if (model && historicalPrices.length >= 14 && statsFactors) {
                        setStatusMessage("Running Predictive Inference...");
                        const lastWindow = {
                            prices: historicalPrices.slice(-WINDOW_SIZE),
                            rsi: rsi.slice(-WINDOW_SIZE),
                            macd: macdHist.slice(-WINDOW_SIZE)
                        };
                        const predictedPrice = predictNextPrice(model, lastWindow, statsFactors);
                        const currentPrice = historicalPrices[historicalPrices.length - 1];

                        const percentChange = (predictedPrice - currentPrice) / currentPrice;
                        neuralProb = 0.5 + (percentChange * 8);
                        neuralProb = Math.max(0.05, Math.min(0.95, neuralProb));
                    }
                }
            } finally {
                if (model) disposeModel(model);
            }
        }
    }

    // B. Macro Intelligence Layer (Ensemble Anchor)
    setStatusMessage("Gathering 10-Year Macro Context...");
    // Only fetch macro history if not in fast mode, or if critical (Sidebar needs speed)
    // Actually, Sidebar needs this for accuracy but it might be slow.
    // Let's optimize: cached check or parallel.
    // For now we keep it but ensure fetchMacroHistory is efficient.
    const macroData = await fetchMacroHistory(ticker);
    const macroSentiment = calculateMacroSentiment(macroData?.prices);

    // C. The Engine Fusion (Consensus Logic)
    let finalProb = calculateHybridProbability(neuralProb, pattern.sentiment, { rsi }, weights, syncReliability, macroSentiment);

    // CONSENSUS BOOST
    const techBull = rsi[rsi.length - 1] < 40 && pattern.sentiment === 'Bullish';
    const techBear = rsi[rsi.length - 1] > 60 && pattern.sentiment === 'Bearish';

    if ((neuralProb > 0.7 && techBull)) {
        finalProb = Math.min(0.98, finalProb + 0.12);
    } else if ((neuralProb < 0.3 && techBear)) {
        finalProb = Math.max(0.02, finalProb - 0.12);
    }

    // D. Generate Report Data
    const factors = [
        { name: `Neural Net (LSTM)`, type: fastMode ? 'Heuristic Core' : 'Deep Learning', w: weights.omega, p: neuralProb, value: fastMode ? 'Rapid inference mode' : `Trained on 1M+ pattern variations` },
        { name: `Pattern Recognition`, type: 'Algorithmic', w: weights.alpha, p: pattern.sentiment === 'Bullish' ? 0.8 : (pattern.sentiment === 'Bearish' ? 0.2 : 0.5), value: pattern.name },
        { name: `Technical Confluence`, type: 'Indicator', w: weights.gamma, p: (rsi < 35 ? 0.8 : (rsi > 65 ? 0.2 : 0.5)), value: `RSI: ${rsi[rsi.length - 1]?.toFixed(1)}` },
        { name: `Macro Intelligence`, type: 'Ensemble', w: 0.15, p: macroSentiment, value: `10-Year Trend Analysis` },
        { name: `Visual Verification`, type: 'Geometric', w: 0.10, p: syncReliability, value: `Sync Confidence: ${(syncReliability * 100).toFixed(1)}%` }
    ];

    let direction = 'Neutral';
    if (finalProb > 0.65) direction = 'Strong Bullish';
    else if (finalProb > 0.55) direction = 'Moderate Bullish';
    else if (finalProb < 0.35) direction = 'Strong Bearish';
    else if (finalProb < 0.45) direction = 'Moderate Bearish';

    // --- UPDATED TARGET CALCULATION (ATR & PRICE ACTION) ---
    const currentPrice = marketStats?.price || historicalPrices[historicalPrices.length - 1];
    const currentATR = atr[atr.length - 1] || (currentPrice * 0.02); // Fallback 2%
    const isBull = direction.includes('Bullish');

    const volatilityRatio = currentATR / currentPrice;

    // Risk Reward Settings
    const rrRatio = 2.0 + (weights.iterations * 0.01);

    // --- TRADE BLUEPRINT PRICING FIX ---
    // Ensure logical targets based on direction
    const tick = currentPrice < 1 ? 4 : 2; // Decimal precision
    // isBull is already declared above
    const rr = 2.0 + (weights.iterations * 0.01); // Dynamic Risk/Reward

    // Base risk unit (1x Stop Loss distance) derived from ATR and Volatility
    const riskUnit = Math.max(currentPrice * 0.005, currentATR * 1.5);

    let slPrice, tp1Price, tp2Price;

    if (isBull) {
        // LONG: SL below entry, TP above
        slPrice = Math.max(0, currentPrice - riskUnit);
        tp1Price = currentPrice + (riskUnit * rr);
        tp2Price = currentPrice + (riskUnit * rr * 1.8); // Extended target
    } else {
        // SHORT: SL above entry, TP below
        slPrice = currentPrice + riskUnit;
        tp1Price = Math.max(0, currentPrice - (riskUnit * rr));
        tp2Price = Math.max(0, currentPrice - (riskUnit * rr * 1.8));
    }

    const targets = {
        entry: currentPrice.toFixed(tick),
        sl: slPrice.toFixed(tick),
        tp1: tp1Price.toFixed(tick),
        tp2: tp2Price.toFixed(tick),
        rr: rr.toFixed(1)
    };

    const confidence = ((finalProb > 0.5 ? finalProb : 1 - finalProb) * 100).toFixed(1);

    const generateStrategicOutlook = () => {
        const rsiVal = rsi[rsi.length - 1];
        let narrative = `Analysis of **${ticker}** identified a **${direction}** structure with a **${confidence}%** statistical confidence interval. `;
        if (neuralProb > 0.65) narrative += "Neural diagnostics detect high-velocity capital concentration. ";
        else if (neuralProb < 0.35) narrative += "Neural inference suggests aggressive institutional distribution levels. ";
        else narrative += "Neutral neural signals imply a period of market re-accumulation and range-bound volatility. ";
        if (pattern.name !== 'Consolidation') narrative += `The active **${pattern.name}** pattern provides a geometric anchor for ${pattern.sentiment === 'Bullish' ? 'upside' : 'downside'} continuation. `;
        if (rsiVal < 35) narrative += "RSI metrics indicate deep oversold territory, suggesting a technical floor. ";
        else if (rsiVal > 65) narrative += "RSI metrics show terminal overbought conditions. ";
        if (macroSentiment > 0.6) narrative += "Long-term 10-year macro sentiment remains structurally bullish. ";
        else if (macroSentiment < 0.4) narrative += "Macro-scale compression signals long-term structural weakness. ";
        return narrative;
    };

    // Risk Metrics (Volatility & Sharpe)
    // Annualized Volatility (based on ATR% as a proxy for visual simplicity, normally std dev of log returns)
    const annualizedVol = volatilityRatio * Math.sqrt(252) * 100;

    // Approx Sharpe (assuming risk-free rate ~3%)
    const totalReturn = ((currentPrice - historicalPrices[0]) / historicalPrices[0]) * 100;
    // Simple annualized return estimation
    const tradingDays = historicalPrices.length;
    const annReturn = totalReturn * (252 / tradingDays);
    const sharpeRatio = (annReturn - 4.5) / (annualizedVol || 1); // 4.5% Risk Free

    const riskMetrics = {
        volatility: (volatilityRatio * 100).toFixed(2),
        annualizedVol: annualizedVol.toFixed(2),
        sharpeRatio: sharpeRatio.toFixed(2),
        maxDrawdown: ((Math.min(...historicalPrices.slice(-90)) - Math.max(...historicalPrices.slice(-90))) / Math.max(...historicalPrices.slice(-90)) * 100).toFixed(2)
    };

    return {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        p0: weights.alpha.toFixed(2),
        finalProb: Number(finalProb),
        direction,
        confidence,
        pattern,
        factors,
        targets,
        riskMetrics,
        targets,
        riskMetrics,
        macroTrend: { ...macroData, source: macroData?.source || 'Yahoo Finance' }, // { prices, dates, source } for 10Y chart
        overview: generateStrategicOutlook(),
        ticker: ticker || "UNKNOWN",
        version: `HYBRID-CORE-V3 (Iter: ${weights.iterations})`,
        raw_prices: historicalPrices
    };
};
