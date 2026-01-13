export const calculateSMA = (prices, period) => {
    if (prices.length < period) return [];
    const sma = [];
    for (let i = period - 1; i < prices.length; i++) {
        const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        sma.push(sum / period);
    }
    return sma;
};

export const calculateEMA = (prices, period) => {
    if (prices.length < period) return [];
    const k = 2 / (period + 1);
    let emaArray = [prices[0]];
    for (let i = 1; i < prices.length; i++) {
        emaArray.push(prices[i] * k + emaArray[i - 1] * (1 - k));
    }
    // Return aligned with input prices, padding finding interval with null or first calc
    // But typically EMA starts after 'period' valid data points. For simplicity we assume valid series.
    return emaArray;
};

export const calculateRSI = (prices, period = 14) => {
    if (prices.length < period + 1) return [];

    let gains = 0;
    let losses = 0;

    // First period
    for (let i = 1; i <= period; i++) {
        const diff = prices[i] - prices[i - 1];
        if (diff >= 0) gains += diff;
        else losses += Math.abs(diff);
    }

    let avgGain = gains / period;
    let avgLoss = losses / period;
    const rsi = [];

    // Calculate initial RSI
    let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));

    // Following periods
    for (let i = period + 1; i < prices.length; i++) {
        const diff = prices[i] - prices[i - 1];
        if (diff >= 0) {
            avgGain = (avgGain * (period - 1) + diff) / period;
            avgLoss = (avgLoss * (period - 1) + 0) / period;
        } else {
            avgGain = (avgGain * (period - 1) + 0) / period;
            avgLoss = (avgLoss * (period - 1) + Math.abs(diff)) / period;
        }
        rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
};

export const calculateBollingerBands = (prices, period = 20, multiplier = 2) => {
    if (prices.length < period) return [];
    const lower = [];
    const upper = [];
    const basis = calculateSMA(prices, period);

    // SMA result is shorter than prices by (period-1)
    // We align indices. basis[0] corresponds to prices[period-1]

    for (let i = 0; i < basis.length; i++) {
        const slice = prices.slice(i, i + period); // window
        const mean = basis[i];
        const squaredDiffs = slice.map(curr => Math.pow(curr - mean, 2));
        const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
        const stdDev = Math.sqrt(variance);

        upper.push(mean + (multiplier * stdDev));
        lower.push(mean - (multiplier * stdDev));
    }

    return { basis, upper, lower };
};

export const calculateMACD = (prices, fast = 12, slow = 26, signal = 9) => {
    const emaFast = calculateEMA(prices, fast);
    const emaSlow = calculateEMA(prices, slow);

    // EMA arrays are same length as prices in our simplified implementation above
    // Real MACD line calculation
    const macdLine = [];
    const minLength = Math.min(emaFast.length, emaSlow.length);
    for (let i = 0; i < minLength; i++) {
        macdLine.push(emaFast[i] - emaSlow[i]);
    }

    const signalLine = calculateEMA(macdLine, signal);
    const histogram = [];
    for (let i = 0; i < Math.min(macdLine.length, signalLine.length); i++) {
        histogram.push(macdLine[i] - signalLine[i]);
    }

    return { macdLine, signalLine, histogram };
};

// --- Pattern Recognition Logic ---
export const detectPatterns = (prices) => {
    if (prices.length < 50) return { name: 'Insufficient Data', sentiment: 'Neutral', confidence: 0 };

    const last = prices[prices.length - 1];
    const prev = prices[prices.length - 2];

    // 1. Trend Detection (Simple moving average slope)
    const shortMA = calculateSMA(prices.slice(-20), 10);
    const longMA = calculateSMA(prices.slice(-50), 40);
    const isUptrend = shortMA[shortMA.length - 1] > longMA[longMA.length - 1];

    // 2. Volatility Squeeze (Bollinger Bands)
    const { upper, lower } = calculateBollingerBands(prices, 20);
    const bandwidth = (upper[upper.length - 1] - lower[lower.length - 1]) / prices[prices.length - 1];
    const isSqueeze = bandwidth < 0.05; // Tight bands

    // 3. RSI Divergence placeholder
    const rsi = calculateRSI(prices, 14);
    const lastRSI = rsi[rsi.length - 1];
    const isOversold = lastRSI < 30;
    const isOverbought = lastRSI > 70;

    let patterns = [];

    if (isSqueeze) patterns.push({ name: 'Volatility Squeeze', sentiment: 'Neutral' });
    if (isUptrend && !isOverbought) patterns.push({ name: 'Bullish Continuation', sentiment: 'Bullish' });
    if (!isUptrend && !isOversold) patterns.push({ name: 'Bearish Continuation', sentiment: 'Bearish' });
    if (isUptrend && isOverbought) patterns.push({ name: 'Potential Reversal (Top)', sentiment: 'Bearish' });
    if (!isUptrend && isOversold) patterns.push({ name: 'Potential Reversal (Bottom)', sentiment: 'Bullish' });

    // Return the most significant pattern
    return patterns.length > 0 ? patterns[0] : { name: 'Consolidation', sentiment: 'Neutral' };
};

export const calculateATR = (highs, lows, closes, period = 14) => {
    // Check if we have triple arrays or single array
    let tr = [];
    const hasFullData = Array.isArray(highs) && Array.isArray(lows) && Array.isArray(closes);

    const len = hasFullData ? highs.length : (Array.isArray(highs) ? highs.length : 0);
    if (len < period + 1) return [];

    for (let i = 1; i < len; i++) {
        let currentTR;
        if (hasFullData) {
            const h = highs[i];
            const l = lows[i];
            const pc = closes[i - 1];
            currentTR = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
        } else {
            // Fallback for single array (assumed Closes)
            const current = highs[i];
            const prev = highs[i - 1];
            const change = Math.abs(current - prev);
            // 1.5x multiplier on Close-to-Close as a better proxy for True Range when H/L are missing
            currentTR = Math.max(change, current * 0.0075);
        }
        tr.push(currentTR);
    }

    const atr = [];
    let initialATR = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
    atr.push(initialATR);

    for (let i = period; i < tr.length; i++) {
        const nextATR = ((atr[atr.length - 1] * (period - 1)) + tr[i]) / period;
        atr.push(nextATR);
    }

    const padding = new Array(len - atr.length).fill(atr[0]);
    return [...padding, ...atr];
};

export const findSupportResistance = (closes, highs, lows) => {
    const prices = Array.isArray(closes) ? closes : [];
    if (prices.length < 20) return { support: Math.min(...prices), resistance: Math.max(...prices), strength: { s: 1, r: 1 } };

    const useHL = Array.isArray(highs) && Array.isArray(lows) && highs.length === prices.length;
    const levels = [];
    const window = 5;

    for (let i = window; i < prices.length - window; i++) {
        // Resistance: Look at highs
        if (useHL) {
            const hSlice = highs.slice(i - window, i + window + 1);
            const currentH = highs[i];
            if (hSlice.every(p => p <= currentH)) levels.push({ price: currentH, type: 'Resistance' });

            const lSlice = lows.slice(i - window, i + window + 1);
            const currentL = lows[i];
            if (lSlice.every(p => p >= currentL)) levels.push({ price: currentL, type: 'Support' });
        } else {
            const slice = prices.slice(i - window, i + window + 1);
            const current = prices[i];
            const isMax = slice.every(p => p <= current);
            const isMin = slice.every(p => p >= current);
            if (isMax) levels.push({ price: current, type: 'Resistance' });
            if (isMin) levels.push({ price: current, type: 'Support' });
        }
    }

    const currentPrice = prices[prices.length - 1];

    // Improved Strength Detection: Group nearby levels
    const getNearestLevel = (type, target) => {
        const filtered = levels.filter(l => l.type === type && (type === 'Support' ? l.price < target : l.price > target));
        if (filtered.length === 0) return { price: (type === 'Support' ? Math.min(...prices) : Math.max(...prices)), strength: 1 };

        // Sort by proximity
        const sorted = filtered.sort((a, b) => Math.abs(a.price - target) - Math.abs(b.price - target));
        const nearest = sorted[0].price;

        // Count how many times this price level (within 0.5% tolerance) was touched
        const hits = levels.filter(l => l.type === type && Math.abs(l.price - nearest) / nearest < 0.005).length;

        return { price: nearest, strength: Math.min(5, hits) };
    };

    const s = getNearestLevel('Support', currentPrice);
    const r = getNearestLevel('Resistance', currentPrice);

    return {
        support: s.price,
        resistance: r.price,
        strength: { s: s.strength, r: r.strength }
    };
};
