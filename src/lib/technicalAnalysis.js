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

export const calculateATR = (prices, period = 14) => {
    if (prices.length < period + 1) return [];

    let tr = [];
    // Calculate True Range for each candle
    for (let i = 1; i < prices.length; i++) {
        const high = prices[i]; // Approximate High (using Close for simplicity if High not avail, but here we only have single price array. For better ATR we need High/Low/Close. 
        // fallback: using Volatility of Close-to-Close as proxy for TR if only Close exists.
        // TR = max( |High - Low|, |High - PrevClose|, |Low - PrevClose| )
        // Since we only have 'prices' (assumed Close), we will use |Close - PrevClose| which is a simplified volatility.
        // ideally we should pass {high, low, close} to this function.
        // Assuming 'prices' is just an array of numbers.

        const current = prices[i];
        const prev = prices[i - 1];

        // Simulating High/Low from Close allows for minimal ATR estimation
        // A better approach for this app since we only fetch 'prices' (closes) usually:
        // Use a multiplier on the absolute change to estimate "True Range" including intraday noise.
        const change = Math.abs(current - prev);
        const estimatedIntradayVolatility = current * 0.005; // 0.5% base noise
        const estimatedTR = Math.max(change, estimatedIntradayVolatility);
        tr.push(estimatedTR);
    }

    // SMA of TR
    const atr = [];
    let initialATR = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
    atr.push(initialATR);

    for (let i = period; i < tr.length; i++) {
        const nextATR = ((atr[atr.length - 1] * (period - 1)) + tr[i]) / period;
        atr.push(nextATR);
    }

    // Pad the beginning to match prices length (roughly)
    const padding = new Array(prices.length - atr.length).fill(null);
    return [...padding, ...atr];
};

export const findSupportResistance = (prices) => {
    if (prices.length < 20) return { support: Math.min(...prices), resistance: Math.max(...prices) };

    // Simple local extrema finding
    const levels = [];
    const window = 5;

    for (let i = window; i < prices.length - window; i++) {
        const slice = prices.slice(i - window, i + window + 1);
        const current = prices[i];
        const isMax = slice.every(p => p <= current);
        const isMin = slice.every(p => p >= current);

        if (isMax) levels.push({ price: current, type: 'Resistance' });
        if (isMin) levels.push({ price: current, type: 'Support' });
    }

    // Find nearest strong levels to current price
    const currentPrice = prices[prices.length - 1];

    const supports = levels.filter(l => l.type === 'Support' && l.price < currentPrice).sort((a, b) => b.price - a.price);
    const resistances = levels.filter(l => l.type === 'Resistance' && l.price > currentPrice).sort((a, b) => a.price - b.price);

    return {
        support: supports.length > 0 ? supports[0].price : Math.min(...prices), // Fallback to all-time low of period
        resistance: resistances.length > 0 ? resistances[0].price : Math.max(...prices) // Fallback to all-time high of period
    };
};
