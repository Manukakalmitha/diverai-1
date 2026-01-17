// Advanced Feature Engineering for Financial ML

/**
 * Calculate rolling standard deviation
 */
export const calculateRollingStd = (prices, period) => {
    if (!prices || prices.length < period) return [];
    const rollingStd = [];

    for (let i = period - 1; i < prices.length; i++) {
        const slice = prices.slice(i - period + 1, i + 1);
        const mean = slice.reduce((a, b) => a + b, 0) / period;
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        rollingStd.push(Math.sqrt(variance));
    }

    return rollingStd;
};

/**
 * Calculate Rate of Change (ROC) - Momentum indicator
 */
export const calculateROC = (prices, period) => {
    if (!prices || prices.length < period + 1) return [];
    const roc = [];

    for (let i = period; i < prices.length; i++) {
        const change = (prices[i] - prices[i - period]) / prices[i - period];
        roc.push(change * 100); // As percentage
    }

    return roc;
};

/**
 * Calculate rolling momentum (sum of returns)
 */
export const calculateMomentum = (prices, period) => {
    if (!prices || prices.length < period + 1) return [];
    const momentum = [];

    for (let i = period; i < prices.length; i++) {
        let sumReturns = 0;
        for (let j = i - period + 1; j <= i; j++) {
            const ret = (prices[j] - prices[j - 1]) / prices[j - 1];
            sumReturns += ret;
        }
        momentum.push(sumReturns);
    }

    return momentum;
};

/**
 * Calculate rolling volatility (annualized)
 */
export const calculateRollingVolatility = (prices, period, annualizationFactor = 252) => {
    if (!prices || prices.length < period + 1) return [];
    const returns = [];

    // Calculate returns
    for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }

    const rollingVol = [];
    for (let i = period - 1; i < returns.length; i++) {
        const slice = returns.slice(i - period + 1, i + 1);
        const mean = slice.reduce((a, b) => a + b, 0) / period;
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        const vol = Math.sqrt(variance) * Math.sqrt(annualizationFactor);
        rollingVol.push(vol);
    }

    return rollingVol;
};

/**
 * Calculate Sharpe Ratio
 * @param {number[]} returns - Array of returns
 * @param {number} riskFreeRate - Annual risk-free rate (default 0)
 * @param {number} periods - Periods per year for annualization (252 for daily, 12 for monthly)
 */
export const calculateSharpeRatio = (returns, riskFreeRate = 0, periods = 252) => {
    if (!returns || returns.length === 0) return 0;

    const excessReturns = returns.map(r => r - (riskFreeRate / periods));
    const avgReturn = excessReturns.reduce((a, b) => a + b, 0) / returns.length;

    if (returns.length < 2) return 0;

    const variance = excessReturns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return 0;

    // Annualized Sharpe Ratio
    return (avgReturn / stdDev) * Math.sqrt(periods);
};

/**
 * Calculate Sortino Ratio (uses downside deviation only)
 * @param {number[]} returns - Array of returns
 * @param {number} targetReturn - Minimum acceptable return (MAR), default 0
 * @param {number} periods - Periods per year for annualization
 */
export const calculateSortinoRatio = (returns, targetReturn = 0, periods = 252) => {
    if (!returns || returns.length === 0) return 0;

    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const target = targetReturn / periods;

    // Calculate downside deviation (only negative deviations from target)
    const downsideReturns = returns.map(r => r < target ? Math.pow(r - target, 2) : 0);
    const downsideVariance = downsideReturns.reduce((a, b) => a + b, 0) / returns.length;
    const downsideDeviation = Math.sqrt(downsideVariance);

    if (downsideDeviation === 0) return 0;

    // Annualized Sortino Ratio
    return ((avgReturn - target) / downsideDeviation) * Math.sqrt(periods);
};

/**
 * Calculate Maximum Drawdown
 */
export const calculateMaxDrawdown = (prices) => {
    if (!prices || prices.length < 2) return 0;

    let maxPrice = prices[0];
    let maxDrawdown = 0;

    for (let i = 1; i < prices.length; i++) {
        if (prices[i] > maxPrice) {
            maxPrice = prices[i];
        }
        const drawdown = (maxPrice - prices[i]) / maxPrice;
        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
        }
    }

    return maxDrawdown * 100; // As percentage
};

/**
 * Calculate Volatility Ratio (recent vs longer-term)
 */
export const calculateVolatilityRatio = (prices, shortPeriod = 7, longPeriod = 21) => {
    const shortVol = calculateRollingVolatility(prices, shortPeriod);
    const longVol = calculateRollingVolatility(prices, longPeriod);

    if (shortVol.length === 0 || longVol.length === 0) return 1.0;

    const recentShortVol = shortVol[shortVol.length - 1];
    const recentLongVol = longVol[longVol.length - 1];

    return recentLongVol === 0 ? 1.0 : recentShortVol / recentLongVol;
};

/**
 * Calculate returns after transaction costs and slippage
 * @param {number[]} returns - Array of raw returns
 * @param {number} spread - Average bid-ask spread (e.g., 0.0001 for 1 bip)
 * @param {number} slippage - Average slippage per trade (e.g., 0.0005 for 5 bips)
 */
export const calculateNetReturns = (returns, spread = 0.0001, slippage = 0.0002) => {
    const totalCost = spread + slippage;
    return returns.map(r => {
        // Only apply costs if there's a "trade" (non-zero return)
        if (Math.abs(r) < 0.0000001) return 0;
        return r - totalCost;
    });
};
