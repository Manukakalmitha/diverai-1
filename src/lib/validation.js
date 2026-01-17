/**
 * Advanced Financial ML Validation Techniques (López de Prado)
 */

/**
 * Deflated Sharpe Ratio (DSR)
 * Adjusts the Sharpe Ratio for multiple testing bias, skewness, and kurtosis.
 * 
 * @param {number} SR - The observed Sharpe Ratio (annualized)
 * @param {number} SR_bench - Benchmark Sharpe Ratio (usually 0)
 * @param {number} T - Number of observations/samples
 * @param {number} N - Number of independent trials (strategies tested)
 * @param {number} skew - Skewness of returns
 * @param {number} kurt - Kurtosis of returns
 */
export const calculateDeflatedSharpeRatio = (SR, T, N = 1, skew = 0, kurt = 3) => {
    // Annualization factor (assuming daily data)
    const SR_daily = SR / Math.sqrt(252);

    // Euler-Mascheroni constant approximation
    const GAMMA = 0.5772156649;

    // Expected maximum SR under the null hypothesis (multiple testing bias)
    const expectedMaxSR = (1 - GAMMA) * Math.sqrt(2 * Math.log(N)) + GAMMA * Math.sqrt(2 * Math.log(N));

    // Standard deviation of the SR estimate
    const sigmaSR = Math.sqrt((1 - skew * SR_daily + ((kurt - 1) / 4) * Math.pow(SR_daily, 2)) / (T - 1));

    // Probabilistic Sharpe Ratio (PSR) vs the expected max SR
    const zScore = (SR_daily - (expectedMaxSR / Math.sqrt(252))) / sigmaSR;

    // Return the DSR (as a confidence/probability score 0-1)
    // Using an approximation for the cumulative normal distribution
    const dsr = 0.5 * (1 + errorFunction(zScore / Math.sqrt(2)));

    return dsr;
};

/**
 * Standard Error function (approximation for normal distribution)
 */
function errorFunction(x) {
    const t = 1 / (1 + 0.5 * Math.abs(x));
    const tau = t * Math.exp(-Math.pow(x, 2) - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - tau : tau - 1;
}

/**
 * Purged and Embargoed Data Splitting
 * Prevents data leakage by removing overlapping and correlated samples.
 * 
 * @param {Array} data - The full dataset
 * @param {number} testIdx - Start index of the test set
 * @param {number} testLen - Length of the test set
 * @param {number} purgeLen - Number of points to remove before/after test set
 */
export const getPurgedTrainTestSplit = (data, testIdx, testLen, purgeLen = 5) => {
    const endIdx = testIdx + testLen;

    // Test set
    const testSet = data.slice(testIdx, endIdx);

    // Training set (everything else, minus purging/embargoing)
    const trainSet = [
        ...data.slice(0, Math.max(0, testIdx - purgeLen)),
        ...data.slice(Math.min(data.length, endIdx + purgeLen))
    ];

    return { trainSet, testSet };
};

/**
 * Higher-order moments for DSR calculation
 */
export const calculateMoments = (returns) => {
    if (returns.length === 0) return { skew: 0, kurt: 3 };

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length) || 1;

    let skew = 0;
    let kurt = 0;

    for (const r of returns) {
        const z = (r - mean) / std;
        skew += Math.pow(z, 3);
        kurt += Math.pow(z, 4);
    }

    return {
        skew: skew / returns.length,
        kurt: kurt / returns.length
    };
};
