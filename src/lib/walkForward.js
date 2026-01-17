import * as tf from '@tensorflow/tfjs';
import { createModel, trainModel, predictNextPrice, WINDOW_SIZE, FEATURES } from './brain.js';
import { calculateRSI, calculateMACD, calculateATR } from './technicalAnalysis.js';
import { calculateROC, calculateRollingVolatility, calculateSharpeRatio, calculateNetReturns } from './featureEngineering.js';
import { getPurgedTrainTestSplit, calculateDeflatedSharpeRatio, calculateMoments } from './validation.js';

/**
 * Walk-Forward Validation Framework (Advanced V2)
 * This prevents overfitting by testing the model on "unseen" future data 
 * in a rolling window fashion, with purging to prevent leakage.
 */
export const runWalkForwardValidation = async (fullPrices, windowSize = 75, testSize = 15) => {
    // 1. Prepare indicators for the entire series
    const rsi = calculateRSI(fullPrices, 14);
    const macdResult = calculateMACD(fullPrices);
    const macdHist = macdResult.histogram;
    const atr = calculateATR(fullPrices, null, fullPrices, 14);
    const roc = calculateROC(fullPrices, 10);
    const vol = calculateRollingVolatility(fullPrices, 20);

    const minLen = Math.min(fullPrices.length, rsi.length, macdHist.length, atr.length, roc.length, vol.length);
    const prices = fullPrices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macdHist.slice(-minLen);
    const a = atr.slice(-minLen);
    const ro = roc.slice(-minLen);
    const v = vol.slice(-minLen);

    // ZIP Data for easier splitting
    const dataObjects = prices.map((p, i) => ({
        price: p, r: r[i], m: m[i], a: a[i], ro: ro[i], v: v[i]
    }));

    const results = [];
    const trainingMetrics = [];
    const testingMetrics = [];
    const allTestReturns = [];

    // 2. Loop through data with a rolling window
    // Use PURGED SPLITS to ensure no information leakage
    for (let i = windowSize; i < prices.length - testSize; i += testSize) {
        const model = createModel();

        // --- PURGED SPLITTING ---
        const { trainSet, testSet } = getPurgedTrainTestSplit(dataObjects, i, testSize, 5);

        const trainDataSeries = {
            prices: trainSet.map(d => d.price),
            rsi: trainSet.map(d => d.r),
            macd: trainSet.map(d => d.m),
            atr: trainSet.map(d => d.a),
            roc: trainSet.map(d => d.ro),
            vol: trainSet.map(d => d.v)
        };

        const { stats } = await trainModel(model, trainDataSeries, 15);

        // Calculate training sharpe
        const rawTrainReturns = trainDataSeries.prices.slice(1).map((p, idx) => (p - trainDataSeries.prices[idx]) / trainDataSeries.prices[idx]);
        const trainReturns = calculateNetReturns(rawTrainReturns); // Apply Costs
        const trainSharpe = calculateSharpeRatio(trainReturns);
        trainingMetrics.push(trainSharpe);

        // --- TESTING PHASE (Out-of-Sample) ---
        const oosPredictions = [];
        const oosActuals = [];

        for (let j = 0; j < testSize; j++) {
            const lastWindowSeries = {
                prices: prices.slice(i + j - WINDOW_SIZE, i + j),
                rsi: r.slice(i + j - WINDOW_SIZE, i + j),
                macd: m.slice(i + j - WINDOW_SIZE, i + j),
                atr: a.slice(i + j - WINDOW_SIZE, i + j),
                roc: ro.slice(i + j - WINDOW_SIZE, i + j),
                vol: v.slice(i + j - WINDOW_SIZE, i + j)
            };

            const predicted = predictNextPrice(model, lastWindowSeries, stats);
            oosPredictions.push(predicted);
            oosActuals.push(prices[i + j]);
        }

        const rawTestReturns = oosActuals.slice(1).map((p, idx) => (p - oosActuals[idx]) / oosActuals[idx]);
        const testReturns = calculateNetReturns(rawTestReturns); // Apply Costs
        allTestReturns.push(...testReturns);
        const testSharpe = calculateSharpeRatio(testReturns);
        testingMetrics.push(testSharpe);

        results.push({
            window: i,
            trainSharpe,
            testSharpe,
            oosPredictions
        });

        model.dispose();
    }

    // 3. ADVANCED METRICS: Deflated Sharpe Ratio
    const avgTrainSharpe = trainingMetrics.reduce((a, b) => a + b, 0) / (trainingMetrics.length || 1);
    const avgTestSharpe = testingMetrics.reduce((a, b) => a + b, 0) / (testingMetrics.length || 1);

    const moments = calculateMoments(allTestReturns);
    const dsr = calculateDeflatedSharpeRatio(avgTestSharpe, allTestReturns.length, 1, moments.skew, moments.kurt);

    return {
        wfe: (avgTestSharpe / (avgTrainSharpe || 1)).toFixed(4),
        dsr: dsr.toFixed(4), // Confidence level (0-1)
        avgTestSharpe: avgTestSharpe.toFixed(4),
        numWindows: results.length,
        isRobust: dsr > 0.90 && (avgTestSharpe / avgTrainSharpe) > 0.5
    };
};
