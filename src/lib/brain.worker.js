import * as tf from '@tensorflow/tfjs';
import { calculateRSI, calculateMACD, detectPatterns, calculateATR } from './technicalAnalysis.js';

// Hyperparameters (Industry Upgraded - V4 Precision)
const WINDOW_SIZE = 45;
const EPOCHS = 35;
const BATCH_SIZE = 32;
const FEATURES = 4; // Price, RSI, MACD Histogram, ATR

const zScoreNormalize = (val, mean, std) => (val - mean) / (std || 1);

const augmentPattern = (patternVector) => {
    return patternVector.map(step => {
        const noise = (Math.random() - 0.5) * 0.0005; // Slightly lower noise for more precision
        const scale = 0.999 + (Math.random() * 0.002);
        return [
            step[0] * scale + noise,
            step[1] * scale + noise,
            step[2] * scale + noise,
            step[3] * scale + noise
        ];
    });
};

/**
 * Calculate stats for Z-Score normalization WITHOUT creating tensors
 */
const calculateStats = (dataSeries) => {
    const { prices, rsi, macd, atr } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length, (atr?.length || 0));
    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);
    const a = atr.slice(-minLen);

    return [p, r, m, a].map(arr => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length) || 1;
        return { mean, std };
    });
};

/**
 * Multivariate Data Preparation with Z-Score Normalization
 */
const prepareData = (dataSeries, windowSize) => {
    const { prices, rsi, macd, atr } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length, atr.length);
    const stats = calculateStats(dataSeries);

    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);
    const a = atr.slice(-minLen);

    const count = minLen - windowSize;
    if (count <= 0) return { xs: tf.tensor3d([], [0, windowSize, FEATURES]), ys: tf.tensor2d([], [0, 1]), stats };

    const xData = [];
    const yData = [];

    for (let i = 0; i < count; i++) {
        const windowBasePrice = p[i];
        const pattern = [];

        for (let j = 0; j < windowSize; j++) {
            pattern.push([
                (p[i + j] - windowBasePrice) / (windowBasePrice || 1),
                zScoreNormalize(r[i + j], stats[1].mean, stats[1].std),
                zScoreNormalize(m[i + j], stats[2].mean, stats[2].std),
                zScoreNormalize(a[i + j], stats[3].mean, stats[3].std)
            ]);
        }

        xData.push(pattern);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1));

        // AUGMENTATION
        const augmented = augmentPattern(pattern);
        xData.push(augmented);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1) * (0.999 + Math.random() * 0.002));
    }

    const xs = tf.tensor3d(xData, [xData.length, windowSize, FEATURES]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);

    return { xs, ys, stats };
};

const createModel = () => {
    const model = tf.sequential();

    // Layer 1: LSTM (Temporal Feature Extraction)
    model.add(tf.layers.lstm({
        units: 64, // Increased for deeper feature extraction
        inputShape: [WINDOW_SIZE, FEATURES],
        returnSequences: true,
        recurrentInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.005 })
    }));

    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Layer 2: LSTM (Deep Pattern Detection)
    model.add(tf.layers.lstm({
        units: 32,
        returnSequences: false,
        recurrentInitializer: 'glorotUniform'
    }));

    model.add(tf.layers.batchNormalization());

    // Layer 3: Dense interpretation layer (V4 ADDITION)
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.005 })
    }));

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.002), // Lower learning rate for precision
        loss: 'meanSquaredError'
    });

    return model;
};

const trainModel = async (model, dataSeries, epochsOverride = null) => {
    if (dataSeries.prices.length < WINDOW_SIZE + 20) return null;
    const { xs, ys, stats } = prepareData(dataSeries, WINDOW_SIZE);
    await model.fit(xs, ys, {
        epochs: epochsOverride || EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle: true,
        verbose: 0
    });
    xs.dispose();
    ys.dispose();
    return { stats };
};

const predictNextPrice = (model, lastWindowSeries, stats) => {
    const basePrice = lastWindowSeries.prices[0];
    const inputData = [];

    for (let i = 0; i < WINDOW_SIZE; i++) {
        inputData.push([
            (lastWindowSeries.prices[i] - basePrice) / (basePrice || 1),
            zScoreNormalize(lastWindowSeries.rsi[i], stats[1].mean, stats[1].std),
            zScoreNormalize(lastWindowSeries.macd[i], stats[2].mean, stats[2].std),
            zScoreNormalize(lastWindowSeries.atr[i], stats[3].mean, stats[3].std)
        ]);
    }

    const inputTensor = tf.tensor3d([inputData], [1, WINDOW_SIZE, FEATURES]);
    const outputTensor = model.predict(inputTensor);
    const normOutput = outputTensor.dataSync()[0];

    inputTensor.dispose();
    outputTensor.dispose();

    return normOutput * basePrice + basePrice;
};

const assessModelAccuracy = async (fullPrices, reportProgress) => {
    const rsi = calculateRSI(fullPrices, 14);
    const macdResult = calculateMACD(fullPrices);
    const macdHist = macdResult.histogram;
    const atr = calculateATR(fullPrices, null, fullPrices, 14); // Using improved ATR

    const minLen = Math.min(fullPrices.length, rsi.length, macdHist.length, atr.length);
    const prices = fullPrices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macdHist.slice(-minLen);
    const a = atr.slice(-minLen);

    const TEST_POINTS = 10;
    if (minLen < WINDOW_SIZE + TEST_POINTS + 10) {
        return {
            accuracy: 50,
            hits: { neural: 0, pattern: 0, technical: 0 },
            recommendedWeights: { omega: 0.5, alpha: 0.3, gamma: 0.2 },
            predictions: []
        };
    }

    const predictions = [];
    let hits = { neural: 0, pattern: 0, technical: 0 };
    const model = createModel();

    const baseIdx = minLen - TEST_POINTS;
    const baseSeries = {
        prices: prices.slice(0, baseIdx),
        rsi: r.slice(0, baseIdx),
        macd: m.slice(0, baseIdx),
        atr: a.slice(0, baseIdx)
    };

    if (reportProgress) reportProgress(10);
    const { stats } = await trainModel(model, baseSeries, 12);
    if (reportProgress) reportProgress(40);

    for (let i = 0; i < TEST_POINTS; i++) {
        const targetIdx = baseIdx + i;
        const actual = prices[targetIdx];
        const prevClose = prices[targetIdx - 1];
        const actualDir = actual > prevClose ? 1 : -1;

        const predicted = tf.tidy(() => {
            const lastWindowSeries = {
                prices: prices.slice(targetIdx - WINDOW_SIZE, targetIdx),
                rsi: r.slice(targetIdx - WINDOW_SIZE, targetIdx),
                macd: m.slice(targetIdx - WINDOW_SIZE, targetIdx),
                atr: a.slice(targetIdx - WINDOW_SIZE, targetIdx)
            };
            return predictNextPrice(model, lastWindowSeries, stats);
        });
        const neuralDir = predicted > prevClose ? 1 : -1;
        if (neuralDir === actualDir) hits.neural++;

        const currentPrices = prices.slice(0, targetIdx);
        const pattern = detectPatterns(currentPrices);
        let patternDir = 0;
        if (pattern.sentiment === 'Bullish') patternDir = 1;
        else if (pattern.sentiment === 'Bearish') patternDir = -1;
        if (patternDir === actualDir) hits.pattern++;

        const currentRSI = r[targetIdx - 1];
        let techDir = 0;
        if (currentRSI < 40) techDir = 1; else if (currentRSI > 60) techDir = -1;
        if (techDir === actualDir) hits.technical++;

        predictions.push({ step: i + 1, actual, predicted, isCorrect: neuralDir === actualDir });

        tf.tidy(() => {
            const basePrice = prices[targetIdx - WINDOW_SIZE];
            const x = [];
            for (let j = 0; j < WINDOW_SIZE; j++) {
                const idx = (targetIdx - WINDOW_SIZE) + j;
                x.push([
                    (prices[idx] - basePrice) / (basePrice || 1),
                    zScoreNormalize(r[idx], stats[1].mean, stats[1].std),
                    zScoreNormalize(m[idx], stats[2].mean, stats[2].std),
                    zScoreNormalize(a[idx], stats[3].mean, stats[3].std)
                ]);
            }
            const xt = tf.tensor3d([x], [1, WINDOW_SIZE, FEATURES]);
            const yt = tf.tensor2d([[(actual - basePrice) / (basePrice || 1)]], [1, 1]);
            model.trainOnBatch(xt, yt);
        });

        if (reportProgress) reportProgress(40 + ((i + 1) / TEST_POINTS) * 60);
    }

    const recommendedWeights = {
        omega: Math.max(0.2, (hits.neural / (hits.neural + hits.pattern + hits.technical || 1))),
        alpha: Math.max(0.15, (hits.pattern / (hits.neural + hits.pattern + hits.technical || 1))),
        gamma: Math.max(0.15, (hits.technical / (hits.neural + hits.pattern + hits.technical || 1)))
    };

    const artifacts = await model.save(tf.io.withSaveHandler(async (art) => art));

    // Convert ArrayBuffer to Base64 for efficient transfer and storage
    if (artifacts.weightData instanceof ArrayBuffer) {
        const bytes = new Uint8Array(artifacts.weightData);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        artifacts.weightData = btoa(binary);
    }

    model.dispose();

    return {
        accuracy: ((hits.neural / TEST_POINTS) * 100).toFixed(1),
        hits,
        recommendedWeights,
        predictions,
        modelArtifacts: artifacts
    };
};

// Message Handler
self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        if (type === 'TRAIN_AND_PREDICT') {
            const { ticker, historicalPrices, rsi, macdHist, atr } = data;
            const model = createModel();
            const dataSeries = { prices: historicalPrices, rsi, macd: macdHist, atr };

            const trainResult = await trainModel(model, dataSeries);
            if (!trainResult) throw new Error(`Neural core training rejected: Insufficient data (${historicalPrices.length} bars)`);

            const lastWindow = {
                prices: historicalPrices.slice(-WINDOW_SIZE),
                rsi: rsi.slice(-WINDOW_SIZE),
                macd: macdHist.slice(-WINDOW_SIZE),
                atr: atr.slice(-WINDOW_SIZE)
            };

            const predictedPrice = predictNextPrice(model, lastWindow, trainResult.stats);
            const modelArtifacts = await model.save(tf.io.withSaveHandler(async (art) => art));

            // Convert ArrayBuffer to Base64 for efficient transfer and storage
            if (modelArtifacts.weightData instanceof ArrayBuffer) {
                const bytes = new Uint8Array(modelArtifacts.weightData);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                modelArtifacts.weightData = btoa(binary);
            }

            self.postMessage({
                type: 'TRAIN_SUCCESS',
                data: {
                    predictedPrice,
                    stats: trainResult.stats,
                    modelArtifacts
                }
            });
            model.dispose();

        } else if (type === 'ASSESS_ACCURACY') {
            const { fullPrices } = data;
            const result = await assessModelAccuracy(fullPrices, (percent) => {
                self.postMessage({ type: 'PROGRESS', data: percent });
            });
            self.postMessage({ type: 'ASSESS_SUCCESS', data: result });
        }
    } catch (err) {
        self.postMessage({ type: 'ERROR', data: err.message });
    }
};
