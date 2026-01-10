import * as tf from '@tensorflow/tfjs';
import { supabase } from './supabase';
import { calculateRSI, calculateMACD, detectPatterns } from './technicalAnalysis';

// Explicitly set backend
const initTf = async () => {
    try {
        await tf.setBackend('webgl');
        await tf.ready();
    } catch (e) {
        console.warn("WebGL failed, falling back to CPU. Performance will be reduced.");
        await tf.setBackend('cpu');
    }
};
initTf();

// --- Background Worker Integration ---
export const runBackgroundTraining = (data) => {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./brain.worker.js', import.meta.url), { type: 'module' });

        worker.onmessage = (e) => {
            const { type, data: result } = e.data;
            if (type === 'TRAIN_SUCCESS') {
                worker.terminate();
                resolve(result);
            } else if (type === 'ERROR') {
                worker.terminate();
                reject(new Error(result));
            }
        };

        worker.onerror = (err) => {
            worker.terminate();
            reject(err);
        };

        worker.postMessage({ type: 'TRAIN_AND_PREDICT', data });

        // Failsafe Timeout
        setTimeout(() => {
            worker.terminate();
            reject(new Error("Neural Training Timed Out (120s Limit)"));
        }, 120000);
    });
};

export const runBackgroundAssessment = (fullPrices, onProgress) => {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./brain.worker.js', import.meta.url), { type: 'module' });

        worker.onmessage = (e) => {
            const { type, data: result } = e.data;
            if (type === 'ASSESS_SUCCESS') {
                worker.terminate();
                resolve(result);
            } else if (type === 'PROGRESS') {
                if (onProgress) onProgress(result);
            } else if (type === 'ERROR') {
                worker.terminate();
                reject(new Error(result));
            }
        };

        worker.onerror = (err) => {
            worker.terminate();
            reject(err);
        };

        worker.postMessage({ type: 'ASSESS_ACCURACY', data: { fullPrices } });
    });
};


// Hyperparameters (Industry Upgraded)
export const WINDOW_SIZE = 30; // Increased from 14 for better context
const EPOCHS = 25; // Adjusted for deeper learning
const BATCH_SIZE = 32;
const FEATURES = 3; // Price, RSI, MACD Histogram

/**
 * Normalizes a pattern using "Zero-Base" (first candle as 0)
 * This is superior for fractal geometry preservation
 */
const zeroBaseNormalize = (arr, baseVal) => {
    return arr.map(v => (v - baseVal) / (baseVal || 1));
};

/**
 * Standard Z-Score for non-price indicators (RSI/MACD)
 */
const zScoreNormalize = (val, mean, std) => (val - mean) / (std || 1);

/**
 * Synthetic Data Augmentation
 * Simulates variations to reach "1M patterns" robustiness
 */
export const augmentPattern = (patternVector) => {
    // patternVector is [WINDOW_SIZE][FEATURES]
    return patternVector.map(step => {
        const noise = (Math.random() - 0.5) * 0.001; // subtle noise
        const scale = 0.998 + (Math.random() * 0.004); // subtle scale shift
        return [
            step[0] * scale + noise,
            step[1] * scale + noise,
            step[2] * scale + noise
        ];
    });
};

/**
 * Calculate stats for Z-Score normalization WITHOUT creating tensors
 */
export const calculateStats = (dataSeries) => {
    const { prices, rsi, macd } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length);
    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);

    return [p, r, m].map(arr => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length) || 1;
        return { mean, std };
    });
};

/**
 * Multivariate Data Preparation with Z-Score Normalization
 * Optimized for RAM using TypedArrays directly
 */
export const prepareData = (dataSeries, windowSize) => {
    const { prices, rsi, macd } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length);
    const stats = calculateStats(dataSeries);

    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);

    const count = minLen - windowSize;
    if (count <= 0) return { xs: tf.tensor3d([], [0, windowSize, FEATURES]), ys: tf.tensor2d([], [0, 1]), stats };

    const xData = [];
    const yData = [];

    for (let i = 0; i < count; i++) {
        const windowBasePrice = p[i]; // The first price in the window is our relative 0
        const pattern = [];

        for (let j = 0; j < windowSize; j++) {
            pattern.push([
                (p[i + j] - windowBasePrice) / (windowBasePrice || 1), // Zero-base price
                zScoreNormalize(r[i + j], stats[1].mean, stats[1].std),
                zScoreNormalize(m[i + j], stats[2].mean, stats[2].std)
            ]);
        }

        xData.push(pattern);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1));

        // AUGMENTATION (On-the-fly)
        // Add a noisy version of the same pattern to double the training set
        const augmented = augmentPattern(pattern);
        xData.push(augmented);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1) * (0.998 + Math.random() * 0.004));
    }

    const xs = tf.tensor3d(xData, [xData.length, windowSize, FEATURES]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);

    return { xs, ys, stats };
};

export const createModel = () => {
    const model = tf.sequential();

    // Layer 1: LSTM (Temporal Feature Extraction)
    model.add(tf.layers.lstm({
        units: 32, // Reduced from 64
        inputShape: [WINDOW_SIZE, FEATURES],
        returnSequences: true,
        recurrentInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));

    model.add(tf.layers.dropout({ rate: 0.2 }));

    // Layer 2: LSTM (Deep Pattern Detection)
    model.add(tf.layers.lstm({
        units: 16, // Reduced from 32
        returnSequences: false,
        recurrentInitializer: 'glorotUniform'
    }));

    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.dense({
        units: 16, // Matches LSTM 2
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));

    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: 'meanSquaredError' // Reverted for compatibility
    });

    return model;
};

export const trainModel = async (model, dataSeries, epochsOverride = null) => {
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

export const predictNextPrice = (model, lastWindowSeries, stats) => {
    // lastWindowSeries: { prices: [30], rsi: [30], macd: [30] }
    const basePrice = lastWindowSeries.prices[0];
    const inputData = [];

    for (let i = 0; i < WINDOW_SIZE; i++) {
        inputData.push([
            (lastWindowSeries.prices[i] - basePrice) / (basePrice || 1),
            zScoreNormalize(lastWindowSeries.rsi[i], stats[1].mean, stats[1].std),
            zScoreNormalize(lastWindowSeries.macd[i], stats[2].mean, stats[2].std)
        ]);
    }

    const inputTensor = tf.tensor3d([inputData], [1, WINDOW_SIZE, FEATURES]);
    const outputTensor = model.predict(inputTensor);
    const normOutput = outputTensor.dataSync()[0];

    inputTensor.dispose();
    outputTensor.dispose();

    // Denormalize using Zero-Base logic
    return normOutput * basePrice + basePrice;
};

/**
 * Cleanup model memory
 */
export const disposeModel = (model) => {
    if (model) model.dispose();
};

/**
 * Runs a rolling window backtest on the data.
 * @param {Array} prices - Full array of historical prices
 * @param {Function} onProgress - Callback(percent)
 * @returns {Object} - { accuracy (0-100), rmse, chartData: [{date, actual, predicted}] }
 */
export const assessModelAccuracy = async (fullPrices, onProgress) => {
    // Requires pre-calculated indicators
    const rsi = calculateRSI(fullPrices, 14);
    const macd = calculateMACD(fullPrices).histogram;
    const minLen = Math.min(fullPrices.length, rsi.length, macd.length);
    const prices = fullPrices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);

    const TEST_POINTS = 10;
    if (minLen < WINDOW_SIZE + TEST_POINTS + 10) {
        return { accuracy: 50, rmse: 0, predictions: [] };
    }

    const predictions = [];
    let hits = { neural: 0, pattern: 0, technical: 0 };

    // OPTIMIZATION: Create one model for the whole assessment
    const model = createModel();

    // 1. Initial bulk training on history
    const baseIdx = minLen - TEST_POINTS;
    const baseSeries = {
        prices: prices.slice(0, baseIdx),
        rsi: r.slice(0, baseIdx),
        macd: m.slice(0, baseIdx)
    };

    if (onProgress) onProgress(10);
    const { stats } = await trainModel(model, baseSeries, 12);
    if (onProgress) onProgress(40);

    // 2. Rolling window update using trainOnBatch
    for (let i = 0; i < TEST_POINTS; i++) {
        const targetIdx = baseIdx + i;
        const actual = prices[targetIdx];
        const prevClose = prices[targetIdx - 1];
        const actualDir = actual > prevClose ? 1 : -1;

        // A. Neural Prediction
        const predicted = tf.tidy(() => {
            const lastWindowSeries = {
                prices: prices.slice(targetIdx - WINDOW_SIZE, targetIdx),
                rsi: r.slice(targetIdx - WINDOW_SIZE, targetIdx),
                macd: m.slice(targetIdx - WINDOW_SIZE, targetIdx)
            };
            return predictNextPrice(model, lastWindowSeries, stats);
        });
        const neuralDir = predicted > prevClose ? 1 : -1;
        if (neuralDir === actualDir) hits.neural++;

        // B. Pattern Prediction (using technicalAnalysis.js logic)
        const currentPrices = prices.slice(0, targetIdx);
        const pattern = detectPatterns(currentPrices);
        let patternDir = 0;
        if (pattern.sentiment === 'Bullish') patternDir = 1;
        else if (pattern.sentiment === 'Bearish') patternDir = -1;
        if (patternDir === actualDir) hits.pattern++;

        // C. Technical Prediction (RSI based)
        const currentRSI = r[targetIdx - 1];
        let techDir = 0;
        if (currentRSI < 40) techDir = 1;      // Bullish bias
        else if (currentRSI > 60) techDir = -1; // Bearish bias
        if (techDir === actualDir) hits.technical++;

        predictions.push({ step: i + 1, actual, predicted, isCorrect: neuralDir === actualDir });

        // Update model weights for next step
        tf.tidy(() => {
            const basePrice = prices[targetIdx - WINDOW_SIZE];
            const x = [];
            for (let j = 0; j < WINDOW_SIZE; j++) {
                const idx = (targetIdx - WINDOW_SIZE) + j;
                x.push([
                    (prices[idx] - basePrice) / (basePrice || 1),
                    zScoreNormalize(r[idx], stats[1].mean, stats[1].std),
                    zScoreNormalize(m[idx], stats[2].mean, stats[2].std)
                ]);
            }
            const xt = tf.tensor3d([x], [1, WINDOW_SIZE, FEATURES]);
            const yt = tf.tensor2d([[(actual - basePrice) / (basePrice || 1)]], [1, 1]);
            model.trainOnBatch(xt, yt);
        });

        if (onProgress) onProgress(40 + ((i + 1) / TEST_POINTS) * 60);
    }

    model.dispose();

    // Calculate Recommended Weights (Normalized 0.0 to 1.0)
    const totalHits = hits.neural + hits.pattern + hits.technical || 1;
    const recommendedWeights = {
        omega: Math.max(0.2, (hits.neural / totalHits)),
        alpha: Math.max(0.15, (hits.pattern / totalHits)),
        gamma: Math.max(0.15, (hits.technical / totalHits))
    };

    return {
        accuracy: ((hits.neural / TEST_POINTS) * 100).toFixed(1),
        hits,
        recommendedWeights,
        predictions
    };
};

/**
 * Saves model to Supabase Storage/DB
 */
export const saveGlobalModel = async (user, model, name, accuracy) => {
    if (!user) return;

    try {
        const modelArtifacts = await model.save(tf.io.withSaveHandler(async (artifacts) => artifacts));

        // Convert ArrayBuffer to Base64 for efficient JSON storage
        if (modelArtifacts.weightData instanceof ArrayBuffer) {
            const bytes = new Uint8Array(modelArtifacts.weightData);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            modelArtifacts.weightData = btoa(binary);
        }

        const { error } = await supabase
            .from('neural_models')
            .upsert([{
                user_id: user.id,
                name: name,
                model_json: modelArtifacts,
                accuracy: accuracy,
                created_at: new Date().toISOString()
            }], { onConflict: 'user_id, name' });

        if (error) throw error;
        return true;
    } catch (err) {
        console.error("Cloud Model Save Error:", err);
        return false;
    }
};

/**
 * Saves raw model artifacts directly to Supabase
 */
export const saveGlobalModelArtifacts = async (user, modelArtifacts, name, accuracy) => {
    if (!user) return;

    try {
        // Convert ArrayBuffer to Base64 for efficient JSON storage if needed
        if (modelArtifacts.weightData instanceof ArrayBuffer) {
            const bytes = new Uint8Array(modelArtifacts.weightData);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            modelArtifacts.weightData = btoa(binary);
        }

        const { error } = await supabase
            .from('neural_models')
            .upsert([{
                user_id: user.id,
                name: name,
                model_json: modelArtifacts,
                accuracy: accuracy,
                created_at: new Date().toISOString()
            }], { onConflict: 'user_id, name' });

        if (error) throw error;
        return true;
    } catch (err) {
        console.error("Cloud Artifact Save Error:", err);
        return false;
    }
};

/**
 * Loads model from Supabase
 */
export const loadGlobalModel = async (user, name) => {
    if (!user) return null;

    try {
        const { data, error } = await supabase
            .from('neural_models')
            .select('model_json')
            .eq('user_id', user.id)
            .eq('name', name)
            .maybeSingle(); // Better handling for empty states

        if (error || !data) return null;

        const modelArtifacts = data.model_json;
        // Restore ArrayBuffer from Base64 or Legacy Array
        if (typeof modelArtifacts.weightData === 'string') {
            const binary = atob(modelArtifacts.weightData);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            modelArtifacts.weightData = bytes.buffer;
        } else if (Array.isArray(modelArtifacts.weightData)) {
            modelArtifacts.weightData = new Uint8Array(modelArtifacts.weightData).buffer;
        }

        const model = await tf.loadLayersModel(tf.io.fromMemory(modelArtifacts));

        // SHAPE VERIFICATION: Ensure the loaded model matches the current WINDOW_SIZE
        const inputShape = model.inputs[0].shape; // [null, windowSize, features]
        if (inputShape[1] !== WINDOW_SIZE) {
            console.warn(`[Brain] Model shape mismatch detected (${inputShape[1]} vs ${WINDOW_SIZE}). Discarding legacy model.`);
            model.dispose();
            return null;
        }

        return model;
    } catch (err) {
        console.error("Cloud Model Load Error:", err);
        return null;
    }
};
