import * as tf from '@tensorflow/tfjs';
import { supabase } from './supabase.js';
import { calculateRSI, calculateMACD, detectPatterns, calculateATR } from './technicalAnalysis.js';
import { calculateROC, calculateRollingVolatility } from './featureEngineering.js';

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
// Training timeout: 5 minutes (300s) - increased from 2 min for mobile devices
const TRAINING_TIMEOUT_MS = 300000;

export const runBackgroundTraining = (data) => {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./brain.worker.js', import.meta.url), { type: 'module' });
        let isSettled = false;

        // Timeout handler to prevent indefinite hangs on slow devices
        const timeoutId = setTimeout(() => {
            if (!isSettled) {
                isSettled = true;
                worker.terminate();
                reject(new Error(`Neural Training Timed Out (${TRAINING_TIMEOUT_MS / 1000}s Limit). Try again or use Fast Mode.`));
            }
        }, TRAINING_TIMEOUT_MS);

        worker.onmessage = (e) => {
            const { type, data: result } = e.data;
            if (type === 'TRAIN_SUCCESS') {
                if (!isSettled) {
                    isSettled = true;
                    clearTimeout(timeoutId);
                    worker.terminate();
                    resolve(result);
                }
            } else if (type === 'ERROR') {
                if (!isSettled) {
                    isSettled = true;
                    clearTimeout(timeoutId);
                    worker.terminate();
                    reject(new Error(result));
                }
            }
        };

        worker.onerror = (err) => {
            if (!isSettled) {
                isSettled = true;
                clearTimeout(timeoutId);
                worker.terminate();
                reject(err);
            }
        };

        if (data.historicalPrices?.length < WINDOW_SIZE + 10) {
            clearTimeout(timeoutId);
            reject(new Error(`Insufficient price history for neural training. Got ${data.historicalPrices?.length || 0} bars, need ${WINDOW_SIZE + 10}.`));
            return;
        }

        worker.postMessage({ type: 'TRAIN_AND_PREDICT', data });
    });
};

export const runBackgroundAssessment = (fullPrices, onProgress) => {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./brain.worker.js', import.meta.url), { type: 'module' });
        let isSettled = false;

        // Assessment timeout: 5 minutes
        const timeoutId = setTimeout(() => {
            if (!isSettled) {
                isSettled = true;
                worker.terminate();
                reject(new Error(`Neural Assessment Timed Out (${TRAINING_TIMEOUT_MS / 1000}s Limit).`));
            }
        }, TRAINING_TIMEOUT_MS);

        worker.onmessage = (e) => {
            const { type, data: result } = e.data;
            if (type === 'ASSESS_SUCCESS') {
                if (!isSettled) {
                    isSettled = true;
                    clearTimeout(timeoutId);
                    worker.terminate();
                    resolve(result);
                }
            } else if (type === 'PROGRESS') {
                if (onProgress) onProgress(result);
            } else if (type === 'ERROR') {
                if (!isSettled) {
                    isSettled = true;
                    clearTimeout(timeoutId);
                    worker.terminate();
                    reject(new Error(result));
                }
            }
        };

        worker.onerror = (err) => {
            if (!isSettled) {
                isSettled = true;
                clearTimeout(timeoutId);
                worker.terminate();
                reject(err);
            }
        };

        worker.postMessage({ type: 'ASSESS_ACCURACY', data: { fullPrices } });
    });
};


// Hyperparameters (Industry Upgraded)
// Hyperparameters (Industry Upgraded - V4 Precision)
export const WINDOW_SIZE = 45;
const EPOCHS = 35;
const BATCH_SIZE = 32;
const FEATURES = 6; // Price, RSI, MACD Histogram, ATR, ROC, Volatility

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
 * Robust Scaling (Median/IQR) for outlier-prone features (Vol, ROC)
 */
const robustNormalize = (val, median, iqr) => (val - median) / (iqr || 1);

/**
 * Calculate Robust Stats (Median, IQR)
 */
export const calculateRobustStats = (arr) => {
    const sorted = [...arr].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1 || 1;
    return { median, iqr };
};
export const augmentPattern = (patternVector) => {
    // patternVector is [WINDOW_SIZE][FEATURES]
    return patternVector.map(step => {
        const noise = (Math.random() - 0.5) * 0.0005; // lower noise
        const scale = 0.999 + (Math.random() * 0.002); // tighter scale
        return [
            step[0] * scale + noise,
            step[1] * scale + noise,
            step[2] * scale + noise,
            step[3] * scale + noise,
            step[4] * scale + noise,
            step[5] * scale + noise
        ];
    });
};

/**
 * Calculate stats for Z-Score normalization WITHOUT creating tensors
 */
export const calculateStats = (dataSeries) => {
    const { prices, rsi, macd, atr, roc, vol } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length, (atr?.length || 0), (roc?.length || 0), (vol?.length || 0));

    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);
    const a = atr.slice(-minLen);
    const ro = roc.slice(-minLen);
    const v = vol.slice(-minLen);

    // [0-Price (Base Only), 1-RSI (Z), 2-MACD (Z), 3-ATR (Z), 4-ROC (Robust), 5-Vol (Robust)]
    const stats = [];

    // Z-Score for 1-3
    [r, m, a].forEach(arr => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length) || 1;
        stats.push({ mean, std });
    });

    // Robust for 4-5
    [ro, v].forEach(arr => {
        const { median, iqr } = calculateRobustStats(arr);
        stats.push({ median, iqr });
    });

    return [{ base: p[0] }, ...stats];
};

/**
 * Multivariate Data Preparation with Z-Score Normalization
 */
export const prepareData = (dataSeries, windowSize) => {
    const { prices, rsi, macd, atr, roc, vol } = dataSeries;
    const minLen = Math.min(prices.length, rsi.length, macd.length, (atr?.length || 0), (roc?.length || 0), (vol?.length || 0));
    const stats = calculateStats(dataSeries);

    const p = prices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);
    const a = atr.slice(-minLen);
    const ro = roc.slice(-minLen);
    const v = vol.slice(-minLen);

    const count = minLen - windowSize;
    if (count <= 0) return { xs: tf.tensor3d([], [0, windowSize, FEATURES]), ys: tf.tensor2d([], [0, 1]), stats };

    const xData = [];
    const yData = [];

    for (let i = 0; i < count; i++) {
        const windowBasePrice = p[i]; // The first price in the window is our relative 0
        const pattern = [];

        for (let j = 0; j < windowSize; j++) {
            pattern.push([
                (p[i + j] - windowBasePrice) / (windowBasePrice || 1),
                zScoreNormalize(r[i + j], stats[1].mean, stats[1].std),
                zScoreNormalize(m[i + j], stats[2].mean, stats[2].std),
                zScoreNormalize(a[i + j], stats[3].mean, stats[3].std),
                robustNormalize(ro[i + j], stats[4].median, stats[4].iqr), // Robust
                robustNormalize(v[i + j], stats[5].median, stats[5].iqr)   // Robust
            ]);
        }

        xData.push(pattern);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1));

        // AUGMENTATION (On-the-fly)
        const augmented = augmentPattern(pattern);
        xData.push(augmented);
        yData.push((p[i + windowSize] - windowBasePrice) / (windowBasePrice || 1) * (0.999 + Math.random() * 0.002));
    }

    const xs = tf.tensor3d(xData, [xData.length, windowSize, FEATURES]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);

    return { xs, ys, stats };
};

export const createModel = () => {
    const input = tf.input({ shape: [WINDOW_SIZE, FEATURES] });

    // Layer 1: LSTM (Temporal Feature Extraction) - Return sequences for Attention
    const lstm1 = tf.layers.lstm({
        units: 64,
        returnSequences: true,
        recurrentInitializer: 'glorotUniform',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.005 })
    }).apply(input);

    const dropout1 = tf.layers.dropout({ rate: 0.2 }).apply(lstm1);

    // --- ATTENTION MECHANISM ---
    // 1. Calculate energy/alignment scores
    const attentionEnergy = tf.layers.dense({
        units: 64,
        activation: 'tanh',
        name: 'attention_energy'
    }).apply(dropout1);

    // 2. Calculate attention weights (Softmax over time)
    const attentionWeights = tf.layers.dense({
        units: 1,
        activation: 'softmax',
        name: 'attention_weights'
    }).apply(attentionEnergy);

    // 3. Apply weights to sequence
    const weightedSequence = tf.layers.multiply().apply([dropout1, attentionWeights]);

    // 4. Global Average Pooling (Context Vector)
    const contextVector = tf.layers.globalAveragePooling1d().apply(weightedSequence);
    // ---------------------------

    const batchNorm1 = tf.layers.batchNormalization().apply(contextVector);

    // Layer 3: Interpretation
    const dense1 = tf.layers.dense({
        units: 32,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.005 })
    }).apply(batchNorm1);

    const dense2 = tf.layers.dense({
        units: 16,
        activation: 'relu'
    }).apply(dense1);

    const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(dense2);

    const model = tf.model({ inputs: input, outputs: output });

    model.compile({
        optimizer: tf.train.adam(0.0015), // Slightly lower LR for attention stability
        loss: 'meanSquaredError'
    });

    return model;
};

export const trainModel = async (model, dataSeries, epochsOverride = null) => {
    if (dataSeries.prices.length < WINDOW_SIZE + 10) return null;

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
    const basePrice = lastWindowSeries.prices[0];
    const inputData = [];

    for (let i = 0; i < WINDOW_SIZE; i++) {
        inputData.push([
            (lastWindowSeries.prices[i] - basePrice) / (basePrice || 1),
            zScoreNormalize(lastWindowSeries.rsi[i], stats[1].mean, stats[1].std),
            zScoreNormalize(lastWindowSeries.macd[i], stats[2].mean, stats[2].std),
            zScoreNormalize(lastWindowSeries.atr[i], stats[3].mean, stats[3].std),
            robustNormalize(lastWindowSeries.roc[i], stats[4].median, stats[4].iqr),
            robustNormalize(lastWindowSeries.vol[i], stats[5].median, stats[5].iqr)
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
 */
export const assessModelAccuracy = async (fullPrices, onProgress) => {
    // Requires pre-calculated indicators
    const rsi = calculateRSI(fullPrices, 14);
    const macd = calculateMACD(fullPrices).histogram;
    const atr = calculateATR(fullPrices, null, fullPrices, 14);
    const roc = calculateROC(fullPrices, 10);
    const vol = calculateRollingVolatility(fullPrices, 20);

    const minLen = Math.min(fullPrices.length, rsi.length, macd.length, atr.length, roc.length, vol.length);
    const prices = fullPrices.slice(-minLen);
    const r = rsi.slice(-minLen);
    const m = macd.slice(-minLen);
    const a = atr.slice(-minLen);
    const ro = roc.slice(-minLen);
    const v = vol.slice(-minLen);

    const TEST_POINTS = 10;
    if (minLen < WINDOW_SIZE + TEST_POINTS + 10) {
        return { accuracy: 50, rmse: 0, predictions: [] };
    }

    const predictions = [];
    let hits = { neural: 0, pattern: 0, technical: 0 };

    const model = createModel();

    // 1. Initial bulk training on history
    const baseIdx = minLen - TEST_POINTS;
    const baseSeries = {
        prices: prices.slice(0, baseIdx),
        rsi: r.slice(0, baseIdx),
        macd: m.slice(0, baseIdx),
        atr: a.slice(0, baseIdx),
        roc: ro.slice(0, baseIdx),
        vol: v.slice(0, baseIdx)
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
                macd: m.slice(targetIdx - WINDOW_SIZE, targetIdx),
                atr: a.slice(targetIdx - WINDOW_SIZE, targetIdx),
                roc: ro.slice(targetIdx - WINDOW_SIZE, targetIdx),
                vol: v.slice(targetIdx - WINDOW_SIZE, targetIdx)
            };
            return predictNextPrice(model, lastWindowSeries, stats);
        });
        const neuralDir = predicted > prevClose ? 1 : -1;
        if (neuralDir === actualDir) hits.neural++;

        // B. Pattern Prediction
        const currentPrices = prices.slice(0, targetIdx);
        const pattern = detectPatterns(currentPrices);
        let patternDir = 0;
        if (pattern.sentiment === 'Bullish') patternDir = 1;
        else if (pattern.sentiment === 'Bearish') patternDir = -1;
        if (patternDir === actualDir) hits.pattern++;

        // C. Technical Prediction
        const currentRSI = r[targetIdx - 1];
        let techDir = 0;
        if (currentRSI < 40) techDir = 1;
        else if (currentRSI > 60) techDir = -1;
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
                    zScoreNormalize(m[idx], stats[2].mean, stats[2].std),
                    zScoreNormalize(a[idx], stats[3].mean, stats[3].std),
                    zScoreNormalize(ro[idx], stats[4].mean, stats[4].std),
                    zScoreNormalize(v[idx], stats[5].mean, stats[5].std)
                ]);
            }
            const xt = tf.tensor3d([x], [1, WINDOW_SIZE, FEATURES]);
            const yt = tf.tensor2d([[(actual - basePrice) / (basePrice || 1)]], [1, 1]);
            model.trainOnBatch(xt, yt);
        });

        if (onProgress) onProgress(40 + ((i + 1) / TEST_POINTS) * 60);
    }

    model.dispose();

    // Calculate Recommended Weights 
    const totalHits = hits.neural + hits.pattern + hits.technical || 1;
    const recommendedWeights = {
        omega: Math.max(0.2, (hits.neural / totalHits)),
        alpha: Math.max(0.15, (hits.pattern / totalHits)),
        gamma: Math.max(0.15, (hits.technical / totalHits))
    };

    // Calculate Brier Score 
    const brierScore = (predictions.reduce((acc, p) => {
        const prob = p.predicted > prices[baseIdx + p.step - 2] ? 0.75 : 0.25;
        const outcome = p.actual > prices[baseIdx + p.step - 2] ? 1 : 0;
        return acc + Math.pow(prob - outcome, 2);
    }, 0) / TEST_POINTS).toFixed(4);

    return {
        accuracy: ((hits.neural / TEST_POINTS) * 100).toFixed(1),
        hits,
        recommendedWeights,
        predictions,
        metrics: {
            brierScore,
            rmse: (Math.sqrt(predictions.reduce((acc, p) => acc + Math.pow(p.actual - p.predicted, 2), 0) / TEST_POINTS)).toFixed(2)
        }
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
        // COMMUNITY BRAIN V1: Search for the LATEST, specified model (ticker) from ANY user
        // We order by created_at desc to get the freshest brain
        const { data, error } = await supabase
            .from('neural_models')
            .select('model_json')
            .eq('name', name)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();

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
        if (inputShape[1] !== WINDOW_SIZE || inputShape[2] !== FEATURES) {
            console.warn(`[Brain] Model shape mismatch detected (${inputShape[1]}x${inputShape[2]} vs ${WINDOW_SIZE}x${FEATURES}). Discarding legacy model.`);
            model.dispose();
            return null;
        }

        return model;
    } catch (err) {
        console.error("Cloud Model Load Error:", err);
        return null;
    }
};
