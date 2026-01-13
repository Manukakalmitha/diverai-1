import { calculateHybridProbability } from './analysis.js';
import { calculateATR, findSupportResistance } from './technicalAnalysis.js';

const runTests = () => {
    console.log("--- Diver AI V4 Precision Validation ---");

    // Mock Data (20 points)
    const prices = [100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100, 102, 104, 105, 106, 107, 108, 107, 106, 105];
    const highs = prices.map(p => p * 1.01);
    const lows = prices.map(p => p * 0.99);

    console.log("\n1. Testing ATR Precision...");
    const atr = calculateATR(highs, lows, prices, 5);
    console.log(`ATR (Last): ${atr[atr.length - 1].toFixed(4)}`);
    if (atr[atr.length - 1] > 0) console.log("✅ ATR calculation successful.");

    console.log("\n2. Testing Support/Resistance Strength (V4)...");
    const sr = findSupportResistance(prices);
    console.log(`Support: ${sr.support}, Resistance: ${sr.resistance}`);
    console.log(`Strength - S: ${sr.strength.s}, R: ${sr.strength.r}`);
    if (sr.strength) console.log("✅ S/R Strength detection active.");

    console.log("\n3. Testing Hybrid Fusion Logic (V4)...");
    const weights = { omega: 0.4, alpha: 0.3, gamma: 0.3, iterations: 10 };
    const rsi = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75];

    // Bullish case
    const probBull = calculateHybridProbability(0.8, 'Bullish', { rsi }, weights, 0.98, 0.7, 0.02);
    console.log(`Bullish Probability: ${(probBull * 100).toFixed(2)}%`);

    // Bearish case
    const probBear = calculateHybridProbability(0.2, 'Bearish', { rsi: [70, 75, 80] }, weights, 0.98, 0.3, 0.02);
    console.log(`Bearish Probability: ${(probBear * 100).toFixed(2)}%`);

    if (probBull > 0.8 && probBear < 0.2) {
        console.log("✅ Fusion logic correctly discriminating direction.");
    }

    console.log("\n--- Validation Complete ---");
};

try {
    runTests();
} catch (e) {
    console.error("❌ Test Failed:", e.message);
}
