import { detectPrice, detectTicker } from './marketData.js';

const testOcrResilience = () => {
    console.log("--- Diver AI V5.2 OCR Resilience Test ---");

    const samples = [
        {
            name: "High-value BTC with commas",
            text: "CRYPTO O H L 92,520.00 92,480.00",
            expectedTicker: "BTC",
            expectedPrice: 92520
        },
        {
            name: "ETH with typical range",
            text: "Market ETH/USD Price 2,450.75",
            expectedTicker: "ETH",
            expectedPrice: 2450.75
        },
        {
            name: "Messy TradingView Header",
            text: "J 13, 2026 13:58 UTC / U.S. 1 - CRYPTO O H L 102,400",
            expectedTicker: "BTC", // Price > 90k + CRYPTO context
            expectedPrice: 102400
        },
        {
            name: "Stock with Dots and Commas",
            text: "Apple Inc. (AAPL) 192.45 Volume 1,234,567",
            expectedTicker: "AAPL",
            expectedPrice: 192.45
        }
    ];

    samples.forEach(s => {
        console.log(`\nTesting sample: ${s.name}`);
        const ticker = detectTicker(s.text);
        const price = detectPrice(s.text);

        const tickerOk = ticker === s.expectedTicker;
        const priceOk = Math.abs(price - s.expectedPrice) < 0.01;

        console.log(`  Ticker: [${ticker}] ${tickerOk ? '✅' : '❌ (Expected ' + s.expectedTicker + ')'}`);
        console.log(`  Price:  [${price}]  ${priceOk ? '✅' : '❌ (Expected ' + s.expectedPrice + ')'}`);
    });

    console.log("\n--- Test Complete ---");
};

testOcrResilience();
