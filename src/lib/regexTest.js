
const COIN_MAP = { 'BTC': 'bitcoin', 'ETH': 'ethereum' };
const STOCK_MAP = { 'AAPL': 'apple' };

const detectPrice = (text) => {
    if (!text) return null;
    const decimalMatches = text.match(/\b\d{1,3}(?:[.,]\d{3})*[.,]\d{1,8}\b/g) || [];
    const complexIntMatches = text.match(/\b\d{1,3}(?:,\d{3})+\b/g) || [];
    const simpleIntMatches = text.match(/\b\d{4,7}\b/g) || [];

    const allMatches = [...decimalMatches, ...complexIntMatches, ...simpleIntMatches];
    console.log(`    [detectPrice] Matches found: ${JSON.stringify(allMatches)}`);

    const candidates = allMatches.map(m => {
        const cleaned = m.replace(/,/g, '');
        const parts = cleaned.split('.');
        if (parts.length > 2) {
            const decimal = parts.pop();
            return parseFloat(parts.join('') + '.' + decimal);
        }
        return parseFloat(cleaned);
    }).filter(n => {
        if (n >= 2000 && n <= 3000) return false;
        return n > 0.0001 && n < 20000000;
    });

    const sorted = candidates.sort((a, b) => {
        const aInTypicalRange = a > 0.1 && a < 150000;
        const bInTypicalRange = b > 0.1 && b < 150000;
        if (aInTypicalRange && !bInTypicalRange) return -1;
        if (!aInTypicalRange && bInTypicalRange) return 1;
        const aHasDecimal = a % 1 !== 0;
        const bHasDecimal = b % 1 !== 0;
        if (aHasDecimal && !bHasDecimal) return -1;
        if (!aHasDecimal && bHasDecimal) return 1;
        return a - b;
    });

    console.log(`    [detectPrice] Candidates: ${JSON.stringify(sorted)}`);
    return sorted[0] || null;
};

const detectTicker = (text) => {
    if (!text) return null;
    const upper = text.toUpperCase();
    console.log(`    [detectTicker] Checking context. CRYPTO: ${upper.includes('CRYPTO')}, O H L: ${upper.includes('O H L')}`);

    const blacklist = ['VOL', 'USD', 'USDT', 'UTC', 'CRYPTO', 'O H L'];
    const words = upper.split(/[^A-Z0-9]/).filter(w => w.length >= 2);
    for (const ticker of Object.keys(COIN_MAP)) {
        if (words.includes(ticker)) return ticker;
    }

    if ((upper.includes('CRYPTO') || upper.includes('O H L'))) {
        const price = detectPrice(text);
        console.log(`    [detectTicker] Context match found. Price detected: ${price}`);
        if (price > 40000 && price < 150000) return 'BTC';
        if (price > 1500 && price < 10000) return 'ETH';
    }
    return null;
};

const samples = [
    {
        name: "BTC with commas",
        text: "CRYPTO O H L 92,520.00 92,480.00",
        expectedTicker: "BTC",
        expectedPrice: 92520
    }
];

samples.forEach(s => {
    console.log(`\n>>> START TEST: ${s.name}`);
    const ticker = detectTicker(s.text);
    const price = detectPrice(s.text);
    console.log(`<<< END TEST. Results: Ticker=${ticker}, Price=${price}`);
});
