// Map common tickers to CoinGecko IDs
export const COIN_MAP = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'XRP': 'ripple',
    'ADA': 'cardano',
    'DOGE': 'dogecoin',
    'AVAX': 'avalanche-2',
    'DOT': 'polkadot',
    'MATIC': 'matic-network',
    'LINK': 'chainlink',
    'LTC': 'litecoin',
    'SHIB': 'shiba-inu',
    'TRX': 'tron',
    'UNI': 'uniswap',
    'ATOM': 'cosmos',
    'XMR': 'monero',
    'ETC': 'ethereum-classic',
    'XLM': 'stellar',
    'BCH': 'bitcoin-cash',
    'FIL': 'filecoin',
    'APT': 'aptos',
    'QNT': 'quant-network',
    'NEAR': 'near',
    'ARB': 'arbitrum',
    'VET': 'vechain',
    'MKR': 'maker',
    'AAVE': 'aave',
    'GRT': 'the-graph',
    'ALGO': 'algorand',
    'AXS': 'axie-infinity',
    'SAND': 'the-sandbox',
    'EOS': 'eos',
    'MANA': 'decentraland',
    'THETA': 'theta-token',
    'EGLD': 'elrond-erd-2',
    'FTM': 'fantom',
    'XTZ': 'tezos',
    'FLOW': 'flow',
    'IMX': 'immutable-x',
    'SNX': 'havven',
    'NEO': 'neo',
    'CVX': 'convex-finance',
    'CRV': 'curve-dao-token',
    'BAT': 'basic-attention-token',
    'CHZ': 'chiliz',
    'ENJ': 'enjincoin',
    'DASH': 'dash',
    'COMP': 'compound-governance-token',
    'ZEC': 'zcash',
    'XEM': 'nem',
    'HOT': 'holo',
    'IOTX': 'iotex',
    'RUNE': 'thorchain',
    'KSM': 'kusama',
    'ZIL': 'zilliqa',
    'RVN': 'ravencoin',
    'CELO': 'celo',
    'ONE': 'harmony',
    'QTUM': 'qtum',
    'BNB': 'binancecoin'
};

// Simple Persistent Cache to prevent 429s
const CACHE_DURATION = 2 * 60 * 1000; // 2 minutes (Reduced from 15 to improve accuracy)
const getCachedData = (key) => {
    try {
        const item = localStorage.getItem(`cache_${key}`);
        if (item) {
            const { data, timestamp } = JSON.parse(item);
            if (Date.now() - timestamp < CACHE_DURATION) return data;
        }
    } catch (e) { console.warn("Cache read compile error", e); }
    return null;
};
const setCachedData = (key, data) => {
    try {
        localStorage.setItem(`cache_${key}`, JSON.stringify({ data, timestamp: Date.now() }));
    } catch (e) {
        // Handle quota exceeded
        try { localStorage.clear(); } catch (err) { }
    }
};

// Common S&P 500 and Tech Stocks
export const STOCK_MAP = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms Inc.',
    'BRK.B': 'Berkshire Hathaway',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'PG': 'Procter & Gamble Co.',
    'MA': 'Mastercard Inc.',
    'LLY': 'Eli Lilly and Co.',
    'HD': 'Home Depot Inc.',
    'CVX': 'Chevron Corp.',
    'MRK': 'Merck & Co.',
    'KO': 'Coca-Cola Co.',
    'PEP': 'PepsiCo Inc.',
    'AVGO': 'Broadcom Inc.',
    'COST': 'Costco Wholesale Corp.',
    'ORCL': 'Oracle Corp.',
    'AMD': 'Advanced Micro Devices',
    'NFLX': 'Netflix Inc.',
    'INTC': 'Intel Corp.',
    'IBM': 'IBM Corp.',
    'QCOM': 'Qualcomm Inc.',
    'TXN': 'Texas Instruments',
    'HON': 'Honeywell',
    'UNH': 'UnitedHealth Group',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust',
    'IWM': 'iShares Russell 2000',
    'DIA': 'SPDR Dow Jones'
};

export const detectPrice = (text) => {
    if (!text) return null;

    // 1. Precise Match: Search for numbers with decimal places (common in trading)
    // Supports US Format (92,014.26) and handles messy OCR (92.014.26)
    // V5.2: Also detect thousand separates WITHOUT decimals (92,520)
    const decimalMatches = text.match(/\b\d{1,3}(?:[.,]\d{3})*[.,]\d{1,8}\b/g);
    const complexIntMatches = text.match(/\b\d{1,3}(?:,\d{3})+\b/g);
    const simpleIntMatches = text.match(/\b\d{4,7}\b/g);

    const allMatches = [
        ...(decimalMatches || []),
        ...(complexIntMatches || []),
        ...(simpleIntMatches || [])
    ];

    const candidates = allMatches
        .map(m => {
            // Standardize: Remove commas if they are thousand decorators
            const cleaned = m.replace(/,/g, '');
            // If we have multiple dots (messy OCR), take the last one as decimal
            const parts = cleaned.split('.');
            if (parts.length > 2) {
                const decimal = parts.pop();
                return parseFloat(parts.join('') + '.' + decimal);
            }
            return parseFloat(cleaned);
        })
        .filter(n => {
            if (n >= 2000 && n <= 3000) return false; // Filter out years
            if (n === 24 || n === 1 || n === 7 || n === 30 || n === 15 || n === 60) return false;
            return n > 0.0001 && n < 20000000;
        });

    if (candidates.length === 0) return null;

    // Heuristics (V5.2):
    // Prefer numbers that match a "typical" price scale (0.1 to 150,000)
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

    return sorted[0];
};

export const detectTicker = (text) => {
    if (!text) return null;
    const upper = text.toUpperCase();
    const blacklist = ['VOL', 'USD', 'USDT', 'UTC', 'CRYPTO', 'CRYPTOCURRENCY', 'PRICE', 'MARKET', 'CHANGE', 'TIME', 'TOTAL', 'LOW', 'HIGH', 'OPEN', 'CLOSE', 'DAILY', 'WEEKLY'];

    // 1. Explicit Ticker Dictionary Strategy
    const words = upper.split(/[^A-Z0-9]/).filter(w => w.length >= 2);
    for (const ticker of Object.keys(COIN_MAP)) {
        if (words.includes(ticker)) return ticker;
    }

    // 2. Pair Strategy (e.g., BTCUSDT, BTC/USD, ETH-PERP, 1000SHIBUSDT)
    const pairMatch = upper.match(/\b([A-Z0-9]{2,10})[\/\-\\]?(?:USDT|USD|BUSD|USDC|PERP|FRAX|DAI)\b/);
    if (pairMatch) {
        const t = pairMatch[1];
        if (COIN_MAP[t] || STOCK_MAP[t]) return t;
        if (t.length >= 2 && !blacklist.includes(t)) return t;
    }

    // 3. Contextual Pattern: CRYPTO O H L (Common TV/IBKR header) followed by Price
    if ((upper.includes('CRYPTO') || upper.includes('O H L')) && !upper.includes('BITCOIN')) {
        const price = detectPrice(text);
        if (price > 40000 && price < 150000) return 'BTC';
        if (price > 1500 && price < 10000) return 'ETH';
    }

    // 4. Full Name Variant Strategy (V5.1: Bitcoin, Ethereum, etc.)
    const nameMap = {
        'BITCOIN': 'BTC',
        'ETHEREUM': 'ETH',
        'SOLANA': 'SOL',
        'RIPPLE': 'XRP',
        'CARDANO': 'ADA',
        'DOGECOIN': 'DOGE',
        'AVALANCHE': 'AVAX'
    };
    for (const [name, ticker] of Object.entries(nameMap)) {
        if (upper.includes(name)) return ticker;
    }

    // 5. Page Title Strategy
    const titleMatch = upper.match(/\(([A-Z0-9]{2,6})\)[ -]|^([A-Z0-9]{2,6})\s+\d/);
    if (titleMatch) {
        const t = titleMatch[1] || titleMatch[2];
        if (COIN_MAP[t] || STOCK_MAP[t]) return t;
        if (t.length >= 2 && !blacklist.includes(t)) return t;
    }

    return null;
};

// ... REST OF FILE (FETCHERS) ...
import { supabase } from './supabase.js';

export const fetchMarketData = async (ticker) => {
    const coinId = COIN_MAP[ticker];
    if (!coinId) return null;

    try {
        const response = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true`);
        const data = await response.json();
        const coinData = data[coinId];
        return {
            price: coinData.usd,
            change24h: coinData.usd_24h_change,
            volume: coinData.usd_24h_vol,
            source: 'CoinGecko'
        };
    } catch (err) {
        console.error("CoinGecko Fetch Error:", err);
        return null;
    }
};

export const fetchHistoricalData = async (ticker, days = 90) => {
    const coinId = COIN_MAP[ticker];
    if (!coinId) return null;

    const cached = getCachedData(`${coinId}_${days}`);
    if (cached) return cached;

    try {
        const response = await fetch(`https://api.coingecko.com/api/v3/coins/${coinId}/market_chart?vs_currency=usd&days=${days}`);
        const data = await response.json();

        // CoinGecko only provides [timestamp, price] pairs. 
        // We synthesize OHLC from this for V4 compatibility if needed, 
        // but for pure prices we extract the second element.
        const prices = data.prices.map(p => p[1]);
        const volumes = data.total_volumes.map(v => v[1]);

        // Synthesize H/L/C objects
        const historical = {
            closes: prices,
            highs: prices.map(p => p * 1.002), // Approximation
            lows: prices.map(p => p * 0.998),
            volumes: volumes
        };

        setCachedData(`${coinId}_${days}`, historical);
        return historical;
    } catch (err) {
        console.error("CoinGecko History Error:", err);
        return null;
    }
};

export const fetchStockData = async (ticker, apiKey) => {
    try {
        const response = await fetch(`https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${apiKey}`);
        const data = await response.json();
        if (!data.c) return null;
        return {
            price: data.c,
            change24h: data.dp,
            volume: 0,
            source: 'Finnhub'
        };
    } catch (err) {
        return null;
    }
};

export const fetchStockHistory = async (ticker, apiKey) => {
    const end = Math.floor(Date.now() / 1000);
    const start = end - (90 * 24 * 60 * 60);
    try {
        const response = await fetch(`https://finnhub.io/api/v1/stock/candle?symbol=${ticker}&resolution=D&from=${start}&to=${end}&token=${apiKey}`);
        const data = await response.json();
        if (data.s !== 'ok') return null;
        return {
            closes: data.c,
            highs: data.h,
            lows: data.l,
            volumes: data.v
        };
    } catch (err) {
        return null;
    }
};

// Yahoo Finance Proxy (No Auth Required)
export const fetchYahooData = async (ticker) => {
    // Standardize ticker for Yahoo
    const yTicker = ticker.includes('-') ? ticker.replace('-', '') : ticker;

    try {
        // V5.2 CORS Hotfix: Prefer corsproxy.io as it is more stable than allorigins
        const baseUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${yTicker}?interval=1d&range=3mo`;
        const proxies = [
            `https://corsproxy.io/?${encodeURIComponent(baseUrl)}`,
            `https://api.allorigins.win/raw?url=${encodeURIComponent(baseUrl)}`
        ];

        let data = null;
        for (const proxy of proxies) {
            try {
                const response = await fetch(proxy);
                if (!response.ok) continue;
                data = await response.json();
                if (data?.chart?.result) break;
            } catch (pErr) { continue; }
        }

        if (!data) throw new Error("All proxies failed");
        const result = data.chart.result[0];

        const closes = result.indicators.quote[0].close.filter(c => c !== null);
        const highs = result.indicators.quote[0].high.filter(h => h !== null);
        const lows = result.indicators.quote[0].low.filter(l => l !== null);
        const volumes = result.indicators.quote[0].volume.filter(v => v !== null);

        const price = result.meta.regularMarketPrice;
        const prevClose = result.meta.previousClose;
        const change = ((price - prevClose) / prevClose) * 100;

        return {
            marketStats: {
                price,
                change24h: change,
                volume: volumes[volumes.length - 1],
                source: 'Yahoo Finance'
            },
            historicalData: {
                closes,
                highs,
                lows,
                volumes
            }
        };
    } catch (err) {
        console.warn("Yahoo data fetch failed for", ticker, err);
        return null;
    }
};

// V5 Multi-Timeframe Fetcher
export const fetchMacroHistory = async (ticker) => {
    if (!ticker || ticker === "VISUAL-SCAN") return null;
    const yTicker = ticker.includes('-') ? ticker.replace('-', '') : ticker;
    try {
        // V5.2 CORS Hotfix: Multi-proxy fallback
        const baseUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${yTicker}?interval=1d&range=1y`;
        const proxies = [
            `https://corsproxy.io/?${encodeURIComponent(baseUrl)}`,
            `https://api.allorigins.win/raw?url=${encodeURIComponent(baseUrl)}`
        ];

        let data = null;
        for (const proxy of proxies) {
            try {
                const response = await fetch(proxy);
                if (!response.ok) continue;
                data = await response.json();
                if (data?.chart?.result) break;
            } catch (pErr) { continue; }
        }

        const result = data?.chart?.result?.[0];
        return result?.indicators?.quote?.[0]?.close?.filter(c => c !== null) || null;
    } catch (err) {
        return null;
    }
};

/**
 * Macro Sentiment Core (V5.1 Restore)
 * Calculates a long-term bias based on 10-year / annual price context
 */
export const calculateMacroSentiment = (prices) => {
    if (!prices || prices.length < 10) return 0.5;
    const current = prices[prices.length - 1];

    // Multi-period context (Approximate SMA alignment)
    const sma50 = prices.slice(-50).reduce((a, b) => a + b, 0) / Math.min(prices.length, 50);
    const sma200 = prices.slice(-200).reduce((a, b) => a + b, 0) / Math.min(prices.length, 200);

    let score = 0.5;
    if (current > sma50) score += 0.1;
    if (current > sma200) score += 0.15;
    if (sma50 > sma200) score += 0.1;
    if (current > prices[0]) score += 0.1; // Baseline annual growth

    // Scale momentum for subtle bias adjustments
    const mom = (current - prices[prices.length - 20]) / prices[prices.length - 20];
    score += (mom * 0.5);

    return Math.max(0.1, Math.min(0.9, score));
};

// Bulk Ticker Fetcher for TickerTape components (V5.1 Restore)
export const fetchTickerData = async (symbols) => {
    return Promise.all(symbols.map(async (ticker) => {
        try {
            // Try Yahoo first for bulk (supports Stocks AND Crypto)
            const standardized = ticker.includes('-') ? ticker.replace('-', '') : ticker;
            const yahooTicker = COIN_MAP[ticker] ? `${ticker}-USD` : standardized;

            // V5.2 CORS Hotfix: Multi-proxy fallback
            const baseUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${yahooTicker}?interval=1h&range=7d`;
            const proxies = [
                `https://corsproxy.io/?${encodeURIComponent(baseUrl)}`,
                `https://api.allorigins.win/raw?url=${encodeURIComponent(baseUrl)}`
            ];

            let data = null;
            for (const proxy of proxies) {
                try {
                    const response = await fetch(proxy);
                    if (!response.ok) continue;
                    data = await response.json();
                    if (data?.chart?.result) break;
                } catch (pErr) { continue; }
            }

            const result = data?.chart?.result?.[0];

            if (result) {
                const prices = result.indicators.quote[0].close.filter(p => p !== null);
                const price = result.meta.regularMarketPrice;
                const prevClose = result.meta.previousClose;
                const change = ((price - prevClose) / prevClose) * 100;

                return {
                    ticker,
                    price,
                    change,
                    sparkline: prices.slice(-24) // Last 24 hourly points for sparkline
                };
            }

            // Fallback to individual fetchers if Yahoo fails
            const crypto = COIN_MAP[ticker] ? await fetchMarketData(ticker) : await fetchStockData(ticker, localStorage.getItem('finnhub_key'));
            return {
                ticker,
                price: crypto?.price || 0,
                change: crypto?.change24h || 0,
                sparkline: [crypto?.price || 0, crypto?.price || 0]
            };
        } catch (e) {
            console.warn(`Fetch failed for ${ticker}:`, e);
            return { ticker, price: 0, change: 0, sparkline: [0, 0] };
        }
    }));
};
