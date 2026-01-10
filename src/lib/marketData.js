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
    const decimalMatches = text.match(/\d{1,3}(?:,\d{3})*\.\d{1,8}/g);

    // 2. Integer Match: Potential large-cap price or indices (e.g. BTC at 102434)
    const intMatches = text.match(/\b\d{3,7}\b/g);

    const allMatches = [...(decimalMatches || []), ...(intMatches || [])];

    const candidates = allMatches
        .map(m => parseFloat(m.replace(/,/g, '')))
        .filter(n => {
            // Filter out obviously wrong numbers
            if (n >= 2020 && n <= 2030) return false; // Likely a year
            if (n === 24 || n === 1 || n === 7) return false; // Timeframes
            return n > 0.0001 && n < 20000000;
        });

    if (candidates.length === 0) return null;

    // Heuristic: Prefer decimals, then larger numbers
    const sorted = candidates.sort((a, b) => {
        const aHasDecimal = a % 1 !== 0;
        const bHasDecimal = b % 1 !== 0;
        if (aHasDecimal && !bHasDecimal) return -1;
        if (!aHasDecimal && bHasDecimal) return 1;
        return b - a;
    });

    return sorted[0];
};

export const detectTicker = (text) => {
    if (!text) return null;

    const upper = text.toUpperCase();

    // 0. Blacklist / Filter common noise
    const blacklist = ['USD', 'USDT', 'VOL', '24H', 'HIGH', 'LOW', 'PRICE', 'INDEX', 'STOCK', 'CRYPTO', 'CHART', 'MARKET', 'TRADE', 'BUY', 'SELL'];

    // Normalize: Remove noise
    const cleanText = upper.replace(/[^A-Z0-9]/g, ' ');
    const words = cleanText.split(/\s+/).filter(w => w.length >= 2 && !blacklist.includes(w));

    // 1. Exact Match Strategy (Priority)
    for (const word of words) {
        if (COIN_MAP[word] || STOCK_MAP[word]) return word;
    }

    // 2. Pair Strategy (e.g., BTCUSDT, BTC/USD, ETH-PERP)
    const pairMatch = upper.match(/([A-Z]{2,10})[\/\-\\]?(?:USDT|USD|BUSD|USDC|PERP|FRAX|DAI)/);
    if (pairMatch) {
        const t = pairMatch[1];
        if (COIN_MAP[t] || STOCK_MAP[t]) return t;
        if (t.length >= 2 && !blacklist.includes(t)) return t;
    }

    // 3. Page Title Context Strategy (Yahoo, TradingView, etc.)
    const titleMatch = upper.match(/\(([A-Z]{2,6})\)[ -]|^([A-Z]{2,6})\s+\d/);
    if (titleMatch) {
        const t = titleMatch[1] || titleMatch[2];
        if (COIN_MAP[t] || STOCK_MAP[t]) return t;
        if (t.length >= 2 && !blacklist.includes(t)) return t;
    }

    // 4. Heuristic Search
    const sortedCoins = Object.keys(COIN_MAP).sort((a, b) => b.length - a.length);
    for (const ticker of sortedCoins) {
        if (new RegExp(`\\b${ticker}\\b`).test(upper)) return ticker;
    }

    const sortedStocks = Object.keys(STOCK_MAP).sort((a, b) => b.length - a.length);
    for (const ticker of sortedStocks) {
        if (new RegExp(`\\b${ticker}\\b|\\$${ticker}`).test(upper)) return ticker;
    }

    return null;
};

// --- CRYPTO API (CoinGecko) ---

const PROXIES = []; // Proxies no longer needed with Edge Function
import { supabase } from './supabase'; // Correct relative path within src/lib

const fetchProxy = async (mode, tickers) => {
    try {
        const url = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/market-proxy`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': import.meta.env.VITE_SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`
            },
            body: JSON.stringify({ mode, tickers })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || `Proxy Error (${response.status})`);
        }

        return await response.json();
    } catch (err) {
        console.warn(`Market Proxy Error (${mode}):`, err);
        return null;
    }
};

const fetchSafe = async (targetUrl) => {
    // Deprecated: client-side fetch. Keeping stub to avoid breaking references if any remain.
    console.warn("Legacy fetchSafe called. Should use fetchProxy.", targetUrl);
    return null;
};

export const fetchMarketData = async (ticker) => {
    const coinId = COIN_MAP[ticker];
    if (!coinId) return null;

    // Single ticker fetch via proxy
    const data = await fetchProxy('simple_price', [coinId]);

    if (data && data[coinId]) {
        return {
            price: Number(data[coinId].usd) || 0,
            change24h: Number(data[coinId].usd_24h_change) || 0,
            volume: Number(data[coinId].usd_24h_vol) || 0,
            source: 'CoinGecko (Edge)'
        };
    }
    return null;
};

export const fetchHistoricalData = async (ticker, days = 90) => {
    const coinId = COIN_MAP[ticker];
    if (!coinId) return null;

    // Chart fetch via proxy
    const data = await fetchProxy('market_chart', [coinId]);

    if (data && data.prices) {
        return data.prices.map(p => p[1]);
    }
    return null;
};

// --- STOCK API (Finnhub) ---

export const fetchStockData = async (ticker, apiKey) => {
    if (!STOCK_MAP[ticker] || !apiKey) return null;

    try {
        // Finnhub Quote Endpoint
        const response = await fetch(`https://finnhub.io/api/v1/quote?symbol=${ticker}&token=${apiKey}`);
        const data = await response.json();

        // data = { c: Current price, d: Change, dp: Percent change, ... }
        if (data.c) {
            return {
                price: Number(data.c),
                change24h: Number(data.dp), // Finnhub returns percentage change in 'dp'
                volume: 0, // Quote endpoint doesn't always have vol, could use 'v' if available but often 0 for delayed
                source: 'Finnhub API'
            };
        }
    } catch (error) {
        console.warn("Stock Data API Error:", error);
    }
    return null;
};

export const fetchStockHistory = async (ticker, apiKey) => {
    if (!STOCK_MAP[ticker] || !apiKey) return null;

    try {
        // Finnhub Stock Candles Endpoint
        // resolution: 'D' (Day)
        // from/to: UNIX timestamps
        const to = Math.floor(Date.now() / 1000);
        const from = to - (90 * 24 * 60 * 60); // 90 days ago

        const response = await fetch(`https://finnhub.io/api/v1/stock/candle?symbol=${ticker}&resolution=D&from=${from}&to=${to}&token=${apiKey}`);
        const data = await response.json();

        if (data.s === 'ok' && data.c) {
            return data.c; // 'c' is the array of close prices
        }
    } catch (error) {
        console.warn("Stock History API Error:", error);
    }
    return null;
};
// --- YAHOO FINANCE API (Key-Free Fallback) ---

export const fetchYahooData = async (ticker) => {
    // Yahoo tickers for indices might need care (e.g. ^GSPC for SPY sometimes preferred, but SPY works)
    // Yahoo uses dot for classes (BRK-B instead of BRK.B)
    const yTicker = ticker.replace('.', '-');

    const tryFetch = async (sym) => {
        try {
            const data = await fetchProxy('yahoo_finance', [sym]);

            if (data && data.chart && data.chart.result && data.chart.result[0]) {
                const quote = data.chart.result[0];
                const meta = quote.meta;
                const indicators = quote.indicators.quote[0];
                const prices = indicators.close || [];

                // Filter out nulls from prices
                const cleanPrices = prices.filter(p => p !== null && p !== undefined);

                if (cleanPrices.length === 0) return null;

                // Last Price
                const currentPrice = meta.regularMarketPrice || cleanPrices[cleanPrices.length - 1];
                const prevClose = meta.chartPreviousClose || cleanPrices[cleanPrices.length - 2] || currentPrice;
                const changePercent = ((currentPrice - prevClose) / prevClose) * 100;

                return {
                    marketStats: {
                        price: currentPrice,
                        change24h: changePercent,
                        volume: meta.regularMarketVolume || 0,
                        source: 'Yahoo Finance (Edge)'
                    },
                    historicalPrices: cleanPrices
                };
            }
        } catch (e) { console.warn(`Yahoo fetch failed for ${sym}:`, e); }
        return null;
    };

    // Attempt 1: Standard
    let result = await tryFetch(yTicker);

    // Attempt 2: Crypto Fallback (Append -USD if not already there and not a stock)
    if (!result && !STOCK_MAP[ticker] && !yTicker.includes('-')) {
        console.log(`Retrying ${yTicker} as crypto-pair (${yTicker}-USD)...`);
        result = await tryFetch(`${yTicker}-USD`);
    }

    return result;
};

// --- TICKER TAPE API ---
export const fetchTickerData = async (tickers) => {
    const cryptoTickers = tickers.filter(t => COIN_MAP[t]);
    const stockTickers = tickers.filter(t => STOCK_MAP[t]);

    // Parallel execution for maximum performance
    const [cryptoData, stockData] = await Promise.all([
        fetchBulkCrypto(cryptoTickers),
        fetchBulkStocks(stockTickers)
    ]);

    return [...cryptoData, ...stockData].filter(Boolean);
};

const fetchBulkCrypto = async (tickers) => {
    if (tickers.length === 0) return [];
    try {
        const ids = tickers.map(t => COIN_MAP[t]).filter(Boolean);
        if (ids.length === 0) return [];

        // Single Bulk Call to Edge Function
        const data = await fetchProxy('simple_price', ids);

        if (!data) return [];

        // Fetch sparklines (historical data) for top tickers
        // We can optimize this later to be a single 'bulk_history' call if needed, 
        // but for now parallel individual calls to Edge Function (which handles caching) is fine.
        const topTickers = tickers.slice(0, 6);
        const sparklineMap = {};

        await Promise.all(topTickers.map(async (t) => {
            const h = await fetchHistoricalData(t, 1);
            if (h) sparklineMap[t] = h;
        }));

        return tickers.map(t => {
            const id = COIN_MAP[t];
            const coinData = data[id];
            if (!coinData) return null;

            return {
                ticker: t,
                price: coinData.usd,
                change: coinData.usd_24h_change,
                sparkline: sparklineMap[t] || [],
                isCrypto: true
            };
        }).filter(Boolean);

    } catch (e) {
        console.warn("Bulk Crypto Error:", e);
        return [];
    }
};

const fetchBulkStocks = async (tickers) => {
    // Yahoo public API isn't bulk-friendly, but we can parallelize individual fetches
    const promises = tickers.map(async (t) => {
        try {
            const data = await fetchYahooData(t);
            if (data?.marketStats) {
                return {
                    ticker: t,
                    price: data.marketStats.price,
                    change: data.marketStats.change24h,
                    sparkline: data.historicalPrices || [],
                    isCrypto: false
                };
            }
        } catch (e) { /* ignore individual failures */ }
        return null;
    });
    return (await Promise.all(promises)).filter(Boolean);
};
// --- MACO INTELLIGENCE LAYER (10-Year Deep Scan) ---

export const fetchMacroHistory = async (ticker) => {
    const isStock = STOCK_MAP[ticker];
    const coinId = COIN_MAP[ticker];

    try {
        let data = null;

        if (isStock) {
            const yTicker = ticker.replace('.', '-');
            data = await fetchProxy('macro_history', [yTicker, 'stock']);
        } else if (coinId) {
            // Use Yahoo Finance for Crypto 10Y (CoinGecko requires paid API for >365d)
            const yTicker = `${ticker.toUpperCase()}-USD`;
            data = await fetchProxy('macro_history', [yTicker, 'stock']);
        }

        // Both now return Yahoo format
        if (data?.chart?.result?.[0]) {
            const quote = data.chart.result[0];
            const timestamps = quote.timestamp || [];
            const closes = quote.indicators.quote[0].close || [];

            // Filter nulls and sync arrays
            const validData = timestamps.map((t, i) => ({ date: t, price: closes[i] })).filter(d => d.price != null);
            return {
                prices: validData.map(d => d.price),
                dates: validData.map(d => d.date),
                source: 'Yahoo Finance (Institutional)'
            };
        }

    } catch (err) {
        console.warn("Macro Data Fetch Error:", err);
    }
    return null;
};

export const calculateMacroSentiment = (historicalPrices) => {
    if (!historicalPrices || historicalPrices.length < 52) return 0.5; // Need at least 1 year of weekly data

    const current = historicalPrices[historicalPrices.length - 1];
    const ath = Math.max(...historicalPrices);
    const atl = Math.min(...historicalPrices);

    // Proximity to ATH/ATL
    const range = ath - atl || 1;
    const percentile = (current - atl) / range; // 0 to 1

    // 200-Period Moving Average (Macro Trend)
    const window = 200;
    const slice = historicalPrices.slice(-window);
    const ma200 = slice.reduce((a, b) => a + b, 0) / slice.length;

    const isAboveMA = current > ma200 ? 1 : 0;
    const maSlope = current / ma200; // > 1 is bullish

    // Formula: 50% percentile position + 50% MA alignment
    let sentiment = (percentile * 0.4) + (isAboveMA * 0.4) + (Math.min(1.5, maSlope) / 1.5 * 0.2);

    return Math.min(0.95, Math.max(0.05, sentiment));
};
