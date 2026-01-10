import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
}

const CG_TO_BINANCE: Record<string, string> = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'solana': 'SOLUSDT',
    'ripple': 'XRPUSDT',
    'cardano': 'ADAUSDT',
    'dogecoin': 'DOGEUSDT',
    'avalanche-2': 'AVAXUSDT',
    'polkadot': 'DOTUSDT',
    'matic-network': 'MATICUSDT',
    'chainlink': 'LINKUSDT',
    'shiba-inu': 'SHIBUSDT'
};

// Simple in-memory cache
const cache = new Map();
const CACHE_TTL_MS = 30 * 1000;

serve(async (req: Request) => {
    if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });

    try {
        const { mode, tickers } = await req.json();
        if (!mode || !tickers) throw new Error('Missing mode or tickers');

        const cacheKey = `${mode}-${tickers.sort().join(',')}`;
        if (cache.has(cacheKey)) {
            const cached = cache.get(cacheKey);
            const ttl = mode.includes('macro') ? 3600000 : CACHE_TTL_MS;
            if (Date.now() - cached.timestamp < ttl) {
                return new Response(JSON.stringify(cached.data), {
                    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                });
            }
        }

        let resultData = null;

        // --- FETCH LOGIC ---
        if (mode === 'simple_price') {
            try {
                // Try CoinGecko First
                const ids = tickers.join(',');
                const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_24h_change=true`);
                if (!res.ok) throw new Error("CG Error");
                resultData = await res.json();
            } catch (e) {
                // FALLBACK TO BINANCE
                console.log("CG Simple Price Failed, using Binance Fallback");
                const binanceData: Record<string, any> = {};
                await Promise.all(tickers.map(async (id: string) => {
                    const sym = CG_TO_BINANCE[id];
                    if (sym) {
                        const bRes = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${sym}`);
                        if (bRes.ok) {
                            const b = await bRes.ok ? await bRes.json() : null;
                            if (b) {
                                binanceData[id] = { usd: parseFloat(b.lastPrice), usd_24h_change: parseFloat(b.priceChangePercent) };
                            }
                        }
                    }
                }));
                resultData = binanceData;
            }
        } else if (mode === 'market_chart') {
            const id = tickers[0];
            try {
                // Try CoinGecko
                const res = await fetch(`https://api.coingecko.com/api/v3/coins/${id}/market_chart?vs_currency=usd&days=90`);
                if (!res.ok) throw new Error("CG Error");
                resultData = await res.json();
            } catch (e) {
                // FALLBACK TO BINANCE KLINES
                const sym = CG_TO_BINANCE[id];
                if (sym) {
                    console.log(`CG Market Chart Failed for ${id}, using Binance Klines for ${sym}`);
                    const bRes = await fetch(`https://api.binance.com/api/v3/klines?symbol=${sym}&interval=1d&limit=90`);
                    if (bRes.ok) {
                        const klines = await bRes.json();
                        resultData = {
                            prices: klines.map((k: any) => [k[0], parseFloat(k[4])]),
                            market_caps: [],
                            total_volumes: []
                        };
                    }
                }
            }
        } else if (mode === 'yahoo_finance') {
            const ticker = tickers[0];
            const res = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=3mo`);
            if (res.ok) resultData = await res.json();
        } else if (mode === 'macro_history') {
            const ticker = tickers[0];
            const isStock = tickers[1] === 'stock';
            const url = isStock
                ? `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1wk&range=10y`
                : `https://api.coingecko.com/api/v3/coins/${ticker}/market_chart?vs_currency=usd&days=3650`;
            const res = await fetch(url);
            if (res.ok) resultData = await res.json();
        }

        if (resultData) cache.set(cacheKey, { data: resultData, timestamp: Date.now() });

        return new Response(JSON.stringify(resultData), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });

    } catch (error) {
        console.error(`Market Proxy Error:`, error);
        return new Response(JSON.stringify({ error: (error as any).message }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            status: 400,
        });
    }
});
