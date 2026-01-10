import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
}

// Simple in-memory cache to prevent hitting CoinGecko limits
// Map<string, { data: any, timestamp: number }>
const cache = new Map();
const CACHE_TTL_MS = 30 * 1000; // 30 seconds (Reduced for better accuracy)

serve(async (req) => {
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
        return new Response('ok', { headers: corsHeaders })
    }

    try {
        const { mode, tickers } = await req.json();

        if (!mode || !tickers) {
            throw new Error('Missing mode or tickers');
        }

        // Generate a cache key
        const cacheKey = `${mode}-${tickers.sort().join(',')}`;

        // Check Cache
        if (cache.has(cacheKey)) {
            const cached = cache.get(cacheKey);
            // Macro data can have longer cache (1 hour)
            const ttl = mode.includes('macro') ? 60 * 60 * 1000 : CACHE_TTL_MS;

            if (Date.now() - cached.timestamp < ttl) {
                console.log(`Serving ${cacheKey} from cache`);
                return new Response(JSON.stringify(cached.data), {
                    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
                });
            }
        }

        let resultData = null;
        console.log(`Fetching fresh data for: ${mode} - ${tickers.join(',')}`);

        if (mode === 'simple_price') {
            const ids = tickers.join(',');
            const url = `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd&include_24h_change=true`;

            const res = await fetch(url);
            if (!res.ok) throw new Error(`CoinGecko API Error: ${res.status}`);
            resultData = await res.json();
        } else if (mode === 'market_chart') {
            const id = tickers[0];
            const days = 90;
            const url = `https://api.coingecko.com/api/v3/coins/${id}/market_chart?vs_currency=usd&days=${days}`;

            const res = await fetch(url);
            if (!res.ok) throw new Error(`CoinGecko API Error: ${res.status}`);
            resultData = await res.json();
        } else if (mode === 'yahoo_finance') {
            const ticker = tickers[0];
            const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=3mo`;

            const res = await fetch(url);
            if (!res.ok) throw new Error(`Yahoo API Error: ${res.status} for ${ticker}`);
            resultData = await res.json();
        } else if (mode === 'macro_history') {
            const ticker = tickers[0];
            const isStock = tickers[1] === 'stock';

            let url;
            if (isStock) {
                url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1wk&range=10y`;
            } else {
                url = `https://api.coingecko.com/api/v3/coins/${ticker}/market_chart?vs_currency=usd&days=3650`;
            }

            const res = await fetch(url);
            if (!res.ok) throw new Error(`Macro API Error: ${res.status} for ${ticker}`);
            resultData = await res.json();
        }

        // Update Cache
        if (resultData) {
            cache.set(cacheKey, { data: resultData, timestamp: Date.now() });
        }

        return new Response(JSON.stringify(resultData), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })

    } catch (error) {
        console.error(`Market Proxy Error:`, error);
        return new Response(JSON.stringify({ error: (error as any).message }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            status: 400,
        })
    }
})
