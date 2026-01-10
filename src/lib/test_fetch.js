
const fetchYahooData = async (ticker) => {
    const yTicker = ticker.replace('.', '-');
    try {
        const url = `https://corsproxy.io/?` + encodeURIComponent(`https://query1.finance.yahoo.com/v8/finance/chart/${yTicker}?interval=1d&range=3mo`);
        console.log(`Fetching: ${url}`);
        const response = await fetch(url);
        if (!response.ok) {
            console.error(`HTTP Error: ${response.status}`);
            return null;
        }
        const result = await response.json();
        if (result.chart && result.chart.result && result.chart.result[0]) {
            console.log(`Success for ${ticker}: Price ${result.chart.result[0].meta.regularMarketPrice}`);
            return true;
        } else {
            console.warn(`No result for ${ticker}:`, JSON.stringify(result).slice(0, 100));
        }
    } catch (error) {
        console.error(`Fetch error for ${ticker}:`, error.message);
    }
    return null;
};

const test = async () => {
    console.log("Testing Crypto (CoinGecko)...");
    try {
        const res = await fetch(`https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24h_change=true`);
        const data = await res.json();
        console.log("Crypto Success:", data);
    } catch (e) {
        console.error("Crypto Fail:", e.message);
    }

    console.log("\nTesting Stocks (Yahoo via Proxy)...");
    await fetchYahooData('AAPL');
    await fetchYahooData('TSLA');
};

test();
