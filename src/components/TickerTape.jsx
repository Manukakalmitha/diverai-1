import React, { useState, useEffect } from 'react';
import { fetchTickerData } from '../lib/marketData';

const TickerTape = () => {
    const [tickers, setTickers] = useState([]);
    const [loading, setLoading] = useState(true);

    // Define the tickers we want to display
    const TICKER_SYMBOLS = [
        'SPY', 'QQQ', 'BTC', 'ETH', 'NVDA', 'AAPL',
        'TSLA', 'AMD', 'AMZN', 'SOL', 'AVAX', 'MATIC'
    ];

    useEffect(() => {
        const loadTickerData = async () => {
            try {
                const data = await fetchTickerData(TICKER_SYMBOLS);

                // Format the data for display
                const formattedData = data.map(item => ({
                    symbol: item.ticker,
                    price: formatPrice(item.price),
                    change: formatChange(item.change)
                }));

                setTickers(formattedData);
                setLoading(false);
            } catch (error) {
                console.error('Ticker data fetch error:', error);
                // Fallback to empty array on error
                setTickers([]);
                setLoading(false);
            }
        };

        // Initial load
        loadTickerData();

        // Refresh every 30 seconds for near real-time updates
        const interval = setInterval(loadTickerData, 30 * 1000);

        return () => clearInterval(interval);
    }, []);

    const formatPrice = (price) => {
        if (!price) return '—';
        if (price >= 1000) {
            return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
    };

    const formatChange = (change) => {
        if (change === null || change === undefined) return '—';
        const sign = change >= 0 ? '+' : '';
        return `${sign}${change.toFixed(2)}%`;
    };

    // Show loading state or empty state
    if (loading || tickers.length === 0) {
        return (
            <div className="w-full bg-slate-950 border-b border-slate-800 h-10 flex items-center overflow-hidden whitespace-nowrap relative z-40">
                <div className="animate-marquee inline-flex items-center gap-8 px-4">
                    {TICKER_SYMBOLS.map((symbol, i) => (
                        <div key={i} className="flex items-center gap-2 text-xs font-medium">
                            <span className="text-slate-200 font-bold">{symbol}</span>
                            <span className="text-slate-400">Loading...</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="w-full bg-slate-950 border-b border-slate-800 h-10 flex items-center overflow-hidden whitespace-nowrap relative z-40">
            <div className="animate-marquee inline-flex items-center gap-8 px-4">
                {[...tickers, ...tickers, ...tickers].map((t, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-medium">
                        <span className="text-slate-200 font-bold">{t.symbol}</span>
                        <span className="text-slate-400">{t.price}</span>
                        <span className={t.change.startsWith('+') ? 'text-emerald-500' : t.change.startsWith('-') ? 'text-rose-500' : 'text-slate-500'}>
                            {t.change}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default TickerTape;
