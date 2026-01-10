import React from 'react';

const TickerTape = () => {
    const tickers = [
        { symbol: 'SPY', price: '478.20', change: '+0.15%' },
        { symbol: 'QQQ', price: '409.30', change: '+0.42%' },
        { symbol: 'BTC', price: '46,230.50', change: '-1.20%' },
        { symbol: 'ETH', price: '3,105.10', change: '+0.85%' },
        { symbol: 'NVDA', price: '540.10', change: '+2.10%' },
        { symbol: 'AAPL', price: '185.40', change: '-0.30%' },
        { symbol: 'TSLA', price: '240.50', change: '+1.15%' },
        { symbol: 'AMD', price: '145.20', change: '+3.40%' },
        { symbol: 'AMZN', price: '155.60', change: '+0.50%' },
        { symbol: 'EUR/USD', price: '1.0950', change: '-0.05%' },
        { symbol: 'USD/JPY', price: '145.20', change: '+0.10%' },
        { symbol: 'GLD', price: '190.50', change: '+0.25%' },
    ];

    return (
        <div className="w-full bg-slate-950 border-b border-slate-800 h-10 flex items-center overflow-hidden whitespace-nowrap relative z-40">
            <div className="animate-marquee inline-flex items-center gap-8 px-4">
                {[...tickers, ...tickers, ...tickers].map((t, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-medium">
                        <span className="text-slate-200 font-bold">{t.symbol}</span>
                        <span className="text-slate-400">{t.price}</span>
                        <span className={t.change.startsWith('+') ? 'text-emerald-500' : 'text-rose-500'}>
                            {t.change}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default TickerTape;
