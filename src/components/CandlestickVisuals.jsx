import React from 'react';

const Candle = ({ open, close, high, low, color, x = 12, width = 6 }) => {
    // Coordinate system: 0 (top) to 24 (bottom)
    const bodyTop = Math.min(open, close);
    const bodyHeight = Math.abs(open - close) || 0.5; // Ensure at least faint line for doji

    // Wick
    const wickX = x + width / 2;

    return (
        <g>
            <line x1={wickX} y1={high} x2={wickX} y2={low} stroke={color} strokeWidth="1" />
            <rect x={x} y={bodyTop} width={width} height={bodyHeight} fill={color} />
        </g>
    );
};

const CandlestickVisual = ({ pattern, sentiment }) => {
    const isBull = sentiment === 'Bullish';
    const bullColor = '#10b981'; // emerald-500
    const bearColor = '#f43f5e'; // rose-500
    const neutralColor = '#94a3b8'; // slate-400

    const color = isBull ? bullColor : (sentiment === 'Bearish' ? bearColor : neutralColor);

    let content = null;

    // Pattern Logic (Schematic Representations)
    switch (true) {
        // --- Single Candle Patterns ---

        case pattern.includes('Doji'):
            // Cross shape
            content = (
                <>
                    {/* Previous Context candle (small) */}
                    <Candle open={10} close={14} high={8} low={16} color={neutralColor} x={4} width={4} />
                    {/* The Doji */}
                    <Candle open={12} close={12.2} high={6} low={18} color={color} x={14} width={6} />
                </>
            );
            break;

        case pattern.includes('Hammer'):
            // Small body at top, long lower wick
            content = (
                <>
                    <Candle open={10} close={14} high={8} low={16} color={neutralColor} x={4} width={4} />
                    <Candle open={6} close={9} high={6} low={20} color={color} x={14} width={6} />
                </>
            );
            break;

        case pattern.includes('Inverted Hammer') || pattern.includes('Shooting Star'):
            // Small body at bottom, long upper wick
            content = (
                <>
                    <Candle open={14} close={18} high={12} low={20} color={neutralColor} x={4} width={4} />
                    <Candle open={16} close={19} high={4} low={19} color={color} x={14} width={6} />
                </>
            );
            break;

        case pattern.includes('Marubozu'):
            // Full body, no wicks
            content = (
                <Candle open={4} close={20} high={4} low={20} color={color} x={10} width={8} />
            );
            break;

        // --- Two Candle Patterns ---

        case pattern.includes('Engulfing'):
            // 1. Small opposite, 2. Large matching
            const prevColor = isBull ? bearColor : bullColor;
            content = (
                <>
                    <Candle open={12} close={16} high={11} low={17} color={prevColor} x={6} width={4} />
                    <Candle open={18} close={6} high={19} low={5} color={color} x={16} width={8} />
                </>
            );
            break;

        case pattern.includes('Harami') || pattern.includes('Inside Bar'):
            // 1. Large Mother, 2. Small Baby
            const motherColor = isBull ? bearColor : bullColor; // Paradox: Bullish Harami usually follows red candle
            content = (
                <>
                    <Candle open={6} close={18} high={5} low={19} color={motherColor} x={4} width={6} />
                    <Candle open={14} close={10} high={9} low={15} color={color} x={16} width={4} />
                </>
            );
            break;

        // --- Generic Trend Patterns ---

        case pattern.includes('Continuation'):
            // Three steps
            content = (
                <>
                    <Candle open={16} close={12} high={17} low={11} color={color} x={2} width={4} />
                    <Candle open={13} close={10} high={14} low={9} color={color} x={10} width={4} />
                    <Candle open={11} close={5} high={12} low={4} color={color} x={18} width={4} />
                </>
            );
            break;

        case pattern.includes('Squeeze'):
            // Tightening range
            content = (
                <>
                    <Candle open={8} close={16} high={6} low={18} color={neutralColor} x={2} width={4} />
                    <Candle open={10} close={14} high={9} low={15} color={neutralColor} x={10} width={4} />
                    <Candle open={11} close={13} high={11} low={13} color={color} x={18} width={4} />
                </>
            );
            break;

        default:
            // Generic single candle
            content = <Candle open={16} close={8} high={18} low={6} color={color} x={10} width={6} />;
    }

    return (
        <svg viewBox="0 0 28 24" className="w-8 h-8 drop-shadow-lg">
            {/* Subtle Grid Background */}
            <rect x="0" y="0" width="28" height="24" fill="none" />
            {content}
        </svg>
    );
};

export default CandlestickVisual;
