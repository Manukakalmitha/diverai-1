/**
 * Tesseract.js Configuration Presets
 * Different OCR configurations optimized for various scenarios
 */

/**
 * Page Segmentation Modes (PSM)
 * Different modes for different text layouts
 */
export const PSM_MODES = {
    // PSM 6: Assume a single uniform block of text (best for clean screenshots)
    UNIFORM_BLOCK: {
        psm: 6,
        name: 'Uniform Block',
        description: 'Best for clean chart screenshots with consistent text',
        tessedit_pageseg_mode: '6'
    },

    // PSM 11: Sparse text - find as much text as possible in no particular order
    SPARSE_TEXT: {
        psm: 11,
        name: 'Sparse Text',
        description: 'Good for scattered text across the image',
        tessedit_pageseg_mode: '11'
    },

    // PSM 3: Fully automatic page segmentation (default)
    AUTO: {
        psm: 3,
        name: 'Auto',
        description: 'Automatic detection - fallback option',
        tessedit_pageseg_mode: '3'
    },

    // PSM 7: Treat image as a single text line
    SINGLE_LINE: {
        psm: 7,
        name: 'Single Line',
        description: 'For single-line text like tickers in headers',
        tessedit_pageseg_mode: '7'
    }
};

/**
 * OCR Configuration for ticker detection
 */
export const TICKER_CONFIG = {
    // Character whitelist optimized for ticker symbols
    tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$-/.:',

    // Improve accuracy for uppercase text
    tessedit_char_blacklist: 'abcdefghijklmnopqrstuvwxyz',

    // Preserve interword spaces
    preserve_interword_spaces: '1'
};

/**
 * OCR Configuration for price detection
 */
export const PRICE_CONFIG = {
    // Character whitelist for numbers and price symbols
    tessedit_char_whitelist: '0123456789,.$ ',

    // Enable numeric mode
    classify_bln_numeric_mode: '1'
};

/**
 * General high-accuracy configuration
 */
export const HIGH_ACCURACY_CONFIG = {
    // Character whitelist for alphanumeric + common symbols
    tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$-/.:,% ',

    // Improve word recognition
    tessedit_enable_dict_correction: '0', // Disable dictionary (we want exact text)

    // Preserve spaces
    preserve_interword_spaces: '1'
};

/**
 * Get recommended OCR configurations for multi-pass recognition
 */
export const getMultiPassConfigs = () => {
    return [
        {
            name: 'ticker_sparse',
            config: { ...TICKER_CONFIG, ...PSM_MODES.SPARSE_TEXT },
            priority: 1,
            description: 'Optimized for ticker symbols with sparse text mode'
        },
        {
            name: 'ticker_uniform',
            config: { ...TICKER_CONFIG, ...PSM_MODES.UNIFORM_BLOCK },
            priority: 2,
            description: 'Ticker detection with uniform block mode'
        },
        {
            name: 'single_line',
            config: { ...TICKER_CONFIG, ...PSM_MODES.SINGLE_LINE },
            priority: 3,
            description: 'Single line mode for header tickers'
        },
        {
            name: 'high_accuracy_auto',
            config: { ...HIGH_ACCURACY_CONFIG, ...PSM_MODES.AUTO },
            priority: 4,
            description: 'High accuracy with automatic segmentation'
        }
    ];
};

/**
 * Filter OCR results based on confidence scores
 */
export const filterByConfidence = (ocrResult, minWordConfidence = 60, minAvgConfidence = 70) => {
    if (!ocrResult || !ocrResult.data) return null;

    const { words, confidence } = ocrResult.data;

    // V5.3 Improvement: If we found a very high confidence word that looks like a ticker, 
    // we can be more lenient on the overall average confidence.
    const potentialTicker = words?.find(w => /^[A-Z]{2,10}$/.test(w.text) && w.confidence > 85);
    const effectiveMinAvg = potentialTicker ? Math.min(minAvgConfidence, 50) : minAvgConfidence;

    // Check average confidence
    if (confidence < effectiveMinAvg) {
        console.log(`[OCR] Result rejected: avg confidence ${confidence}% < ${effectiveMinAvg}%`);
        return null;
    }

    // Filter words by confidence
    const highConfidenceWords = words?.filter(word =>
        word.confidence >= minWordConfidence && word.text.trim().length > 0
    ) || [];

    if (highConfidenceWords.length === 0) {
        console.log(`[OCR] No high-confidence words found (min: ${minWordConfidence}%)`);
        return null;
    }

    // Reconstruct text from high-confidence words
    const filteredText = highConfidenceWords.map(w => w.text).join(' ');

    return {
        text: filteredText,
        confidence: confidence,
        wordCount: highConfidenceWords.length,
        words: highConfidenceWords
    };
};

/**
 * Validate detected ticker against known symbols
 */
export const validateTicker = (ticker, knownTickers) => {
    if (!ticker) return false;

    const upperTicker = ticker.toUpperCase();

    // Check if it's in our known list
    if (knownTickers[upperTicker]) {
        return true;
    }

    // Check if it matches common ticker patterns
    const tickerPattern = /^[A-Z]{1,5}$/; // 1-5 uppercase letters
    if (tickerPattern.test(upperTicker)) {
        return true;
    }

    // Check for crypto pairs (e.g., BTC/USD, BTCUSDT)
    const cryptoPairPattern = /^[A-Z]{2,10}[\/\-]?(USD|USDT|BUSD|EUR|BTC|ETH)$/;
    if (cryptoPairPattern.test(upperTicker)) {
        return true;
    }

    return false;
};
