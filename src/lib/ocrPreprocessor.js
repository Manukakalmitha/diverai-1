/**
 * Advanced OCR Image Preprocessing Module
 * Provides multiple preprocessing strategies to improve Tesseract.js accuracy
 */

/**
 * Apply adaptive thresholding to convert image to binary (black/white)
 * Optimized version using Integral Image (Summed-Area Table) - O(N)
 */
export const applyAdaptiveThreshold = (canvas, ctx) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;

    // Step 1: Create Integral Image
    // Using Float64Array to prevent overflow for large images
    const integral = new Float64Array(width * height);

    for (let y = 0; y < height; y++) {
        let rowSum = 0;
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const gray = data[idx]; // Assume grayscale
            rowSum += gray;
            if (y === 0) {
                integral[y * width + x] = rowSum;
            } else {
                integral[y * width + x] = integral[(y - 1) * width + x] + rowSum;
            }
        }
    }

    // Step 2: Apply threshold using O(1) window sum lookups
    const radius = 15;
    const k = 0.15; // Slightly more sensitive for OCR
    const output = new Uint8ClampedArray(data.length);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;

            const y1 = Math.max(0, y - radius);
            const y2 = Math.min(height - 1, y + radius);
            const x1 = Math.max(0, x - radius);
            const x2 = Math.min(width - 1, x + radius);

            const count = (y2 - y1 + 1) * (x2 - x1 + 1);

            // Formula: Sum = I(x2, y2) - I(x1-1, y2) - I(x2, y1-1) + I(x1-1, y1-1)
            let sum = integral[y2 * width + x2];
            if (x1 > 0) sum -= integral[y2 * width + (x1 - 1)];
            if (y1 > 0) sum -= integral[(y1 - 1) * width + x2];
            if (x1 > 0 && y1 > 0) sum += integral[(y1 - 1) * width + (x1 - 1)];

            const localMean = sum / count;
            const threshold = localMean * (1 - k);

            const value = data[idx] > threshold ? 255 : 0;
            output[idx] = output[idx + 1] = output[idx + 2] = value;
            output[idx + 3] = 255; // Alpha
        }
    }

    ctx.putImageData(new ImageData(output, width, height), 0, 0);
};

/**
 * Apply median blur to reduce noise while preserving edges
 * Optimized version with reduced allocations
 */
export const applyMedianBlur = (canvas, ctx, radius = 1) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;
    const output = new Uint8ClampedArray(data.length);

    // Copy alpha channel and handle edges
    output.set(data);

    const windowSize = (radius * 2 + 1) * (radius * 2 + 1);
    const window = new Uint8Array(windowSize);

    for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
            const idx = (y * width + x) * 4;

            // Collect neighboring pixel values (Grayscale assumption)
            let winIdx = 0;
            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    window[winIdx++] = data[((y + dy) * width + (x + dx)) * 4];
                }
            }

            // Sort window values
            window.sort();
            const median = window[Math.floor(windowSize / 2)];

            output[idx] = output[idx + 1] = output[idx + 2] = median;
            // index idx+3 (alpha) is already set by output.set(data)
        }
    }

    ctx.putImageData(new ImageData(output, width, height), 0, 0);
};

/**
 * Apply unsharp mask to enhance edges and text clarity
 */
export const applySharpen = (canvas, ctx) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;

    // Sharpening kernel
    const kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];

    const output = new Uint8ClampedArray(data.length);
    output.set(data);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;

            // Unrolled 3x3 sharpening kernel: [0, -1, 0, -1, 5, -1, 0, -1, 0]
            let sum = data[idx] * 5;
            sum -= data[idx - 4];             // x-1, y
            sum -= data[idx + 4];             // x+1, y
            sum -= data[idx - width * 4];     // x, y-1
            sum -= data[idx + width * 4];     // x, y+1

            const b = Math.max(0, Math.min(255, sum));
            output[idx] = output[idx + 1] = output[idx + 2] = b;
        }
    }

    ctx.putImageData(new ImageData(output, width, height), 0, 0);
};

/**
 * Detect and extract Region of Interest (ROI) - typically top-left corner for ticker
 */
export const extractROI = (canvas, ctx, region = 'top-left') => {
    const width = canvas.width;
    const height = canvas.height;

    let x = 0, y = 0, w = width, h = height;

    switch (region) {
        case 'top-left':
            w = Math.floor(width * 0.4);
            h = Math.floor(height * 0.22); // Increased height to capture ticker + price info
            break;
        case 'top-center':
            x = Math.floor(width * 0.35);
            w = Math.floor(width * 0.3);
            h = Math.floor(height * 0.15);
            break;
        case 'left-axis':
            w = Math.floor(width * 0.1);
            break;
        case 'right-axis':
            x = Math.floor(width * 0.88);
            w = Math.floor(width * 0.12);
            break;
        default:
            return canvas.toDataURL('image/png');
    }

    // V5.3 Diagnostic: Track down why ultra-small ROIs are being passed
    if (w < 40 || h < 40) {
        console.warn(`[OCR] ROI suspected too small (${w}x${h}) from region ${region}. Input canvas size: ${width}x${height}. Skipping to avoid Tesseract WASM errors.`);
        return canvas.toDataURL('image/png');
    }

    const roiCanvas = document.createElement('canvas');
    roiCanvas.width = w;
    roiCanvas.height = h;
    const roiCtx = roiCanvas.getContext('2d', { willReadFrequently: true });

    const imageData = ctx.getImageData(x, y, w, h);
    roiCtx.putImageData(imageData, 0, 0);

    return roiCanvas.toDataURL('image/png');
};

/**
 * Generate multiple preprocessed variants of the image
 * Returns an array of preprocessed image data URLs
 */
export const generatePreprocessedVariants = (imageElement) => {
    const variants = [];
    const width = imageElement.width;
    const scale = Math.max(1, 1200 / width); // Higher resolution for better OCR
    const w = width * scale;
    const h = imageElement.height * scale;

    // Detect if image is dark mode - Optimized Sampling Pass (V5.6)
    const testCanvas = document.createElement('canvas');
    testCanvas.width = 100; // Small proxy for brightness check
    testCanvas.height = 100;
    const testCtx = testCanvas.getContext('2d', { willReadFrequently: true });
    testCtx.drawImage(imageElement, 0, 0, 100, 100);
    const testData = testCtx.getImageData(0, 0, 100, 100).data;
    let totalBrightness = 0;
    const sampleStep = 4; // Sample every 4th pixel even on the small canvas
    let count = 0;
    for (let i = 0; i < testData.length; i += 4 * sampleStep) {
        totalBrightness += (testData[i] + testData[i + 1] + testData[i + 2]) / 3;
        count++;
    }
    const avgBrightness = totalBrightness / count;
    const isDark = avgBrightness < 120;

    // Variant 1: High Contrast + Grayscale (Original method, improved)
    const canvas1 = document.createElement('canvas');
    canvas1.width = w;
    canvas1.height = h;
    const ctx1 = canvas1.getContext('2d', { willReadFrequently: true });
    ctx1.filter = isDark
        ? 'invert(100%) grayscale(100%) contrast(180%) brightness(120%)'
        : 'grayscale(100%) contrast(180%) brightness(110%)';
    ctx1.drawImage(imageElement, 0, 0, w, h);
    variants.push({
        name: 'high_contrast',
        dataUrl: canvas1.toDataURL('image/jpeg', 0.85), // Fast encoding
        description: 'High contrast grayscale'
    });

    // Variant 2: Adaptive Threshold + Noise Reduction
    const canvas2 = document.createElement('canvas');
    canvas2.width = w;
    canvas2.height = h;
    const ctx2 = canvas2.getContext('2d', { willReadFrequently: true });
    ctx2.filter = 'grayscale(100%)';
    ctx2.drawImage(imageElement, 0, 0, w, h);
    applyMedianBlur(canvas2, ctx2, 1); // V4 Addition: Median blur to reduce noise
    applyAdaptiveThreshold(canvas2, ctx2);
    if (isDark) {
        ctx2.filter = 'invert(100%)';
        ctx2.drawImage(canvas2, 0, 0);
    }
    variants.push({
        name: 'adaptive_threshold_v4',
        dataUrl: canvas2.toDataURL('image/jpeg', 0.8), // Binary images compress extremely fast as JPEG
        description: 'Adaptive binary with noise reduction'
    });

    // Variant 3: Multi-Stage Precision (V4 ADDITION)
    const canvas3 = document.createElement('canvas');
    canvas3.width = w;
    canvas3.height = h;
    const ctx3 = canvas3.getContext('2d', { willReadFrequently: true });
    ctx3.filter = isDark ? 'invert(100%) grayscale(100%)' : 'grayscale(100%)';
    ctx3.drawImage(imageElement, 0, 0, w, h);
    applySharpen(canvas3, ctx3);
    applyMedianBlur(canvas3, ctx3, 1);
    ctx3.filter = 'contrast(160%) brightness(110%)';
    ctx3.drawImage(canvas3, 0, 0);
    variants.push({
        name: 'precision_v4',
        dataUrl: canvas3.toDataURL('image/jpeg', 0.85),
        description: 'Sharpened + Denoised High Contrast'
    });

    // ROI Generation
    const roiDataUrl = extractROI(canvas1, ctx1, 'top-left');
    variants.push({
        name: 'roi_top_left',
        dataUrl: roiDataUrl,
        description: 'Top-left region extraction'
    });

    // Variant 5: ROI - Right Axis (Price scale)
    const roiRightDataUrl = extractROI(canvas3, ctx3, 'right-axis');
    variants.push({
        name: 'roi_right_axis',
        dataUrl: roiRightDataUrl,
        description: 'Right-axis price extraction'
    });

    return variants;
};
