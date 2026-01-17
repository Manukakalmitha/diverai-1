/**
 * Advanced OCR Image Preprocessing Module
 * Provides multiple preprocessing strategies to improve Tesseract.js accuracy
 */

/**
 * Apply adaptive thresholding to convert image to binary (black/white)
 * This helps with low-contrast images and varying lighting conditions
 */
export const applyAdaptiveThreshold = (canvas, ctx) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;

    // Calculate local threshold for each pixel using a sliding window
    const windowSize = 15;
    const k = 0.2; // Sensitivity parameter

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;

            // Calculate local mean in window
            let sum = 0;
            let count = 0;
            for (let wy = Math.max(0, y - windowSize); wy < Math.min(height, y + windowSize); wy++) {
                for (let wx = Math.max(0, x - windowSize); wx < Math.min(width, x + windowSize); wx++) {
                    const widx = (wy * width + wx) * 4;
                    sum += data[widx]; // Grayscale, so R = G = B
                    count++;
                }
            }
            const localMean = sum / count;
            const threshold = localMean * (1 - k);

            // Apply threshold
            const value = data[idx] > threshold ? 255 : 0;
            data[idx] = data[idx + 1] = data[idx + 2] = value;
        }
    }

    ctx.putImageData(imageData, 0, 0);
};

/**
 * Apply median blur to reduce noise while preserving edges
 */
export const applyMedianBlur = (canvas, ctx, radius = 1) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;
    const output = new Uint8ClampedArray(data);

    for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
            const idx = (y * width + x) * 4;
            const values = [];

            // Collect neighboring pixel values
            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    const nidx = ((y + dy) * width + (x + dx)) * 4;
                    values.push(data[nidx]);
                }
            }

            // Sort and get median
            values.sort((a, b) => a - b);
            const median = values[Math.floor(values.length / 2)];

            output[idx] = output[idx + 1] = output[idx + 2] = median;
        }
    }

    for (let i = 0; i < data.length; i++) {
        data[i] = output[i];
    }

    ctx.putImageData(imageData, 0, 0);
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

    const output = new Uint8ClampedArray(data);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = ((y + ky) * width + (x + kx)) * 4;
                    const kernelIdx = (ky + 1) * 3 + (kx + 1);
                    sum += data[idx] * kernel[kernelIdx];
                }
            }
            const idx = (y * width + x) * 4;
            const value = Math.max(0, Math.min(255, sum));
            output[idx] = output[idx + 1] = output[idx + 2] = value;
        }
    }

    for (let i = 0; i < data.length; i++) {
        data[i] = output[i];
    }

    ctx.putImageData(imageData, 0, 0);
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
        default:
            return canvas.toDataURL('image/png');
    }

    const roiCanvas = document.createElement('canvas');
    roiCanvas.width = w;
    roiCanvas.height = h;
    const roiCtx = roiCanvas.getContext('2d', { willReadFrequently: true });

    const imageData = ctx.getImageData(x, y, w, h);
    roiCtx.putImageData(imageData, 0, 0);

    // V5.2 Fix: Prevent "too small to scale" errors by ensuring minimum dimensions
    // Tesseract.js WASM engine often fails on images smaller than 3-5px in any dimension.
    if (w < 10 || h < 10) {
        console.warn(`[OCR] ROI too small (${w}x${h}), skipping region extraction`);
        return canvas.toDataURL('image/png');
    }

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

    // Detect if image is dark mode
    const testCanvas = document.createElement('canvas');
    testCanvas.width = w;
    testCanvas.height = h;
    const testCtx = testCanvas.getContext('2d', { willReadFrequently: true });
    testCtx.drawImage(imageElement, 0, 0, w, h);
    const testData = testCtx.getImageData(0, 0, w, h).data;
    let totalBrightness = 0;
    for (let i = 0; i < testData.length; i += 4) {
        totalBrightness += (testData[i] + testData[i + 1] + testData[i + 2]) / 3;
    }
    const avgBrightness = totalBrightness / (testData.length / 4);
    const isDark = avgBrightness < 128;

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
        dataUrl: canvas1.toDataURL('image/png'),
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
        dataUrl: canvas2.toDataURL('image/png'),
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
        dataUrl: canvas3.toDataURL('image/png'),
        description: 'Sharpened + Denoised High Contrast'
    });

    // Variant 4: ROI - Top Left (where ticker usually is)
    const roiDataUrl = extractROI(canvas1, ctx1, 'top-left');
    variants.push({
        name: 'roi_top_left',
        dataUrl: roiDataUrl,
        description: 'Top-left region extraction'
    });

    return variants;
};
