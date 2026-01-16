
/**
 * Mobile-Optimized Vision Library
 * Handles cropping, denoising, and contrast enhancement for mobile screenshots (TWA/PWA)
 */

export const detectChartROI = (canvas, ctx) => {
    const width = canvas.width;
    const height = canvas.height;
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // 1. Identify "Chart Area" by searching for the density of gray/black grid lines or gradients
    // Mobile status bars (top) and nav bars (bottom) are usually solid colors or different textures.

    let topCrop = 0;
    let bottomCrop = height;

    // Scan from top to find where the status bar ends (usually within top 15%)
    for (let y = 0; y < height * 0.15; y++) {
        let uniqueColors = new Set();
        for (let x = 0; x < width; x += 10) {
            const i = (y * width + x) * 4;
            const rgb = `${data[i]},${data[i + 1]},${data[i + 2]}`;
            uniqueColors.add(rgb);
        }
        // If row has many colors, it's likely where the chart/content starts
        if (uniqueColors.size > 15) {
            topCrop = y;
            break;
        }
    }

    // Scan from bottom to find where the nav bar/toolbar ends (usually within bottom 20%)
    for (let y = height - 1; y > height * 0.8; y--) {
        let uniqueColors = new Set();
        for (let x = 0; x < width; x += 10) {
            const i = (y * width + x) * 4;
            const rgb = `${data[i]},${data[i + 1]},${data[i + 2]}`;
            uniqueColors.add(rgb);
        }
        if (uniqueColors.size > 15) {
            bottomCrop = y;
            break;
        }
    }

    return { top: topCrop, bottom: bottomCrop, left: 0, right: width };
};

export const enhanceMobileChart = (imgElement) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Scale up for better OCR of tiny mobile text
    const scale = 2.0;
    canvas.width = imgElement.width * scale;
    canvas.height = imgElement.height * scale;

    // Apply adaptive sharpening and contrast
    ctx.filter = 'contrast(130%) brightness(105%) saturate(0%)';
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

    // Initial ROI detection
    const roi = detectChartROI(canvas, ctx);

    // Create final cropped and enhanced canvas
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = canvas.width;
    finalCanvas.height = (roi.bottom - roi.top);
    const finalCtx = finalCanvas.getContext('2d');

    finalCtx.drawImage(canvas,
        0, roi.top, canvas.width, (roi.bottom - roi.top), // Source
        0, 0, canvas.width, (roi.bottom - roi.top)        // Dest
    );

    return finalCanvas.toDataURL('image/png', 1.0);
};

export const denoiseChart = (canvas) => {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Simple grid-reduction filter:
    // Grid lines are often static H/V lines. If we see a very long thin line, we can dim it.
    // However, for neural analysis, we just want to highlight the main trend.

    // We'll use a local-variance mask to preserve edges (candles/lines) but flatten noisy backgrounds.
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        // High-pass filter for clarity
        data[i] = data[i + 1] = data[i + 2] = avg > 150 ? 255 : (avg < 50 ? 0 : avg);
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
};
