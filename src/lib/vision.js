
// Visual Analysis Logic to extract chart data from images when OCR fails
// This converts image pixels into a time-series array for the Neural/Technical engines

export const extractChartData = (imageSrc) => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            try {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                // Downscale for performance and noise reduction
                const width = 200;
                const scale = width / img.width;
                const height = Math.floor(img.height * scale);

                canvas.width = width;
                canvas.height = height;

                // Draw high contrast version
                ctx.filter = 'grayscale(100%) contrast(150%)';
                ctx.drawImage(img, 0, 0, width, height);

                const imageData = ctx.getImageData(0, 0, width, height);
                const data = imageData.data;
                const points = [];

                // Auto-detect Light/Dark mode
                let totalBrightness = 0;
                const totalPixels = width * height;
                for (let i = 0; i < totalPixels * 4; i += 4) {
                    totalBrightness += data[i];
                }
                const avgBrightness = totalBrightness / totalPixels;
                const isLightMode = avgBrightness > 127;

                // Scan columns (left to right)
                for (let x = 0; x < width; x++) {
                    let totalY = 0;
                    let count = 0;

                    // Scan rows (top to bottom)
                    for (let y = 0; y < height; y++) {
                        const i = (y * width + x) * 4;
                        const r = data[i]; // Grayscale, so r=g=b

                        // Logic: Find pixels that CONTRAST with the background
                        let isSignal = false;
                        if (isLightMode) {
                            // Light Background -> Look for Dark pixels (Trend line)
                            if (r < 100) isSignal = true;
                        } else {
                            // Dark Background -> Look for Bright pixels
                            if (r > 155) isSignal = true;
                        }

                        if (isSignal) {
                            totalY += y;
                            count++;
                        }
                    }

                    // Validity Filter:
                    // If a column is mostly signal, it's probably not a line (maybe a grid line or axis).
                    // We expect the line to be thin (e.g., < 20% of height, but let's be generous < 50%)
                    if (count > 0 && count < height * 0.5) {
                        const avgY = totalY / count;
                        // Invert Y so graph goes up
                        points.push(height - avgY);
                    }
                }

                // Quality Guardrails -> Geometric Confidence (c_i)
                let verticalJumps = 0;
                for (let i = 1; i < points.length; i++) {
                    if (Math.abs(points[i] - points[i - 1]) > height * 0.3) verticalJumps++;
                }

                const continuity = points.length / width;
                const entropy = verticalJumps / (points.length || 1);

                // Final Geometric Confidence Calculation
                let geometricConfidence = (continuity * 0.7) + ((1 - Math.min(1, entropy * 2)) * 0.3);
                geometricConfidence = Math.max(0, Math.min(1, geometricConfidence));

                // Final check for minimum points
                if (points.length < 20 || geometricConfidence < 0.2) {
                    resolve(null);
                    return;
                }

                resolve({
                    points,
                    confidence: geometricConfidence,
                    isLightMode
                });

            } catch (err) {
                console.error("Visual Extraction Error:", err);
                resolve(null);
            }
        };
        img.onerror = (err) => resolve(null);
        img.src = imageSrc;
    });
};

export const anchorPriceToVisual = (points, anchorPrice) => {
    if (!points || points.length === 0 || !anchorPrice) return points;
    const lastPoint = points[points.length - 1];
    // If the last point is 0, we can't scale it properly
    if (lastPoint === 0) return points;

    const scale = anchorPrice / lastPoint;
    return points.map(p => p * scale);
};
