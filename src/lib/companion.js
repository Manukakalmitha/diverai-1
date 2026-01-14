/**
 * Companion API - Enables a floating overlay on mobile using Picture-in-Picture
 */

export class CompanionManager {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 300;
        this.canvas.height = 150;
        this.ctx = this.canvas.getContext('2d');
        this.video = document.createElement('video');
        this.video.muted = true;
        this.video.playsInline = true;
        this.stream = null;
    }

    async start() {
        if (!document.pictureInPictureEnabled) {
            throw new Error("PiP not supported on this device.");
        }

        // Draw initial state
        this.update({ ticker: 'DiverAI', direction: 'Ready', confidence: '0' });

        this.stream = this.canvas.captureStream(10); // 10 FPS is enough
        this.video.srcObject = this.stream;

        await this.video.play();
        return await this.video.requestPictureInPicture();
    }

    update(result) {
        if (!this.ctx) return;
        const { ticker, direction, confidence, status } = result;

        const isAnalyzing = status === 'analyzing';
        const isBull = !isAnalyzing && direction?.toLowerCase().includes('bullish');
        const isBear = !isAnalyzing && direction?.toLowerCase().includes('bearish');
        const color = isAnalyzing ? '#94a3b8' : (isBull ? '#10b981' : (isBear ? '#f43f5e' : '#3b82f6'));

        // Clear
        this.ctx.fillStyle = '#020617';
        this.ctx.fillRect(0, 0, 300, 150);

        // Gradient Background
        const grad = this.ctx.createLinearGradient(0, 0, 0, 150);
        grad.addColorStop(0, '#0f172a');
        grad.addColorStop(1, '#020617');
        this.ctx.fillStyle = grad;
        this.ctx.fillRect(0, 0, 300, 150);

        // Border
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 6;
        this.ctx.strokeRect(0, 0, 300, 150);

        // Header
        this.ctx.fillStyle = '#64748b';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.fillText('DIVER AI COMPANION', 20, 30);

        // Ticker
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 36px sans-serif';
        this.ctx.fillText(isAnalyzing ? 'SCANNIG...' : ticker, 20, 75);

        // Direction Pill
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        if (this.ctx.roundRect) {
            this.ctx.roundRect(20, 90, 160, 28, 6);
        } else {
            this.ctx.rect(20, 90, 160, 28);
        }
        this.ctx.fill();

        this.ctx.fillStyle = '#000000';
        this.ctx.font = 'bold 16px sans-serif';
        const directionText = isAnalyzing ? 'PROCESSING' : direction.toUpperCase();
        this.ctx.fillText(directionText, 32, 110);

        // Confidence
        if (!isAnalyzing) {
            this.ctx.fillStyle = '#94a3b8';
            this.ctx.font = 'bold 12px sans-serif';
            this.ctx.fillText('CONFIDENCE', 200, 100);

            this.ctx.fillStyle = color;
            this.ctx.font = 'bold 24px monospace';
            this.ctx.fillText(`${confidence}%`, 200, 130);
        }

        // Pulse Indicator
        const time = Date.now() / 1000;
        const pulse = Math.abs(Math.sin(time * (isAnalyzing ? 5 : 2)));
        this.ctx.fillStyle = isAnalyzing ? '#fbbf24' : '#10b981';
        this.ctx.beginPath();
        this.ctx.arc(260, 35, 8 * pulse, 0, Math.PI * 2);
        this.ctx.fill();
    }

    async stop() {
        if (document.pictureInPictureElement) {
            await document.exitPictureInPicture();
        }
    }
}

export const companion = new CompanionManager();
