import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import fs from 'fs';

export default defineConfig(({ mode }) => {
    // Standard Vite env loading
    const env = loadEnv(mode, process.cwd(), '');

    // Fallback: Read .env manually if loadEnv misses it (sometimes happens in non-standard build scripts)
    let finalEnv = { ...env };
    try {
        const dotEnvPath = resolve(process.cwd(), '.env');
        if (fs.existsSync(dotEnvPath)) {
            const content = fs.readFileSync(dotEnvPath, 'utf8');
            content.split('\n').forEach(line => {
                const [key, ...valueParts] = line.split('=');
                if (key && valueParts.length > 0) {
                    const value = valueParts.join('=').trim();
                    if (!finalEnv[key.trim()]) finalEnv[key.trim()] = value;
                }
            });
        }
    } catch (e) { console.error("Manual .env parse error:", e); }

    return {
        base: './',
        mode: 'production',
        define: {
            'process.env.NODE_ENV': JSON.stringify('production'),
            'import.meta.env.VITE_SUPABASE_URL': JSON.stringify(finalEnv.VITE_SUPABASE_URL),
            'import.meta.env.VITE_SUPABASE_ANON_KEY': JSON.stringify(finalEnv.VITE_SUPABASE_ANON_KEY),
        },
        plugins: [
            react(),
            {
                name: 'extension-setup',
                writeBundle() {
                    const distPath = resolve(__dirname, 'dist-extension');
                    fs.copyFileSync(
                        resolve(__dirname, 'chrome-extension/manifest.json'),
                        resolve(distPath, 'manifest.json')
                    );
                    const nestedHtml = resolve(distPath, 'chrome-extension/sidepanel.html');
                    const rootHtml = resolve(distPath, 'sidepanel.html');
                    if (fs.existsSync(nestedHtml)) {
                        let html = fs.readFileSync(nestedHtml, 'utf8');
                        // Fix relative paths for moved file
                        html = html.replace(/src="\.\.\//g, 'src="./');
                        html = html.replace(/href="\.\.\//g, 'href="./');
                        fs.writeFileSync(rootHtml, html);
                        fs.unlinkSync(nestedHtml);
                        fs.rmdirSync(resolve(distPath, 'chrome-extension'));
                    }
                }
            }
        ],
        build: {
            outDir: 'dist-extension',
            emptyOutDir: true,
            rollupOptions: {
                input: {
                    sidepanel: resolve(__dirname, 'chrome-extension/sidepanel.html'),
                    background: resolve(__dirname, 'chrome-extension/background.js'),
                    content: resolve(__dirname, 'chrome-extension/content.js'),
                },
                output: {
                    entryFileNames: '[name].js',
                    chunkFileNames: 'assets/[name].js',
                    assetFileNames: 'assets/[name].[ext]',
                },
            },
        },
        resolve: {
            alias: {
                '@': resolve(__dirname, './src'),
            },
        },
    };
});
