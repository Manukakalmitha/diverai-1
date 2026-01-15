import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.js',
      includeAssets: ['favicon.png', 'apple-touch-icon.png', 'widget_ticker.html', 'icon-16.png', 'icon-48.png'],
      manifest: {
        id: 'com.diverai.app',
        name: 'Diver AI Trading',
        short_name: 'Diver AI',
        description: '99% Accurate Chart Pattern Analysis | Quantum Trading Engine',
        theme_color: '#020617',
        background_color: '#020617',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        scope: '/',
        categories: ['finance', 'investment', 'utilities'],
        share_target: {
          action: '/share-target',
          method: 'POST',
          enctype: 'multipart/form-data',
          params: {
            files: [
              {
                name: 'files',
                accept: ['image/*']
              }
            ]
          }
        },
        shortcuts: [
          {
            name: 'Analysis Terminal',
            short_name: 'Terminal',
            description: 'Open AI Trading Terminal',
            url: '/analysis',
            icons: [{ src: 'pwa-192x192-v2.png', sizes: '192x192' }]
          },
          {
            name: 'Subscription Plans',
            short_name: 'Pricing',
            description: 'Check Premium Plans',
            url: '/pricing',
            icons: [{ src: 'pwa-192x192-v2.png', sizes: '192x192' }]
          },
          {
            name: 'Documentation',
            short_name: 'Docs',
            description: 'Learn how to use Diver AI',
            url: '/docs',
            icons: [{ src: 'pwa-192x192-v2.png', sizes: '192x192' }]
          },
          {
            name: 'My Account',
            short_name: 'Account',
            description: 'Manage your profile',
            url: '/profile',
            icons: [{ src: 'pwa-192x192-v2.png', sizes: '192x192' }]
          }
        ],
        icons: [
          {
            src: 'pwa-192x192-v2.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512-v2.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512-v2.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      }
    })
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-tf': ['@tensorflow/tfjs'],
          'vendor-ui': ['framer-motion', 'lucide-react'],
          'vendor-utils': ['@supabase/supabase-js', 'tesseract.js']
        }
      }
    }
  }
})

