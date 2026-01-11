import { useEffect } from 'react';

/**
 * Custom hook for managing page-specific meta tags
 * Lightweight alternative to react-helmet that works with React 19
 * 
 * @param {Object} config - SEO configuration
 * @param {string} config.title - Page title
 * @param {string} config.description - Page description  
 * @param {string} config.keywords - Meta keywords (comma separated)
 * @param {string} config.canonicalUrl - Canonical URL for the page
 * @param {string} config.ogImage - Open Graph image URL
 * @param {string} config.ogType - Open Graph type (default: 'website')
 * @param {Object} config.schema - JSON-LD schema object
 */
export const useSEO = ({
    title,
    description,
    keywords,
    canonicalUrl,
    ogImage = 'https://diverai.flisoft.agency/og-image.png',
    ogType = 'website',
    schema
}) => {
    useEffect(() => {
        // Store original values
        const originalTitle = document.title;

        // Update title
        if (title) {
            document.title = title;
        }

        // Helper to update meta tag
        const updateMetaTag = (property, content, isProperty = false) => {
            if (!content) return null;

            const selector = isProperty
                ? `meta[property="${property}"]`
                : `meta[name="${property}"]`;

            let element = document.querySelector(selector);

            if (!element) {
                element = document.createElement('meta');
                if (isProperty) {
                    element.setAttribute('property', property);
                } else {
                    element.setAttribute('name', property);
                }
                document.head.appendChild(element);
            }

            element.setAttribute('content', content);
            return element;
        };

        // Update meta tags
        const createdElements = [];

        if (description) {
            createdElements.push(updateMetaTag('description', description));
            createdElements.push(updateMetaTag('og:description', description, true));
            createdElements.push(updateMetaTag('twitter:description', description));
        }

        if (keywords) {
            createdElements.push(updateMetaTag('keywords', keywords));
        }

        if (title) {
            createdElements.push(updateMetaTag('og:title', title, true));
            createdElements.push(updateMetaTag('twitter:title', title));
        }

        if (ogImage) {
            createdElements.push(updateMetaTag('og:image', ogImage, true));
            createdElements.push(updateMetaTag('twitter:image', ogImage));
        }

        if (ogType) {
            createdElements.push(updateMetaTag('og:type', ogType, true));
        }

        // Update canonical URL
        let canonicalElement = document.querySelector('link[rel="canonical"]');
        if (canonicalUrl) {
            if (!canonicalElement) {
                canonicalElement = document.createElement('link');
                canonicalElement.setAttribute('rel', 'canonical');
                document.head.appendChild(canonicalElement);
            }
            canonicalElement.setAttribute('href', canonicalUrl);
        }

        // Add page-specific JSON-LD schema
        let schemaScript = null;
        if (schema) {
            schemaScript = document.createElement('script');
            schemaScript.type = 'application/ld+json';
            schemaScript.id = 'page-schema';
            schemaScript.textContent = JSON.stringify(schema);

            // Remove existing page schema if present
            const existingSchema = document.getElementById('page-schema');
            if (existingSchema) {
                existingSchema.remove();
            }

            document.head.appendChild(schemaScript);
        }

        // Cleanup on unmount
        return () => {
            document.title = originalTitle;

            if (schemaScript) {
                schemaScript.remove();
            }
        };
    }, [title, description, keywords, canonicalUrl, ogImage, ogType, schema]);
};

/**
 * Pre-configured SEO configs for common pages
 */
export const SEO_CONFIGS = {
    home: {
        title: 'Diver AI | AI Stock Analysis & Crypto Trading Signals',
        description: 'Get institutional-grade trading signals using AI pattern recognition. Analyze any stock or crypto chart instantly. Free trial available.',
        keywords: 'AI stock analysis, trading signals, crypto analysis, pattern recognition, stock prediction',
        canonicalUrl: 'https://diverai.flisoft.agency/'
    },
    analysis: {
        title: 'AI Chart Analysis Terminal | Diver AI',
        description: 'Upload or scan any stock, crypto, or forex chart for instant AI analysis. Get trading signals, pattern recognition, and price predictions.',
        keywords: 'chart analysis, AI scanner, trading terminal, stock screener, crypto scanner',
        canonicalUrl: 'https://diverai.flisoft.agency/analysis'
    },
    pricing: {
        title: 'Pricing Plans | Diver AI Trading Analysis',
        description: 'Choose your Diver AI plan. Start free with 3 daily scans or go Pro for unlimited AI analysis, priority signals, and advanced features.',
        keywords: 'AI trading price, stock analysis cost, trading software pricing',
        canonicalUrl: 'https://diverai.flisoft.agency/pricing'
    },
    updates: {
        title: 'Product Updates & Changelog | Diver AI',
        description: 'Stay up to date with the latest Diver AI features, improvements, and fixes. See what\'s new in the AI trading platform.',
        keywords: 'Diver AI updates, new features, changelog, product news',
        canonicalUrl: 'https://diverai.flisoft.agency/updates',
        schema: {
            '@context': 'https://schema.org',
            '@type': 'TechArticle',
            'headline': 'Diver AI Product Updates',
            'description': 'Latest features and improvements to the Diver AI trading analysis platform',
            'datePublished': '2026-01-11',
            'dateModified': '2026-01-11',
            'author': {
                '@type': 'Organization',
                'name': 'Diver AI'
            }
        }
    },
    docs: {
        title: 'Documentation | Diver AI',
        description: 'Learn how to use Diver AI for stock and crypto analysis. Guides, tutorials, and API documentation.',
        keywords: 'Diver AI docs, trading AI guide, how to use, tutorial',
        canonicalUrl: 'https://diverai.flisoft.agency/docs'
    }
};

export default useSEO;
