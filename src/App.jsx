import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Analytics } from '@vercel/analytics/react';
import { AppProvider } from './context/AppContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import ReloadPrompt from './components/ReloadPrompt';
import { Loader2 } from 'lucide-react';

// Lazy Load Heavy Components
const Terminal = lazy(() => import('./pages/Terminal'));
const PricingPage = lazy(() => import('./pages/PricingPage'));
const DocsPage = lazy(() => import('./pages/DocsPage'));
const LegalPage = lazy(() => import('./pages/LegalPage'));
const ProfilePage = lazy(() => import('./pages/ProfilePage'));
const WhatIsDiverAI = lazy(() => import('./pages/WhatIsDiverAI'));
const ArticlesHub = lazy(() => import('./pages/ArticlesHub'));

const ArticlePage = lazy(() => import('./pages/ArticlePage'));
const AuthPage = lazy(() => import('./pages/AuthPage'));
const UpdatesPage = lazy(() => import('./pages/UpdatesPage'));
const ReferralPage = lazy(() => import('./pages/ReferralPage'));

const LoadingFallback = () => (
  <div className="min-h-screen bg-black flex items-center justify-center">
    <div className="text-center space-y-4">
      <div className="relative mx-auto w-12 h-12">
        <div className="absolute inset-0 bg-brand/20 rounded-full animate-ping"></div>
        <div className="relative bg-brand/10 rounded-full w-12 h-12 flex items-center justify-center border border-brand/20">
          <Loader2 className="w-6 h-6 text-brand animate-spin" />
        </div>
      </div>
      <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold animate-pulse">Initializing Neural Core...</p>
    </div>
  </div>
);

import { HelmetProvider } from 'react-helmet-async';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <HelmetProvider>
        <AppProvider>
          <Router>
            <Suspense fallback={<LoadingFallback />}>
              <Routes>
                <Route path="/" element={<Layout><Home /></Layout>} />
                <Route path="/login" element={<AuthPage initialMode="login" />} />
                <Route path="/signup" element={<AuthPage initialMode="signup" />} />
                <Route path="/analysis" element={<Layout showNav={true} navStyle="minimal" showTicker={false}><Terminal /></Layout>} />
                <Route path="/pricing" element={<Layout><PricingPage /></Layout>} />
                <Route path="/docs" element={<Layout showTicker={false}><DocsPage /></Layout>} />
                <Route path="/legal/:type" element={<Layout><LegalPage /></Layout>} />
                <Route path="/profile" element={<Layout><ProfilePage /></Layout>} />
                <Route path="/what-is-diver-ai" element={<Layout><WhatIsDiverAI /></Layout>} />
                <Route path="/articles" element={<Layout><ArticlesHub /></Layout>} />
                <Route path="/articles/:slug" element={<Layout><ArticlePage /></Layout>} />
                <Route path="/updates" element={<Layout><UpdatesPage /></Layout>} />
                <Route path="/referral" element={<Layout><ReferralPage /></Layout>} />
              </Routes>
            </Suspense>
          </Router>
          <ReloadPrompt />
          <Analytics />
        </AppProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
}

export default App;
