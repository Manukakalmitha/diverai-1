import { useEffect } from 'react';
import LandingPage from '../components/LandingPage';
import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import { Helmet } from 'react-helmet-async';

const Home = () => {
    const navigate = useNavigate();
    const { user, loading } = useAppContext();

    // Redirect logged-in users directly to the analysis page
    useEffect(() => {
        if (!loading && user) {
            navigate('/analysis', { replace: true });
        }
    }, [user, loading, navigate]);

    // Show nothing while checking auth status to prevent flash
    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <>
            <Helmet>
                <title>Diver AI | Institutional-Grade AI Stock Analysis & Crypto Signals</title>
                <meta name="description" content="Master the markets with Diver AI. Institutional-grade optical pattern recognition and probabilistic forecasting for stocks, crypto, and forex." />
                <meta property="og:title" content="Diver AI | AI-Powered Market Intelligence" />
                <meta property="og:description" content="Get institutional-grade trading signals using AI pattern recognition. Try free today." />
            </Helmet>
            <LandingPage
                onStart={() => navigate('/analysis')}
                onOpenInfo={(type) => navigate(`/legal/${type}`)}
                onOpenDocs={() => navigate('/docs')}
                onOpenPricing={() => navigate('/pricing')}
            />
        </>
    );
};

export default Home;
