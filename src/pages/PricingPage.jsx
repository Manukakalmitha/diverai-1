import React from 'react';
import Pricing from '../components/Pricing';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

const PricingPage = () => {
    const { user, profile, refreshProfile } = useAppContext();
    const navigate = useNavigate();

    return (
        <div className="min-h-screen">
            <Helmet>
                <title>Pricing & Plans | Diver AI - Unlock Institutional Alpha</title>
                <meta name="description" content="Choose the right plan for your trading style. From 3 free scans to unlimited institutional-grade analysis with Diver AI Pro." />
                <meta property="og:title" content="Diver AI Pricing | Upgrade to Pro" />
                <meta property="og:description" content="Unlock unlimited scans and priority analysis. See our competitive pricing plans." />
            </Helmet>
            <Pricing
                user={user}
                profile={profile}
                onClose={() => navigate(-1)}
                onUpgrade={refreshProfile}
            />
        </div>
    );
};

export default PricingPage;
