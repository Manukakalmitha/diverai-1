import React from 'react';
import Pricing from '../components/Pricing';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';

const PricingPage = () => {
    const { user, profile, refreshProfile } = useAppContext();
    const navigate = useNavigate();

    return (
        <div className="py-20 md:py-32">
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
