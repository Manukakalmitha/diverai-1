import React from 'react';
import InfoPages from '../components/InfoPages';
import { useParams, useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

const LegalPage = () => {
    const { type } = useParams();
    const navigate = useNavigate();

    const pageTitle = type ? `${type.charAt(0).toUpperCase() + type.slice(1)} | Diver AI Legal` : 'Legal | Diver AI';

    return (
        <>
            <Helmet>
                <title>{pageTitle}</title>
                <meta name="description" content={`Diver AI ${type || 'Legal'} documentation. Rules, risks, and privacy information.`} />
            </Helmet>
            <InfoPages type={type} onClose={() => navigate('/')} />
        </>
    );
};

export default LegalPage;
