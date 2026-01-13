import React from 'react';
import Documentation from '../components/Documentation';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

const DocsPage = () => {
    const navigate = useNavigate();

    return (
        <div className="py-20 md:py-32">
            <Helmet>
                <title>Documentation | Diver AI - Technical Reference & Guides</title>
                <meta name="description" content="Learn how to use Diver AI's optical pattern recognition and neural engine. Comprehensive guides and technical documentation." />
            </Helmet>
            <Documentation onClose={() => navigate(-1)} />
        </div>
    );
};

export default DocsPage;
