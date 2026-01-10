import React from 'react';
import Documentation from '../components/Documentation';
import { useNavigate } from 'react-router-dom';

const DocsPage = () => {
    const navigate = useNavigate();

    return (
        <div className="py-20 md:py-32">
            <Documentation onClose={() => navigate(-1)} />
        </div>
    );
};

export default DocsPage;
