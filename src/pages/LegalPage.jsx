import React from 'react';
import InfoPages from '../components/InfoPages';
import { useParams, useNavigate } from 'react-router-dom';

const LegalPage = () => {
    const { type } = useParams();
    const navigate = useNavigate();

    return (
        <InfoPages type={type} onClose={() => navigate('/')} />
    );
};

export default LegalPage;
