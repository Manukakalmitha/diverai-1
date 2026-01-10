import React from 'react';
import ReactDOM from 'react-dom/client';
import Sidebar from './Sidebar';
import '../src/index.css'; // Reuse main app styles

import { AppProvider } from '../src/context/AppContext';

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <AppProvider>
            <Sidebar />
        </AppProvider>
    </React.StrictMode>
);
