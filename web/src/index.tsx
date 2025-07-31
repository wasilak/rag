import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement
);

root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

// Log app startup
console.log('RAG Web Interface started');

// Report web vitals if needed
import('./reportWebVitals').then(({ default: reportWebVitals }) => {
    reportWebVitals();
}).catch(() => {
    // reportWebVitals is optional
});
