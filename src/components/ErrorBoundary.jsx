import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 text-center">
                    <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-[32px] p-8 shadow-2xl">
                        <div className="w-16 h-16 bg-rose-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-rose-500"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>
                        </div>
                        <h1 className="text-2xl font-black text-white mb-4">Neural Interface Error</h1>
                        <p className="text-slate-400 text-sm font-bold mb-8 leading-relaxed">
                            The application encountered an unexpected runtime exception. This is often caused by browser interference or extension conflicts.
                        </p>
                        <button
                            onClick={() => window.location.reload()}
                            className="w-full py-4 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-xl transition-all uppercase tracking-widest text-xs shadow-xl shadow-emerald-500/10"
                        >
                            Reboot Terminal
                        </button>
                        <p className="mt-6 text-[10px] font-mono text-slate-600 uppercase tracking-tighter">
                            Error: {this.state.error?.message || "Unknown Exception"}
                        </p>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
