import React from 'react';
import { useRegisterSW } from 'virtual:pwa-register/react';
import { RefreshCw, X, Zap } from 'lucide-react';

function ReloadPrompt() {
    const {
        offlineReady: [offlineReady, setOfflineReady],
        needRefresh: [needRefresh, setNeedRefresh],
        updateServiceWorker,
    } = useRegisterSW({
        onRegistered(r) {
            console.log('SW Registered');
        },
        onRegisterError(error) {
            console.log('SW registration error', error);
        },
    });

    const close = () => {
        setOfflineReady(false);
        setNeedRefresh(false);
    };

    if (!offlineReady && !needRefresh) return null;

    return (
        <div className="fixed bottom-20 left-4 right-4 md:bottom-6 md:right-6 md:left-auto z-[100] animate-in slide-in-from-bottom-5 duration-500">
            <div className="bg-black-ash/90 backdrop-blur-xl border border-emerald-500/20 rounded-2xl p-4 shadow-2xl flex flex-col gap-3 min-w-[280px] max-w-sm">
                <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div className="bg-emerald-500/10 p-2 rounded-xl">
                            <Zap className="w-5 h-5 text-emerald-500" />
                        </div>
                        <div>
                            <h4 className="text-white text-xs font-black uppercase tracking-widest">
                                {offlineReady ? 'Terminal Ready' : 'Update Available'}
                            </h4>
                            <p className="text-slate-400 text-[10px] font-bold mt-0.5">
                                {offlineReady ? 'Diver AI is now available offline.' : 'New neural weights are ready for uplink.'}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={close}
                        className="text-slate-500 hover:text-white transition-colors"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="flex gap-2 mt-1">
                    {needRefresh && (
                        <button
                            onClick={() => updateServiceWorker(true)}
                            className="flex-1 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/10"
                        >
                            <RefreshCw className="w-3.5 h-3.5 animate-spin-slow" />
                            Sync Neural Core
                        </button>
                    )}
                    <button
                        onClick={close}
                        className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-white text-[10px] font-black uppercase tracking-widest rounded-xl transition-colors"
                    >
                        Dismiss
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ReloadPrompt;
