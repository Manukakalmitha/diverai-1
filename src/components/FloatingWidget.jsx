import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Clipboard, Layout, X, Zap, Smartphone, Share2, Monitor } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const FloatingWidget = ({ onPaste, onCapture, onCompanion }) => {
    const [isOpen, setIsOpen] = useState(false);
    const navigate = useNavigate();

    const menuItems = [
        {
            icon: <Clipboard className="w-4 h-4" />,
            label: 'Paste Chart',
            action: () => {
                onPaste?.();
                setIsOpen(false);
            },
            color: 'text-emerald-400'
        },
        {
            icon: <Monitor className="w-4 h-4" />,
            label: 'Companion Mode',
            action: () => {
                onCompanion?.();
                setIsOpen(false);
            },
            color: 'text-amber-400'
        },
        {
            icon: <Smartphone className="w-4 h-4" />,
            label: 'Mobile Guide',
            action: () => {
                navigate('/docs#mobile');
                setIsOpen(false);
            },
            color: 'text-blue-400'
        },
        {
            icon: <Layout className="w-4 h-4" />,
            label: 'Terminal',
            action: () => {
                navigate('/analysis');
                setIsOpen(false);
            },
            color: 'text-purple-400'
        }
    ];

    return (
        <div className="fixed bottom-6 right-6 z-[1000] flex flex-col items-end gap-3 pointer-events-none">
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.9 }}
                        className="bg-slate-900/90 backdrop-blur-xl border border-slate-800 rounded-3xl p-3 shadow-2xl pointer-events-auto min-w-[180px]"
                    >
                        <div className="flex flex-col gap-1">
                            {menuItems.map((item, i) => (
                                <button
                                    key={i}
                                    onClick={item.action}
                                    className="flex items-center gap-3 w-full p-3 hover:bg-white/5 rounded-2xl transition-all group"
                                >
                                    <div className={`p-2 rounded-xl bg-slate-950/50 ${item.color} border border-white/5 ring-1 ring-white/5 group-hover:ring-emerald-500/50 transition-all`}>
                                        {item.icon}
                                    </div>
                                    <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest group-hover:text-white transition-colors">
                                        {item.label}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <motion.button
                layout
                onClick={() => setIsOpen(!isOpen)}
                className="pointer-events-auto relative w-16 h-16 bg-black-ash rounded-[24px] border border-slate-700 shadow-[0_8px_32px_rgba(0,0,0,0.4)] flex items-center justify-center group overflow-hidden"
                style={{ backdropFilter: 'blur(12px)' }}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
            >
                {/* Background Pulse Effect */}
                <div className="absolute inset-0 bg-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity animate-pulse" />

                <div className="relative z-10">
                    {isOpen ? (
                        <X className="w-6 h-6 text-slate-400" />
                    ) : (
                        <img
                            src="/pulse-icon.png"
                            alt="DiverAI"
                            className="w-10 h-10 object-contain rounded-lg animate-[heartbeat_3s_infinite_ease-in-out]"
                        />
                    )}
                </div>

                {/* Status Indicator */}
                {!isOpen && (
                    <div className="absolute top-3 right-3 w-2 h-2 bg-emerald-500 rounded-full border border-slate-900 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                )}
            </motion.button>

            {/* Hint Tag */}
            <AnimatePresence>
                {!isOpen && (
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: 2 }}
                        className="bg-slate-900/80 backdrop-blur-md border border-slate-800 px-3 py-1.5 rounded-full text-[9px] font-black text-slate-500 uppercase tracking-widest shadow-xl"
                    >
                        Mobile Quick Action
                    </motion.div>
                )}
            </AnimatePresence>

            <style dangerouslySetInnerHTML={{
                __html: `
                @keyframes heartbeat {
                    0% { transform: scale(1); }
                    15% { transform: scale(1.1); }
                    30% { transform: scale(1); }
                    45% { transform: scale(1.15); }
                    70% { transform: scale(1); }
                }
            ` }} />
        </div>
    );
};

export default FloatingWidget;
