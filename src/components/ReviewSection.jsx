import React, { useState, useEffect } from 'react';
import { Star, MessageSquare, User, CheckCircle, Quote, Send, Loader2, Sparkles, Filter } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { supabase } from '../lib/supabase';
import { useAppContext } from '../context/AppContext';

export default function ReviewSection() {
    const { user } = useAppContext();
    const [reviews, setReviews] = useState([
        { id: 1, user: "Trader_X", text: "The Bayesian engine is surprisingly accurate on BTC 15m timeframes. TP1 hits almost every time.", rating: 5, date: "2d ago", verified: true },
        { id: 2, user: "QuantAlpha", text: "Finally an OCR that actually reads TradingView charts correctly. Saves me 10 mins of manual entry.", rating: 4, date: "1w ago", verified: true },
        { id: 3, user: "Nikki_T", text: "Clean UI, the trade blueprint is a lifesaver for risk management. Highly recommend for scalpers.", rating: 5, date: "3d ago", verified: true },
        { id: 4, user: "InstitutionalProxy", text: "The precision rank is a great second opinion for my manual technical analysis. Pro plan is worth it.", rating: 5, date: "5d ago", verified: true },
        { id: 5, user: "MarketMaven", text: "Solid pattern recognition. Just wish it had more forex pairs, but for crypto it is 10/10.", rating: 4, date: "2w ago", verified: true }
    ]);

    const [isWriting, setIsWriting] = useState(false);
    const [newReview, setNewReview] = useState({ text: '', rating: 5 });
    const [loading, setLoading] = useState(false);

    const handleSubmit = async () => {
        if (!newReview.text.trim()) return;
        setLoading(true);

        // In a real app, we'd save to Supabase 'reviews' table
        // For now, we simulate success
        await new Promise(r => setTimeout(r, 1000));

        const review = {
            id: Date.now(),
            user: user?.email?.split('@')[0] || "Anonymous",
            text: newReview.text,
            rating: newReview.rating,
            date: "Just now",
            verified: !!user
        };

        setReviews([review, ...reviews]);
        setNewReview({ text: '', rating: 5 });
        setIsWriting(false);
        setLoading(false);
        alert("Review submitted! Thank you for the feedback.");
    };

    return (
        <div className="py-24 max-w-7xl mx-auto px-6" id="reviews">
            <div className="flex flex-col md:flex-row items-end justify-between mb-12 gap-6">
                <div className="space-y-4 text-center md:text-left">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 border border-slate-800 rounded-full">
                        <Star className="w-4 h-4 text-emerald-500 fill-emerald-500" />
                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Community Feedback</span>
                    </div>
                    <h2 className="text-4xl md:text-6xl font-black text-white leading-none tracking-tighter">
                        Verified by <br /> <span className="bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent italic">Elite Traders.</span>
                    </h2>
                </div>

                <button
                    onClick={() => setIsWriting(!isWriting)}
                    className="px-8 py-4 bg-emerald-600 text-white font-black rounded-xl text-[10px] uppercase tracking-widest hover:bg-emerald-500 transition-all shadow-xl flex items-center gap-3"
                >
                    {isWriting ? "Close Terminal" : <><MessageSquare className="w-4 h-4" /> Share My Experience</>}
                </button>
            </div>

            <AnimatePresence>
                {isWriting && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden mb-16"
                    >
                        <div className="p-8 md:p-12 bg-slate-900/50 border border-slate-800 rounded-xl shadow-2xl relative">
                            <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none -rotate-12">
                                <Sparkles className="w-48 h-48 text-emerald-400" />
                            </div>

                            <div className="max-w-2xl mx-auto space-y-8 relative z-10">
                                <div className="text-center">
                                    <h3 className="text-3xl font-black text-white mb-2 uppercase tracking-tight">Write a Review</h3>
                                    <p className="text-slate-500 font-bold text-sm">Help the community optimize their neural strategies.</p>
                                </div>

                                <div className="flex justify-center gap-3">
                                    {[1, 2, 3, 4, 5].map(star => (
                                        <button
                                            key={star}
                                            onClick={() => setNewReview({ ...newReview, rating: star })}
                                            className="transition-transform active:scale-90"
                                        >
                                            <Star className={`w-10 h-10 ${star <= newReview.rating ? 'text-emerald-500 fill-emerald-500' : 'text-slate-800'}`} />
                                        </button>
                                    ))}
                                </div>

                                <div className="relative">
                                    <textarea
                                        value={newReview.text}
                                        onChange={(e) => setNewReview({ ...newReview, text: e.target.value })}
                                        placeholder="Describe your terminal experience... (OCR accuracy, pattern matching, blueprint utility)"
                                        className="w-full h-40 bg-slate-950 border border-slate-800 rounded-xl p-8 text-white placeholder:text-slate-700 focus:border-emerald-500/50 focus:ring-4 focus:ring-emerald-500/5 outline-none transition-all resize-none"
                                    />
                                    {!user && (
                                        <div className="absolute top-4 right-8 bg-amber-500/10 text-amber-500 px-3 py-1 rounded-full text-[8px] font-black uppercase tracking-widest border border-amber-500/20">
                                            Posting as Guest
                                        </div>
                                    )}
                                </div>

                                <button
                                    onClick={handleSubmit}
                                    disabled={loading || !newReview.text.trim()}
                                    className="w-full py-5 bg-emerald-500 text-slate-950 font-black rounded-xl uppercase tracking-[0.2em] text-xs hover:bg-emerald-400 transition-all flex items-center justify-center gap-3 disabled:opacity-50 shadow-xl shadow-emerald-500/20"
                                >
                                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <><Send className="w-5 h-5" /> Deploy Feedback</>}
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <div className="columns-1 md:columns-2 lg:columns-3 gap-6 space-y-6">
                {reviews.map((rev) => (
                    <motion.div
                        layout
                        key={rev.id}
                        className="break-inside-avoid bg-slate-900 border border-slate-800 rounded-xl p-8 hover:border-slate-700 transition-all shadow-xl group cursor-default"
                    >
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex -space-x-1">
                                {[...Array(5)].map((_, i) => (
                                    <Star key={i} className={`w-4 h-4 ${i < rev.rating ? 'text-emerald-500 fill-emerald-500' : 'text-slate-800'}`} />
                                ))}
                            </div>
                            <span className="text-[10px] font-mono text-slate-600">{rev.date}</span>
                        </div>

                        <div className="relative mb-6">
                            <Quote className="absolute -top-4 -left-4 w-8 h-8 text-white/5 group-hover:text-emerald-500/10 transition-colors" />
                            <p className="text-slate-300 font-bold leading-relaxed relative z-10 italic">
                                "{rev.text}"
                            </p>
                        </div>

                        <div className="flex items-center justify-between pt-6 border-t border-slate-800/50">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-slate-800 rounded-lg flex items-center justify-center text-slate-500 font-black text-xs uppercase">
                                    {rev.user.slice(0, 2)}
                                </div>
                                <div>
                                    <h4 className="text-white font-black text-xs uppercase tracking-tight">{rev.user}</h4>
                                    <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Precision Tier</p>
                                </div>
                            </div>
                            {rev.verified && (
                                <div className="flex items-center gap-1.5 text-[8px] font-black text-emerald-400 uppercase tracking-widest bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                                    <CheckCircle className="w-3 h-3" /> Verified
                                </div>
                            )}
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
