import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Loader2, Zap, Shield, X, ArrowRight, ChevronDown, ChevronUp, HelpCircle, CreditCard, Lock, Star } from 'lucide-react';
import { supabase } from '../lib/supabase';
import GooglePayButton from './GooglePayButton';

const FAQItem = ({ question, answer }) => {
    const [isOpen, setIsOpen] = useState(false);
    return (
        <div className="border-b border-slate-800 last:border-0">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full py-6 flex items-center justify-between text-left group"
            >
                <span className="text-lg font-medium text-slate-200 group-hover:text-emerald-400 transition-colors">{question}</span>
                {isOpen ? <ChevronUp className="w-5 h-5 text-slate-500" /> : <ChevronDown className="w-5 h-5 text-slate-500" />}
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                        className="overflow-hidden"
                    >
                        <p className="pb-6 text-slate-400 leading-relaxed">
                            {answer}
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

const Pricing = ({ user, profile, onClose, onUpgrade }) => {
    const [loading, setLoading] = useState(false);
    const [payMethod, setPayMethod] = useState(null); // 'gpay' or 'card'
    const [billingCycle, setBillingCycle] = useState('monthly'); // 'monthly' or 'annual'

    const PAYHERE_LINK = "https://payhere.lk/pay/o8f8b823";

    const handleUpgrade = async (method = 'card') => {
        if (!user) {
            alert("Please log in to upgrade.");
            return;
        }
        setLoading(true);
        setPayMethod(method);

        try {
            const redirectUrl = `${PAYHERE_LINK}?custom_1=${user.id}`;
            setTimeout(() => {
                window.open(redirectUrl, '_blank');
                setLoading(false);
                setPayMethod(null);
            }, 800);
        } catch (err) {
            console.error("Payment Redirect Error:", err);
            alert("Payment initialization failed. Please try again.");
            setLoading(false);
            setPayMethod(null);
        }
    };

    const features = [
        { name: "Neural scans per day", starter: "3", pro: "Unlimited" },
        { name: "Core Optical Engine", starter: true, pro: true },
        { name: "OCR Passes per Scan", starter: "4", pro: "8" },
        { name: "ROI Ticker Extraction", starter: false, pro: true },
        { name: "Standard Latency", starter: true, pro: true },
        { name: "Cloud OCR Priority", starter: false, pro: true },
        { name: "10-Year Macro Analysis", starter: false, pro: true },
        { name: "Pattern Recognition Alpha", starter: false, pro: true },
        { name: "Multi-Timeframe Confluence", starter: false, pro: true },
        { name: "PDF Intelligence Reports", starter: false, pro: true },
        { name: "Incognito Mode", starter: false, pro: true },
        { name: "Priority Support", starter: false, pro: true },
    ];

    const faqs = [
        {
            question: "How does the Neural analysis work?",
            answer: "Our core engine uses a proprietary Bayesian neural network that processes OCR data from your charts in real-time, identifying institutional-grade patterns and confluence zones with ultra-low latency."
        },
        {
            question: "Can I cancel my subscription any time?",
            answer: "Yes, you can cancel your Pro subscription at any time from your profile settings. You will maintain access to Pro features until the end of your current billing period."
        },
        {
            question: "What payment methods do you accept?",
            answer: "We support major credit cards and Google Pay through PayHere, ensuring your transactions are secure and encrypted."
        },
        {
            question: "Is there a free trial for the Pro plan?",
            answer: "We offer a 'Starter' plan with 3 free scans per day so you can test the core technology before upgrading to an unlimited Pro account."
        }
    ];

    return (
        <div className="w-full min-h-screen bg-[#0f172a] text-white selection:bg-emerald-500/30">
            {/* Background elements */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-500/5 blur-[120px] rounded-full" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full" />
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-6 py-20 lg:py-32">
                {/* Hero Section */}
                <div className="text-center mb-16 lg:mb-24">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold uppercase tracking-widest mb-6"
                    >
                        <Star className="w-3 h-3 fill-current" />
                        Join the Elite
                    </motion.div>
                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-4xl md:text-6xl font-black tracking-tight mb-6 bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent"
                    >
                        Precision Tools for <br className="hidden md:block" /> Institutional Alpha
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto font-medium"
                    >
                        Choose the plan that powers your trading desk with neural-grade intelligence and zero-limit analysis.
                    </motion.p>

                    {/* Billing Toggle */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 }}
                        className="mt-12 flex items-center justify-center gap-4"
                    >
                        <span className={`text-sm font-bold tracking-wide transition-colors ${billingCycle === 'monthly' ? 'text-white' : 'text-slate-500'}`}>Monthly</span>
                        <button
                            onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'annual' : 'monthly')}
                            className="relative w-14 h-7 bg-slate-800 rounded-full p-1 transition-colors hover:bg-slate-700"
                        >
                            <motion.div
                                animate={{ x: billingCycle === 'monthly' ? 0 : 28 }}
                                className="w-5 h-5 bg-emerald-500 rounded-full shadow-lg shadow-emerald-500/20"
                            />
                        </button>
                        <div className="flex items-center gap-2">
                            <span className={`text-sm font-bold tracking-wide transition-colors ${billingCycle === 'annual' ? 'text-white' : 'text-slate-500'}`}>Annual</span>
                            <span className="bg-emerald-500/10 text-emerald-400 text-[10px] font-black px-2 py-0.5 rounded border border-emerald-500/20 uppercase tracking-wider">
                                2 Months Free
                            </span>
                        </div>
                    </motion.div>
                </div>

                {/* Pricing Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto mb-32">
                    {/* Starter Plan */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                        className="group relative bg-slate-900/50 border border-slate-800 rounded-[32px] p-10 flex flex-col hover:border-slate-700 transition-all duration-500"
                    >
                        <div className="mb-8">
                            <h3 className="text-xl font-bold text-slate-300 mb-2">Starter Core</h3>
                            <div className="flex items-baseline gap-1">
                                <div className="text-5xl font-black text-white">$0</div>
                                <div className="text-slate-500 font-bold text-lg uppercase tracking-widest">/mo</div>
                            </div>
                            <p className="text-slate-500 text-sm mt-4 font-medium leading-relaxed">Perfect for exploring the bayesian core and basic neural scans.</p>
                        </div>

                        <ul className="space-y-4 mb-10 flex-1">
                            <li className="flex gap-4 text-slate-400 text-sm font-medium">
                                <div className="bg-slate-800 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-500" /></div>
                                3 Neural scans per day
                            </li>
                            <li className="flex gap-4 text-slate-400 text-sm font-medium">
                                <div className="bg-slate-800 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-500" /></div>
                                Standard Optical Engine
                            </li>
                            <li className="flex gap-4 text-slate-400 text-sm font-medium opacity-50">
                                <X className="w-5 h-5 text-slate-600" /> Institutional OCR
                            </li>
                        </ul>

                        <button
                            onClick={onClose}
                            className="w-full py-5 rounded-2xl bg-white/5 border border-white/10 text-slate-300 font-bold hover:bg-white/10 hover:text-white transition-all text-xs uppercase tracking-[0.2em]"
                        >
                            Active Architecture
                        </button>
                    </motion.div>

                    {/* Pro Plan */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 }}
                        className="group relative bg-[#0f172a] border border-emerald-500/30 rounded-[32px] p-10 flex flex-col shadow-2xl shadow-emerald-500/10 overflow-hidden"
                    >
                        {/* Premium Gradient Overlay */}
                        <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-emerald-500 to-transparent opacity-50" />
                        <div className="absolute -top-40 -right-40 w-80 h-80 bg-emerald-500/10 blur-[100px] rounded-full pointer-events-none" />

                        <div className="absolute top-6 right-6 px-3 py-1 rounded-full bg-emerald-500 text-slate-950 text-[10px] font-black uppercase tracking-widest">
                            Most Advanced
                        </div>

                        <div className="mb-8 relative z-10">
                            <h3 className="text-xl font-bold text-emerald-400 mb-2 flex items-center gap-2">
                                <Zap className="w-6 h-6 fill-current" /> Pro Quant
                            </h3>
                            <div className="flex items-baseline gap-1">
                                <div className="text-5xl font-black text-white tracking-tighter">
                                    {billingCycle === 'monthly' ? '$29' : '$290'}
                                </div>
                                <div className="text-slate-500 font-bold text-lg uppercase tracking-widest">
                                    {billingCycle === 'monthly' ? '/mo' : '/yr'}
                                </div>
                            </div>
                            <p className="text-slate-400 text-sm mt-4 font-medium leading-relaxed">Unlocked neural bandwidth for desk traders and professional analysts.</p>
                        </div>

                        <ul className="space-y-4 mb-10 flex-1 relative z-10">
                            <li className="flex gap-4 text-white text-sm font-bold">
                                <div className="bg-emerald-500/20 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-400" /></div>
                                Unlimited Neural Analysis
                            </li>
                            <li className="flex gap-4 text-slate-200 text-sm font-medium">
                                <div className="bg-emerald-500/20 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-400" /></div>
                                Institutional OCR Priority
                            </li>
                            <li className="flex gap-4 text-slate-200 text-sm font-medium">
                                <div className="bg-emerald-500/20 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-400" /></div>
                                Pattern Recognition Alpha
                            </li>
                            <li className="flex gap-4 text-slate-200 text-sm font-medium">
                                <div className="bg-emerald-500/20 rounded-full p-1"><Check className="w-3.5 h-3.5 text-emerald-400" /></div>
                                PDF Intelligence Reports
                            </li>
                        </ul>

                        <div className="space-y-4 relative z-10">
                            <GooglePayButton
                                onClick={() => handleUpgrade('gpay')}
                                disabled={loading || (profile?.subscription_tier === 'pro')}
                            />

                            <button
                                onClick={() => handleUpgrade('card')}
                                disabled={loading || (profile?.subscription_tier === 'pro')}
                                className="relative w-full py-5 rounded-2xl bg-emerald-500 text-slate-950 font-black hover:bg-emerald-400 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed group/btn active:scale-[0.98] shadow-xl shadow-emerald-500/20 overflow-hidden"
                            >
                                <div className="absolute inset-0 bg-white/20 translate-x-[-100%] group-hover/btn:translate-x-[100%] transition-transform duration-700 pointer-events-none" />
                                {loading && payMethod === 'card' ? (
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                ) : (
                                    <>
                                        <span className="text-xs uppercase tracking-[0.2em]">
                                            {profile?.subscription_tier === 'pro' ? 'Active Subscription' : 'Upgrade with Card'}
                                        </span>
                                        <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-1 transition-transform" />
                                    </>
                                )}
                            </button>
                        </div>

                        <p className="flex items-center justify-center gap-2 text-[10px] text-slate-500 font-bold mt-6 uppercase tracking-widest">
                            <Lock className="w-3 h-3" /> Secure SSL • Powered by PayHere
                        </p>
                    </motion.div>
                </div>

                {/* Feature Comparison */}
                <div className="mb-32">
                    <div className="text-center mb-12">
                        <h2 className="text-2xl font-bold mb-4">Compare Features</h2>
                        <div className="w-12 h-1 bg-emerald-500 mx-auto rounded-full" />
                    </div>
                    <div className="max-w-4xl mx-auto overflow-hidden border border-slate-800 rounded-3xl bg-slate-900/30 backdrop-blur-sm">
                        <table className="w-full border-collapse">
                            <thead>
                                <tr className="border-b border-slate-800">
                                    <th className="p-6 text-left text-sm font-bold text-slate-400 uppercase tracking-widest">Architecture</th>
                                    <th className="p-6 text-center text-sm font-bold text-slate-400 uppercase tracking-widest">Starter</th>
                                    <th className="p-6 text-center text-sm font-bold text-emerald-400 uppercase tracking-widest">Pro</th>
                                </tr>
                            </thead>
                            <tbody>
                                {features.map((feature, idx) => (
                                    <tr key={idx} className="border-b border-slate-800/50 hover:bg-white/[0.02] transition-colors">
                                        <td className="p-6 text-sm font-medium text-slate-300">{feature.name}</td>
                                        <td className="p-6 text-center">
                                            {typeof feature.starter === 'string' ? (
                                                <span className="text-slate-400 text-sm font-bold">{feature.starter}</span>
                                            ) : (
                                                feature.starter ? <Check className="w-5 h-5 text-slate-600 mx-auto" /> : <X className="w-5 h-5 text-slate-800 mx-auto" />
                                            )}
                                        </td>
                                        <td className="p-6 text-center">
                                            {typeof feature.pro === 'string' ? (
                                                <span className="text-emerald-400 text-sm font-bold">{feature.pro}</span>
                                            ) : (
                                                <Check className="w-5 h-5 text-emerald-500 mx-auto" />
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Trust Section */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-32">
                    <div className="p-8 rounded-3xl bg-slate-900/50 border border-slate-800 flex flex-col items-center text-center group hover:bg-slate-800/50 transition-colors">
                        <div className="w-12 h-12 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 border border-emerald-500/20 group-hover:scale-110 transition-transform">
                            <Shield className="w-6 h-6 text-emerald-400" />
                        </div>
                        <h4 className="text-lg font-bold mb-2">Secure Infrastructure</h4>
                        <p className="text-sm text-slate-400 leading-relaxed">Enterprise-grade encryption protecting your data and payment information.</p>
                    </div>
                    <div className="p-8 rounded-3xl bg-slate-900/50 border border-slate-800 flex flex-col items-center text-center group hover:bg-slate-800/50 transition-colors">
                        <div className="w-12 h-12 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 border border-emerald-500/20 group-hover:scale-110 transition-transform">
                            <CreditCard className="w-6 h-6 text-emerald-400" />
                        </div>
                        <h4 className="text-lg font-bold mb-2">Flexible Billing</h4>
                        <p className="text-sm text-slate-400 leading-relaxed">Upgrade or cancel anytime through our intuitive dashboard. No hidden fees.</p>
                    </div>
                    <div className="p-8 rounded-3xl bg-slate-900/50 border border-slate-800 flex flex-col items-center text-center group hover:bg-slate-800/50 transition-colors">
                        <div className="w-12 h-12 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 border border-emerald-500/20 group-hover:scale-110 transition-transform">
                            <HelpCircle className="w-6 h-6 text-emerald-400" />
                        </div>
                        <h4 className="text-lg font-bold mb-2">Expert Support</h4>
                        <p className="text-sm text-slate-400 leading-relaxed">Direct access to our engineering team for technical queries and integration help.</p>
                    </div>
                </div>

                {/* FAQ Section */}
                <div className="max-w-3xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-black mb-4">Frequently Asked Questions</h2>
                        <p className="text-slate-500 font-medium tracking-wide">Everything you need to know about DiverAI Pro.</p>
                    </div>
                    <div className="space-y-2">
                        {faqs.map((faq, idx) => (
                            <FAQItem key={idx} question={faq.question} answer={faq.answer} />
                        ))}
                    </div>
                </div>

                {/* Footer CTA */}
                <div className="mt-32 p-12 rounded-[40px] bg-gradient-to-br from-emerald-500 to-blue-600 relative overflow-hidden text-center group">
                    <div className="absolute inset-0 bg-black/10 backdrop-blur-[2px]" />
                    <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-white/10 to-transparent pointer-events-none" />

                    <div className="relative z-10">
                        <h2 className="text-3xl md:text-4xl font-black text-white mb-6">Ready to Scale Your Analysis?</h2>
                        <p className="text-white/80 text-lg mb-10 max-w-xl mx-auto font-medium">Join over 1,000+ traders using DiverAI to hunt institutional alpha daily.</p>
                        <button
                            onClick={() => handleUpgrade('card')}
                            className="bg-white text-slate-950 px-10 py-4 rounded-2xl font-black text-sm uppercase tracking-[0.2em] hover:bg-slate-100 transition-all hover:shadow-2xl hover:shadow-white/20 active:scale-95"
                        >
                            Get Started Now
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Pricing;
