import React from 'react';
import { X, Shield, Scale, Cpu, Globe, Activity, CheckCircle2, AlertCircle, Users, BookOpen, AlertTriangle } from 'lucide-react';

const InfoPages = ({ type, onClose }) => {
    const content = {
        privacy: {
            title: "Privacy Policy",
            subtitle: "Trust & Safety",
            icon: Shield,
            color: "text-blue-500",
            sections: [
                {
                    h: "Data Collection",
                    p: "Diver AI collects minimal data. Analysis images are processed in volatile memory and are not permanently stored unless you choose to save them to your terminal history."
                },
                {
                    h: "Usage Analytics",
                    p: "We use anonymized usage data to improve our analysis models. This includes prediction outcomes and indicator confluences, but never personal identification."
                },
                {
                    h: "Security",
                    p: "We implement institutional-grade encryption (AES-256) for all user-stored history and use military-grade CSP headers to prevent XSS and data injection."
                }
            ]
        },
        terms: {
            title: "Terms of Service",
            subtitle: "Usage Framework",
            icon: Scale,
            color: "text-blue-500",
            sections: [
                {
                    h: "Intellectual Property",
                    p: "The Pattern Engine and its proprietary algorithms are the direct intellectual property of Diver AI."
                },
                {
                    h: "Financial Disclaimer",
                    p: "Diver AI is an analysis tool, not a financial advisor. All predictions are probabilistic and involve significant risk. Past performance does not guarantee future results."
                },
                {
                    h: "Account Usage",
                    p: "Users are responsible for maintaining the security of their credentials. One account per user is permitted for free tier usage."
                }
            ]
        },
        api: {
            title: "API Documentation",
            subtitle: "Institutional Access",
            icon: Globe,
            color: "text-purple-500",
            sections: [
                {
                    h: "Status: Restricted",
                    p: "Institutional API access is currently in closed beta. We are optimizing our WebSocket feeds for ultra-low latency execution."
                },
                {
                    h: "Endpoint: /v1/predict/ocr",
                    p: "Available for Pro Quant users. Allows programmatic submission of chart screenshots for structural analysis."
                },
                {
                    h: "Rate Limits",
                    p: "Standard rate limits apply: 100 requests per minute for Pro users. Contact support for higher throughput requirements."
                }
            ]
        },
        status: {
            title: "System Status",
            subtitle: "Real-time Monitoring",
            icon: Activity,
            color: "text-blue-500",
            sections: [
                {
                    h: "Operational Services",
                    p: "All core engines are currently operational with 99.9% uptime over the last 30 days."
                }
            ]
        },
        about: {
            title: "About Us",
            subtitle: "Team & Mission",
            icon: Users,
            color: "text-blue-500",
            sections: [
                {
                    h: "Our Mission",
                    p: "Diver AI was founded on the belief that data-driven analysis should be accessible to everyone. We exist to provide honest, transparent market insights to independent traders."
                },
                {
                    h: "The Team",
                    p: "Our diverse team consists of lead researchers, quantitative analysts, and software engineers dedicated to building the most accurate pattern recognition engine in the world."
                },
                {
                    h: "Contact",
                    p: "For research inquiries, partnership opportunities, or support, please reach out to our team directly at support.diverai@flisoft.agency. We typically respond within 24 hours."
                }
            ]
        },
        research: {
            title: "Research",
            subtitle: "Methodology",
            icon: BookOpen,
            color: "text-indigo-500",
            sections: [
                {
                    h: "Abstract",
                    p: "Our proprietary Optical Pattern Recognition (OPR) engine utilizes a hybrid computer vision architecture to identify geometric chart patterns with high statistical probability."
                },
                {
                    h: "Data Sources",
                    p: "We train our models on over 10 years of historical tick-data across Forex, Crypto, and Equities markets, totaling over 1 million analyzed chart patterns."
                },
                {
                    h: "Whitepaper",
                    p: "Our technical whitepaper detailing our pattern recognition process is currently under peer review and will be published here shortly."
                }
            ]
        },
        risk: {
            title: "Risk Disclosure",
            subtitle: "Important Information",
            icon: AlertTriangle,
            color: "text-rose-500",
            sections: [
                {
                    h: "General Risk Warning",
                    p: "Trading in financial markets involves a high degree of risk and exists the possibility of losing some or all of your initial investment. You should not invest money that you cannot afford to lose."
                },
                {
                    h: "No Financial Advice",
                    p: "The information and signals provided by Diver AI are for educational and informational purposes only and do not constitute financial advice, investment recommendations, or an offer to buy or sell any financial instruments."
                },
                {
                    h: "Accuracy Disclaimer",
                    p: "While we strive for high accuracy, our algorithms are based on probabilities derived from historical data. Past performance is not indicative of future results. Market conditions can change rapidly."
                },
                {
                    h: "Non-Custodial",
                    p: "Diver AI is a technical analysis tool only. We do not facilitate the buying or selling of assets, do not hold user funds, and do not offer digital wallet services."
                }
            ]
        },
        cookies: {
            title: "Cookie Policy",
            subtitle: "Data Persistence",
            icon: CheckCircle2,
            color: "text-blue-500",
            sections: [
                {
                    h: "Essential Cookies",
                    p: "We use essential cookies to maintain your session security and authentication state. These are required for the application to function."
                },
                {
                    h: "Analytics",
                    p: "We use anonymous analytical cookies to measure system performance and optimization metrics. No personally identifiable trading data is tracked."
                },
                {
                    h: "Preferences",
                    p: "Local storage is used to save your chart settings and theme preferences. You may clear this at any time via your browser settings."
                }
            ]
        }
    };

    const active = content[type] || content.privacy;

    return (
        <div className="min-h-screen bg-transparent animate-in fade-in duration-300">
            <div className="max-w-3xl mx-auto px-6 py-12">
                <div className="flex items-center justify-between mb-12 border-b border-slate-800 pb-12">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <active.icon className={`w-8 h-8 ${active.color}`} />
                            <h1 className="text-3xl md:text-5xl font-bold text-white">{active.title}</h1>
                        </div>
                        <p className="text-slate-500 text-xs uppercase tracking-widest font-semibold">{active.subtitle}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-4 bg-slate-900 rounded-xl hover:bg-rose-500/10 hover:text-rose-500 transition-all text-slate-500"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="space-y-12">
                    {active.sections.map((s, i) => (
                        <section key={i} className="group">
                            <h3 className="text-xl font-bold text-white mb-4 uppercase tracking-tight flex items-center gap-3">
                                <span className={`w-1 h-6 rounded-full ${active.color.replace('text-', 'bg-')} opacity-30 group-hover:opacity-100 transition-opacity`}></span>
                                {s.h}
                            </h3>
                            <p className="text-slate-400 leading-relaxed text-lg font-medium">
                                {s.p}
                            </p>
                        </section>
                    ))}

                    {type === 'status' && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-8">
                            {[
                                { n: "Analysis Engine", s: "Operational", c: "text-blue-500" },
                                { n: "Pattern Recognition", s: "Operational", c: "text-blue-500" },
                                { n: "Market Data Feeds", s: "Operational", c: "text-blue-500" },
                                { n: "Historical Database", s: "Operational", c: "text-blue-500" }
                            ].map((svc, i) => (
                                <div key={i} className="p-6 bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-between">
                                    <span className="text-sm font-semibold text-slate-300">{svc.n}</span>
                                    <div className="flex items-center gap-2">
                                        <div className={`w-1.5 h-1.5 rounded-full ${svc.c.replace('text-', 'bg-')} animate-pulse`} />
                                        <span className={`text-[10px] font-bold uppercase tracking-widest ${svc.c}`}>{svc.s}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <footer className="mt-20 pt-12 border-t border-slate-800 text-center">
                    <p className="text-slate-500 text-sm font-medium">
                        © 2026 Diver AI by <a href="https://flisoft.agency" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-500 transition-colors">Fli SOFT</a>. All rights reserved.
                    </p>
                </footer>
            </div>
        </div>
    );
};

export default InfoPages;
