import React from 'react';
import { articles } from '../data/articles';
import { Link } from 'react-router-dom';
import { ArrowRight, BookOpen } from 'lucide-react';
import { Helmet } from 'react-helmet-async';

const ArticlesHub = () => {
    return (
        <div className="min-h-screen bg-black text-slate-200 font-sans selection:bg-brand/30 pt-24 pb-20">
            <Helmet>
                <title>Market Intelligence Hub | Diver AI Articles</title>
                <meta name="description" content="Deep dives into algorithmic trading, pattern recognition theory, and market psychology. Learn the science behind Diver AI." />
                <meta property="og:title" content="Diver AI Articles | Trading Intelligence" />
                <meta property="og:description" content="Read our latest insights on AI trading and market analysis techniques." />
            </Helmet>
            <div className="max-w-7xl mx-auto px-6">

                <div className="text-center mb-20 space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-brand/10 border border-brand/20 rounded-full text-brand text-xs font-semibold uppercase tracking-widest">
                        <BookOpen className="w-4 h-4" /> Market Intelligence
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
                        Trading <span className="text-brand">Insights</span>.
                    </h1>
                    <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
                        Deep dives into algorithmic trading, pattern recognition theory, and market psychology.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {articles.map((article) => (
                        <Link
                            to={`/articles/${article.id}`}
                            key={article.id}
                            className="group bg-black-ash border border-slate-800 rounded-2xl p-8 hover:border-brand/30 transition-all hover:-translate-y-1 shadow-sm"
                        >
                            <div className="mb-6 space-y-3">
                                <div className="text-xs font-bold text-brand uppercase tracking-widest">{article.date}</div>
                                <h3 className="text-xl font-bold text-white group-hover:text-brand transition-colors leading-tight">
                                    {article.title}
                                </h3>
                                <p className="text-slate-400 text-sm leading-relaxed line-clamp-3">
                                    {article.summary}
                                </p>
                            </div>
                            <div className="flex items-center gap-2 text-sm font-semibold text-white group-hover:gap-3 transition-all">
                                Read Article <ArrowRight className="w-4 h-4 text-brand" />
                            </div>
                        </Link>
                    ))}
                </div>

            </div>
        </div>
    );
};

export default ArticlesHub;
