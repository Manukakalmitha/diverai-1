import React from 'react';
import { articles } from '../data/articles';
import { Link } from 'react-router-dom';
import { ArrowRight, BookOpen } from 'lucide-react';

const ArticlesHub = () => {
    return (
        <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans selection:bg-blue-500/30 pt-24 pb-20">
            <div className="max-w-7xl mx-auto px-6">

                <div className="text-center mb-20 space-y-6">
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400 text-xs font-semibold uppercase tracking-widest">
                        <BookOpen className="w-4 h-4" /> Market Intelligence
                    </div>
                    <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight">
                        Trading <span className="text-blue-500">Insights</span>.
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
                            className="group bg-slate-900 border border-slate-800 rounded-2xl p-8 hover:border-blue-500/30 transition-all hover:-translate-y-1 shadow-sm"
                        >
                            <div className="mb-6 space-y-3">
                                <div className="text-xs font-bold text-blue-500 uppercase tracking-widest">{article.date}</div>
                                <h3 className="text-xl font-bold text-white group-hover:text-blue-400 transition-colors leading-tight">
                                    {article.title}
                                </h3>
                                <p className="text-slate-400 text-sm leading-relaxed line-clamp-3">
                                    {article.summary}
                                </p>
                            </div>
                            <div className="flex items-center gap-2 text-sm font-semibold text-white group-hover:gap-3 transition-all">
                                Read Article <ArrowRight className="w-4 h-4 text-blue-500" />
                            </div>
                        </Link>
                    ))}
                </div>

            </div>
        </div>
    );
};

export default ArticlesHub;
