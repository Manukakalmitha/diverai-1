import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { articles } from '../data/articles';
import { ArrowLeft, Calendar, Clock, Share2 } from 'lucide-react';
import { Helmet } from 'react-helmet-async';

const ArticlePage = () => {
    const { slug } = useParams();
    const navigate = useNavigate();
    const article = articles.find(a => a.id === slug);

    if (!article) {
        return (
            <div className="min-h-screen bg-[#0f172a] flex items-center justify-center text-slate-400">
                Article not found.
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans selection:bg-blue-500/30 pt-24 pb-20">
            <Helmet>
                <title>{`${article.title} | Diver AI Insights`}</title>
                <meta name="description" content={article.summary} />
                <meta property="og:title" content={article.title} />
                <meta property="og:description" content={article.summary} />
                <meta property="og:type" content="article" />
            </Helmet>
            <article className="max-w-3xl mx-auto px-6">

                <button
                    onClick={() => navigate('/articles')}
                    className="flex items-center gap-2 text-blue-500 font-bold uppercase tracking-widest text-xs mb-8 hover:text-blue-400 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" /> Back to Articles
                </button>

                <header className="mb-12 border-b border-slate-800 pb-12">
                    <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight mb-6 leading-tight">
                        {article.title}
                    </h1>

                    <div className="flex flex-wrap items-center gap-6 text-sm text-slate-500 font-medium">
                        <div className="flex items-center gap-2">
                            <Calendar className="w-4 h-4" /> {article.date}
                        </div>
                        <div className="flex items-center gap-2">
                            <Clock className="w-4 h-4" /> {article.readTime}
                        </div>
                        <div className="flex-1 text-right">
                            <button className="p-2 hover:bg-slate-900 rounded-full transition-colors" title="Share">
                                <Share2 className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </header>

                <div
                    className="prose prose-invert prose-lg max-w-none prose-headings:font-bold prose-headings:tracking-tight prose-a:text-blue-400 hover:prose-a:text-blue-300 prose-img:rounded-xl"
                    dangerouslySetInnerHTML={{ __html: article.content }}
                />

                <footer className="mt-20 pt-12 border-t border-slate-800">
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 md:p-12 text-center">
                        <h3 className="text-2xl font-bold text-white mb-4">Ready to test these concepts?</h3>
                        <p className="text-slate-400 mb-8 max-w-lg mx-auto">
                            See Diver AI's pattern recognition engine in action on live markets.
                        </p>
                        <button
                            onClick={() => navigate('/analysis')}
                            className="px-8 py-3.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg transition-colors shadow-lg shadow-blue-500/20"
                        >
                            Launch Terminal
                        </button>
                    </div>
                </footer>

            </article>
        </div>
    );
};

export default ArticlePage;
