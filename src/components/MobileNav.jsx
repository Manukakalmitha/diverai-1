import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, Book, Crown, User, Share2 } from 'lucide-react';

const MobileNav = () => {
    const location = useLocation();
    const isActive = (path) => location.pathname === path;

    // Hide on landing page and auth pages if they were governed by this layout (though AuthPage is usually separate)
    // Explicitly hide on landing page '/'
    if (location.pathname === '/') return null;

    return (
        <div className="md:hidden fixed bottom-0 left-0 right-0 bg-black-ash/90 backdrop-blur-xl border-t border-slate-800 z-50 pb-safe">
            <div className="flex justify-around items-center h-16 px-2">
                <Link to="/analysis" className={`flex flex-col items-center justify-center w-full h-full space-y-1 ${isActive('/analysis') ? 'text-brand' : 'text-slate-500'}`}>
                    <Activity className={`w-5 h-5 ${isActive('/analysis') ? 'fill-brand/20' : ''}`} />
                    <span className="text-[9px] font-black uppercase tracking-widest">Terminal</span>
                </Link>
                <Link to="/pricing" className={`flex flex-col items-center justify-center w-full h-full space-y-1 ${isActive('/pricing') ? 'text-brand' : 'text-slate-500'}`}>
                    <Crown className={`w-5 h-5 ${isActive('/pricing') ? 'fill-brand/20' : ''}`} />
                    <span className="text-[9px] font-black uppercase tracking-widest">Pricing</span>
                </Link>
                <Link to="/docs" className={`flex flex-col items-center justify-center w-full h-full space-y-1 ${isActive('/docs') ? 'text-brand' : 'text-slate-500'}`}>
                    <Book className={`w-5 h-5 ${isActive('/docs') ? 'fill-brand/20' : ''}`} />
                    <span className="text-[9px] font-black uppercase tracking-widest">Docs</span>
                </Link>
                <Link to="/profile" className={`flex flex-col items-center justify-center w-full h-full space-y-1 ${isActive('/profile') ? 'text-brand' : 'text-slate-500'}`}>
                    <Activity className={`w-5 h-5 ${isActive('/profile') ? 'fill-brand/20' : ''}`} />
                    <span className="text-[9px] font-black uppercase tracking-widest">Account</span>
                </Link>
                <Link to="/referral" className={`flex flex-col items-center justify-center w-full h-full space-y-1 ${isActive('/referral') ? 'text-brand' : 'text-slate-500'}`}>
                    <Share2 className={`w-5 h-5 ${isActive('/referral') ? 'fill-brand/20' : ''}`} />
                    <span className="text-[9px] font-black uppercase tracking-widest">Referral</span>
                </Link>
            </div>
        </div>
    );
};

export default MobileNav;
