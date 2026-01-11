import React from 'react';

const GooglePayButton = ({ onClick, disabled }) => {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className="group relative w-full h-[52px] bg-black hover:bg-zinc-900 rounded-[4px] flex items-center justify-center transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden border border-zinc-800"
            aria-label="Pay with Google Pay"
        >
            <div className="flex items-center justify-center gap-3">
                <span className="text-white font-semibold text-[15px] tracking-tight opacity-90">Buy with</span>
                <img
                    src="/google-pay-logo.png"
                    alt="Google Pay"
                    className="h-[18px] w-auto relative top-[0.5px]"
                />
            </div>

            {/* Subtle shine effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
        </button>
    );
};

export default GooglePayButton;
