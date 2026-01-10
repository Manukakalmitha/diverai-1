import React from 'react';

const GooglePayButton = ({ onClick, disabled }) => {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className="group relative w-full h-[52px] bg-black hover:bg-zinc-900 rounded-[4px] flex items-center justify-center transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden border border-zinc-800"
            aria-label="Pay with Google Pay"
        >
            <div className="flex items-center justify-center gap-2">
                <span className="text-white font-medium text-[16px] tracking-tight">Buy with</span>
                <svg
                    width="54"
                    height="22"
                    viewBox="0 0 54 22"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="relative top-[1px]"
                >
                    <path
                        d="M51.92 10.5C51.92 14.94 48.33 18.52 43.83 18.52C39.34 18.52 35.75 14.94 35.75 10.5C35.75 6.06 39.34 2.47 43.83 2.47C48.33 2.47 51.92 6.06 51.92 10.5ZM48.61 10.5C48.61 7.68 46.54 5.61 43.83 5.61C41.13 5.61 39.06 7.68 39.06 10.5C39.06 13.31 41.13 15.39 43.83 15.39C46.54 15.39 48.61 13.31 48.61 10.5Z"
                        fill="white"
                    />
                    <path
                        d="M19.16 10.5C19.16 14.94 15.57 18.52 11.08 18.52C6.58 18.52 3 14.94 3 10.5C3 6.06 6.58 2.47 11.08 2.47C15.57 2.47 19.16 6.06 19.16 10.5ZM15.86 10.5C15.86 7.68 13.79 5.61 11.08 5.61C8.38 5.61 6.31 7.68 6.31 10.5C6.31 13.31 8.38 15.39 11.08 15.39C13.79 15.39 15.86 13.31 15.86 10.5Z"
                        fill="white"
                    />
                    <path
                        d="M34.33 7.82V17.7C34.33 21.78 31.93 23.45 28.96 23.45C26.11 23.45 24.39 21.53 23.74 19.98L26.61 18.79C27.13 20.01 28.29 20.57 28.96 20.57C30.85 20.57 32.02 19.41 32.02 17.21V16.48H31.91C31.33 17.18 30.22 17.82 28.79 17.82C25.86 17.82 23.23 15.28 23.23 12.01C23.23 8.74 25.86 6.19 28.79 6.19C30.22 6.19 31.33 6.83 31.91 7.51H32.02V6.63H34.33V7.82ZM32.14 12.01C32.14 9.17 30.85 7.12 28.94 7.12C26.96 7.12 25.5 9.17 25.5 12.01C25.5 14.82 26.96 16.89 28.94 16.89C30.85 16.89 32.14 14.82 32.14 12.01Z"
                        fill="white"
                    />
                    <path
                        d="M57.65 6.63H60.03V17.82H57.65V6.63Z"
                        fill="white"
                    />
                    <path
                        d="M68.73 13.88L70.62 15.14C70.02 16.03 68.32 17.82 65.41 17.82C62.08 17.82 59.57 15.24 59.57 12C59.57 8.68 62.11 6.19 65.11 6.19C68.12 6.19 69.57 8.58 70.05 9.8L70.36 10.6L62.7 13.77C63.29 14.93 64.21 15.52 65.41 15.52C66.62 15.52 67.5 14.92 68.11 14.15L68.73 13.88ZM62.11 11.85L67.11 9.77C66.86 9.13 66.1 8.67 65.23 8.67C64.12 8.67 62.08 9.65 62.11 11.85Z"
                        fill="white"
                    />
                    <path
                        d="M10.74 3.73C10.74 1.77 9.18 0.17 7.22 0.17H0V16.89H2.38V12.11H7.22C9.18 12.11 10.74 10.51 10.74 8.55V3.73ZM8.36 8.55C8.36 9.18 7.85 9.69 7.22 9.69H2.38V2.58H7.22C7.85 2.58 8.36 3.09 8.36 3.73V8.55Z"
                        fill="white"
                    />
                    <path
                        d="M21.16 6.63H18.78V16.89H21.16V13.84H25.03L28.1 16.89H31.14L27.11 12.87C28.46 12.29 29.41 10.95 29.41 9.38C29.41 7.86 28.24 6.63 26.68 6.63H21.16ZM26.68 11.33H21.16V8.55H26.68C27.31 8.55 27.82 9.06 27.82 9.69C27.82 10.32 27.31 10.83 26.68 11.33Z"
                        fill="white"
                    />
                    <path
                        d="M38.86 16.89H41.28L42.53 14.07H47.41L48.66 16.89H51.08L46.2 6.63H43.74L38.86 16.89ZM44.97 8.52L46.66 12.33H43.28L44.97 8.52Z"
                        fill="white"
                    />
                    <path
                        d="M54.54 6.63L57.8 11.59L61.06 6.63H63.69L59.01 13.19V16.89H56.59V13.19L51.91 6.63H54.54Z"
                        fill="white"
                    />
                </svg>
            </div>

            {/* Subtle shine effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
        </button>
    );
};

export default GooglePayButton;
