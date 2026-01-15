const colors = require('tailwindcss/colors')

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ash: '#0f1011',
        'black-ash': '#0a0a0a',
        slate: colors.neutral,
        brand: {
          light: '#fb923c', // Amber 400
          DEFAULT: '#f59e0b', // Amber 500
          dark: '#ea580c', // Orange 600
        },
        flame: {
          DEFAULT: '#ef4444', // Red 500
          dark: '#b91c1c',    // Red 700
          deep: '#7f1d1d',    // Red 900
        }
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { opacity: 0.3 },
          '50%': { opacity: 0.6 },
        }
      }
    },
  },
  plugins: [
    require("tailwindcss-animate"),
  ],
}
