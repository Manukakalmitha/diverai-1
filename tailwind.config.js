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
      }
    },
  },
  plugins: [
    require("tailwindcss-animate"),
  ],
}
