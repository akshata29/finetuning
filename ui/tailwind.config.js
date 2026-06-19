/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0e17",
          900: "#0f1420",
          850: "#141b2b",
          800: "#1a2335",
          700: "#243150",
          600: "#33425f",
        },
        accent: {
          DEFAULT: "#6366f1",
          soft: "#818cf8",
          glow: "#a5b4fc",
        },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "Consolas", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 12px 32px -12px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};
