import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#070B12",
        "background-raised": "#0B1220",
        surface: "#0F1827",
        "surface-soft": "#142134",
        foreground: "#EEF7FF",
        muted: "#8EA3B8",
        subtle: "#5E7287",
        border: "#26384B",
        accent: "#68E8FF",
        success: "#8BFFCA",
        warning: "#FFD183",
        danger: "#FF7C8A",
        violet: "#B9A5FF",
      },
      borderRadius: {
        sm: "12px",
        md: "18px",
        lg: "24px",
        xl: "34px",
        pill: "999px",
      },
      fontFamily: {
        sans: ["var(--font-geist)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-ibm-plex-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(100%)" },
        },
        breathe: {
          "0%, 100%": { transform: "scale(1)", filter: "brightness(1)" },
          "50%": { transform: "scale(0.94)", filter: "brightness(1.25)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-5px)" },
        },
        "spin-slow": {
          to: { transform: "rotate(360deg)" },
        },
      },
      animation: {
        scan: "scan 2.6s linear infinite",
        breathe: "breathe 3.4s ease-in-out infinite",
        float: "float 4.5s ease-in-out infinite",
        "spin-slow": "spin-slow 9s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
