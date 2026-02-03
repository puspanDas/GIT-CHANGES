/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#0052CC", // Atlassian Blue
        "primary-hover": "#0065FF",
        secondary: "#1E293B", // Slate 800 (Sidebar/Surface)
        background: "#0F172A", // Slate 900 (Main BG)
        surface: "#1E293B", // Slate 800 (Cards)
        border: "#334155", // Slate 700
        foreground: "#F8FAFC", // Slate 50
        muted: "#94A3B8", // Slate 400
        success: "#10B981", // Emerald 500
        warning: "#F59E0B", // Amber 500
        danger: "#EF4444", // Red 500
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
