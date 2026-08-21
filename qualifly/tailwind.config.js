/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#355872",
        secondary: "#7AAACE",
        accent: "#9CD5FF",
        surface: "#F7F8F0",
      },
    },
  },
  plugins: [],
};