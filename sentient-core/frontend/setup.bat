@echo off
echo Installing Next.js dependencies...

npm install next@15 react@19 react-dom@19

echo Installing UI dependencies...
npm install @radix-ui/react-dropdown-menu @radix-ui/react-slot @radix-ui/react-tabs
npm install class-variance-authority clsx lucide-react
npm install next-themes
npm install tailwind-merge tailwindcss-animate

echo Installing Data Fetching and State Management...
npm install @tanstack/react-query zustand axios

echo Installing Form and Utility Libraries...
npm install react-hook-form react-textarea-autosize react-markdown

echo Installing Developer Dependencies...
npm install --save-dev typescript @types/node @types/react @types/react-dom
npm install --save-dev autoprefixer postcss tailwindcss

echo Setup complete! You can now run 'npm run dev' to start the development server.
