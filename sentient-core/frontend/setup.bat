@echo off
echo Installing Next.js dependencies...

echo Cleaning node_modules and package-lock.json if they exist...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json

echo Installing Next.js with React 18 (for better compatibility)...
npm install next@14.1.0 react@18.2.0 react-dom@18.2.0

echo Installing UI dependencies...
npm install @radix-ui/react-dropdown-menu @radix-ui/react-slot @radix-ui/react-tabs
npm install class-variance-authority clsx lucide-react
npm install next-themes
npm install tailwind-merge tailwindcss-animate

echo Installing Data Fetching and State Management...
npm install @tanstack/react-query@5 zustand@4 axios

echo Installing Form and Utility Libraries...
npm install react-hook-form react-textarea-autosize react-markdown

echo Installing Developer Dependencies...
npm install --save-dev typescript @types/node @types/react @types/react-dom
npm install --save-dev autoprefixer postcss tailwindcss

echo Installing Shadcn CLI for UI components...
npm install -g shadcn-ui@latest

echo Creating components directory if it doesn't exist...
if not exist components mkdir components
if not exist components\ui mkdir components\ui

echo Setup complete! You can now run 'npm run dev' to start the development server.
echo.
echo NOTE: If you still encounter dependency issues, try installing with:
echo npm install --legacy-peer-deps
