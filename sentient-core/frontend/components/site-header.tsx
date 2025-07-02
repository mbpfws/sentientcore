'use client';

import dynamic from 'next/dynamic';

// Dynamically import ThemeToggle to prevent SSR mismatches
const ThemeToggle = dynamic(() => import('@/components/theme-toggle').then(m => m.ThemeToggle), { ssr: false });

export default function SiteHeader() {
  return (
    <header className="border-b">
      <div className="container flex items-center justify-between py-4">
        <h1 className="text-2xl font-bold">Sentient Core</h1>
        <div className="flex items-center gap-4">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
