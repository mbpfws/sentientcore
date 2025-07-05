'use client';

import React from 'react';
import { InteractiveWorkflowDashboard } from '@/components/interactive-workflow/InteractiveWorkflowDashboard';

export default function InteractiveWorkflowsPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="space-y-6">
        {/* Page Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Interactive Workflows</h1>
          <p className="text-muted-foreground">
            Create, manage, and monitor interactive workflows with step-by-step execution, 
            approval gates, and real-time progress tracking.
          </p>
        </div>

        {/* Dashboard */}
        <InteractiveWorkflowDashboard />
      </div>
    </div>
  );
}