'use client';

import { useState, useEffect } from 'react';

interface HealthStatus {
  status: string;
  frontend: any;
  backend: any;
  connection: string;
  error?: string;
}

interface SystemStatus {
  timestamp: string;
  system_status: string;
  frontend: any;
  backend: any;
  recommendations: string[];
}

export default function TestConnectionPage() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [chatTest, setChatTest] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/health');
      const data = await response.json();
      setHealthStatus(data);
    } catch (err) {
      setError(`Health check failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const testStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (err) {
      setError(`Status check failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const testChat = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: 'Hello from frontend test!',
          model: 'test-model',
          temperature: 0.7
        })
      });
      
      const data = await response.json();
      setChatTest(data);
    } catch (err) {
      setError(`Chat test failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const runAllTests = async () => {
    await testHealth();
    await testStatus();
    await testChat();
  };

  useEffect(() => {
    // Auto-run tests on page load
    runAllTests();
  }, []);

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">🔧 Connection Test Dashboard</h1>
      
      <div className="grid gap-6">
        {/* Control Panel */}
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Test Controls</h2>
          <div className="flex gap-4 flex-wrap">
            <button 
              onClick={testHealth}
              disabled={loading}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
            >
              Test Health
            </button>
            <button 
              onClick={testStatus}
              disabled={loading}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
            >
              Test Status
            </button>
            <button 
              onClick={testChat}
              disabled={loading}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
            >
              Test Chat
            </button>
            <button 
              onClick={runAllTests}
              disabled={loading}
              className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-50"
            >
              Run All Tests
            </button>
          </div>
          {loading && <p className="mt-2 text-blue-600">⏳ Testing...</p>}
          {error && <p className="mt-2 text-red-600">❌ {error}</p>}
        </div>

        {/* Health Status */}
        {healthStatus && (
          <div className="bg-white border rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-3">🏥 Health Status</h2>
            <div className={`p-3 rounded ${healthStatus.status === 'healthy' ? 'bg-green-100' : 'bg-red-100'}`}>
              <p className="font-medium">Status: {healthStatus.status}</p>
              <p>Connection: {healthStatus.connection}</p>
              {healthStatus.error && <p className="text-red-600">Error: {healthStatus.error}</p>}
            </div>
            <details className="mt-3">
              <summary className="cursor-pointer font-medium">View Details</summary>
              <pre className="mt-2 bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(healthStatus, null, 2)}
              </pre>
            </details>
          </div>
        )}

        {/* System Status */}
        {systemStatus && (
          <div className="bg-white border rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-3">📊 System Status</h2>
            <div className={`p-3 rounded ${
              systemStatus.system_status === 'fully_operational' ? 'bg-green-100' :
              systemStatus.system_status === 'partially_operational' ? 'bg-yellow-100' : 'bg-red-100'
            }`}>
              <p className="font-medium">System: {systemStatus.system_status}</p>
              <p>Timestamp: {systemStatus.timestamp}</p>
            </div>
            
            {systemStatus.recommendations && systemStatus.recommendations.length > 0 && (
              <div className="mt-3">
                <h3 className="font-medium mb-2">💡 Recommendations:</h3>
                <ul className="list-disc list-inside space-y-1">
                  {systemStatus.recommendations.map((rec, index) => (
                    <li key={index} className="text-sm">{rec}</li>
                  ))}
                </ul>
              </div>
            )}
            
            <details className="mt-3">
              <summary className="cursor-pointer font-medium">View Details</summary>
              <pre className="mt-2 bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(systemStatus, null, 2)}
              </pre>
            </details>
          </div>
        )}

        {/* Chat Test */}
        {chatTest && (
          <div className="bg-white border rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-3">💬 Chat Test</h2>
            <div className={`p-3 rounded ${chatTest.error ? 'bg-red-100' : 'bg-green-100'}`}>
              {chatTest.error ? (
                <>
                  <p className="font-medium text-red-600">❌ Chat Failed</p>
                  <p>Error: {chatTest.error}</p>
                  <p>Details: {chatTest.details}</p>
                </>
              ) : (
                <>
                  <p className="font-medium text-green-600">✅ Chat Successful</p>
                  <p>Response: {chatTest.response}</p>
                  <p>Model: {chatTest.model}</p>
                  <p>Processing Time: {chatTest.processing_time}s</p>
                </>
              )}
            </div>
            <details className="mt-3">
              <summary className="cursor-pointer font-medium">View Details</summary>
              <pre className="mt-2 bg-gray-100 p-2 rounded text-sm overflow-auto">
                {JSON.stringify(chatTest, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
      
      <div className="mt-8 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-semibold mb-2">🔗 Connection Info</h3>
        <p><strong>Frontend URL:</strong> http://localhost:3000</p>
        <p><strong>API Proxy:</strong> http://localhost:3000/api</p>
        <p><strong>Backend URL:</strong> http://localhost:8007</p>
        <p><strong>Environment:</strong> {process.env.NODE_ENV}</p>
      </div>
    </div>
  );
}