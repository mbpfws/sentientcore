import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  
  try {
    // Test multiple backend endpoints
    const [healthResponse, statusResponse, testResponse] = await Promise.allSettled([
      fetch(`${BACKEND_URL}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      }),
      fetch(`${BACKEND_URL}/api/status`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      }),
      fetch(`${BACKEND_URL}/api/test`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      })
    ]);

    const responseTime = Date.now() - startTime;

    // Process results
    const results = {
      health: healthResponse.status === 'fulfilled' && healthResponse.value.ok,
      status: statusResponse.status === 'fulfilled' && statusResponse.value.ok,
      test: testResponse.status === 'fulfilled' && testResponse.value.ok
    };

    const successfulConnections = Object.values(results).filter(Boolean).length;
    const totalTests = Object.keys(results).length;

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      system_status: successfulConnections === totalTests ? 'fully_operational' : 
                    successfulConnections > 0 ? 'partially_operational' : 'down',
      frontend: {
        status: 'operational',
        environment: process.env.NODE_ENV || 'development',
        api_url: process.env.NEXT_PUBLIC_API_URL || 'not_configured',
        proxy_enabled: true
      },
      backend: {
        url: BACKEND_URL,
        connectivity: {
          health_endpoint: results.health,
          status_endpoint: results.status,
          test_endpoint: results.test
        },
        response_time_ms: responseTime,
        success_rate: `${successfulConnections}/${totalTests}`
      },
      recommendations: successfulConnections === 0 ? [
        'Check if development_api_server.py is running',
        'Verify backend is accessible on port 8008',
        'Check Windows firewall settings',
        'Try running server directly in IDE'
      ] : successfulConnections < totalTests ? [
        'Some backend endpoints are not responding',
        'Check backend logs for errors',
        'Verify all services are properly initialized'
      ] : [
        'System is fully operational',
        'All backend endpoints are responding correctly'
      ]
    });

  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    return NextResponse.json(
      {
        timestamp: new Date().toISOString(),
        system_status: 'error',
        frontend: {
          status: 'operational',
          environment: process.env.NODE_ENV || 'development',
          api_url: process.env.NEXT_PUBLIC_API_URL || 'not_configured',
          proxy_enabled: true
        },
        backend: {
          url: BACKEND_URL,
          connectivity: 'failed',
          response_time_ms: responseTime,
          error: error instanceof Error ? error.message : 'Unknown error'
        },
        recommendations: [
          'Backend is completely unreachable',
          'Start development_api_server.py',
          'Check network connectivity',
          'Review server logs for startup errors'
        ]
      },
      { status: 503 }
    );
  }
}