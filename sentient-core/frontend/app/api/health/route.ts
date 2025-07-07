import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = 'http://127.0.0.1:8000';

export async function GET(request: NextRequest) {
  try {
    // Check backend health
    const response = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add timeout to prevent hanging
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });

    if (!response.ok) {
      throw new Error(`Backend health check failed with status: ${response.status}`);
    }

    const backendHealth = await response.json();
    
    return NextResponse.json({
      status: 'healthy',
      frontend: {
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV,
        api_url: process.env.NEXT_PUBLIC_API_URL,
      },
      backend: backendHealth,
      connection: 'successful'
    });
  } catch (error) {
    console.error('Health Check Error:', error);
    
    return NextResponse.json(
      {
        status: 'unhealthy',
        frontend: {
          timestamp: new Date().toISOString(),
          environment: process.env.NODE_ENV,
          api_url: process.env.NEXT_PUBLIC_API_URL,
        },
        backend: null,
        connection: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 503 }
    );
  }
}