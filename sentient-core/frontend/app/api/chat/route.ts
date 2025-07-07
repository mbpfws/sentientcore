import { NextRequest, NextResponse } from 'next/server';

// API proxy to handle backend communication
// This bypasses Windows process isolation issues

const BACKEND_URL = 'http://127.0.0.1:8000';

export async function GET(request: NextRequest) {
  try {
    // Forward GET requests to backend
    const response = await fetch(`${BACKEND_URL}/api/chat/message/json`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('API Proxy Error (GET):', error);
    return NextResponse.json(
      { 
        error: 'Failed to connect to backend', 
        details: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      }, 
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const body = await request.json();
    
    // Forward POST requests to backend
    const response = await fetch(`${BACKEND_URL}/api/chat/message/json`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('API Proxy Error (POST):', error);
    return NextResponse.json(
      { 
        error: 'Failed to connect to backend', 
        details: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      }, 
      { status: 500 }
    );
  }
}