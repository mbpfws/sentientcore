# Test script for Sentient Core API
$body = @{
    message = "Hello, can you help me with a simple task?"
    workflow_mode = "intelligent"
} | ConvertTo-Json

Write-Host "Testing API endpoint..."
Write-Host "Request body: $body"

try {
    $response = Invoke-RestMethod -Uri 'http://localhost:8000/api/chat/message' -Method POST -ContentType 'application/json' -Body $body
    Write-Host "Success! Response:"
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host "Status Code: $($_.Exception.Response.StatusCode)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body: $responseBody"
    }
}