# run_simulation.ps1

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   🏥  FEDERATED HEALTH OS - AUTO LAUNCHER  🏥"
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Define Paths
$VenvPath = ".\venv\Scripts\Activate.ps1"
$CoordScript = "coordinator/app/main.py"
$ClientScript = "hospital/app/client.py"

# 2. Start the Coordinator (The Brain)
Write-Host "🚀 Step 1: Starting Coordinator API..." -ForegroundColor Yellow
$CoordinatorProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "& $VenvPath; python $CoordScript" -PassThru

# Give it time to boot up (Uvicorn needs a moment)
Write-Host "   ⏳ Waiting for API to initialize (5 seconds)..." -ForegroundColor DarkGray
Start-Sleep -Seconds 5

# 3. Open the Dashboard automatically
Write-Host "🖥️  Step 2: Opening Dashboard..." -ForegroundColor Yellow
Start-Process "http://localhost:8000/dashboard"

# 4. Trigger the Training (Auto-Click the 'Start' button via API)
Write-Host "📡 Step 3: Auto-Starting Training Session..." -ForegroundColor Yellow
try {
    $Params = @{
        num_rounds = 5
        min_clients = 3
    } | ConvertTo-Json

    Invoke-RestMethod -Uri "http://localhost:8000/api/training/start" `
        -Method Post `
        -ContentType "application/json" `
        -Body $Params
    
    Write-Host "   ✅ Training signal sent successfully!" -ForegroundColor Green
}
catch {
    Write-Host "   ⚠️  Could not auto-start training. Please click 'Start' on the dashboard manually." -ForegroundColor Red
}

# Give Flower server a moment to spin up on port 8080
Start-Sleep -Seconds 3

# 5. Launch the 3 Hospitals
Write-Host "🚑 Step 4: Dispatching Hospital Clients..." -ForegroundColor Yellow

$Hospitals = 1..3
foreach ($id in $Hospitals) {
    Write-Host "   - Launching Hospital $id..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "& $VenvPath; python $ClientScript $id --server localhost:8080"
}

Write-Host "`n✅ SIMULATION RUNNING!" -ForegroundColor Green
Write-Host "   Check the web dashboard to watch progress."
Write-Host "   Run 'stop_simulation.ps1' to close everything."