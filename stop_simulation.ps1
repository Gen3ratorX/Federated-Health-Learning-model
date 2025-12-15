# stop_simulation.ps1

Write-Host "🛑 STOPPING FEDERATED SIMULATION..." -ForegroundColor Red

# Kill all Python processes (Aggressive clean up)
# Note: This kills ALL python processes. If you are running other python apps, be careful.
taskkill /F /IM python.exe /T 2>$null

Write-Host "✅ All system processes terminated." -ForegroundColor Green
Write-Host "   (You may close the remaining PowerShell windows manually if they stay open)"