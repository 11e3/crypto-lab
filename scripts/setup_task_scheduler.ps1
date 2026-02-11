
    # Get the current user
    $user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    
    # Create trigger (daily at specified time)
    $trigger = New-ScheduledTaskTrigger -Daily -At 09:00
    
    # Create action
    $action = New-ScheduledTaskAction -Execute 'C:\Users\moons\AppData\Local\Programs\Python\Python314\python.exe' `
        -Argument '-m scripts.real_time_monitor --tickers KRW-BTC KRW-ETH --output "C:\workspace\dev\crypto-lab\reports"' `
        -WorkingDirectory 'C:\workspace\dev\crypto-lab'
    
    # Create task settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    # Register task
    Register-ScheduledTask -TaskName "CryptoQuantMonitoring" `
        -Trigger $trigger `
        -Action $action `
        -Settings $settings `
        -User $user `
        -Force -RunLevel Highest `
        -ErrorAction Stop
    
    Write-Host "Task created successfully: CryptoQuantMonitoring"
    Write-Host "Scheduled to run daily at 09:00"
    