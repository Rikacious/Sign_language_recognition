@echo off

echo 1. Start Hand Tracking
echo 2. Start Collecting Data
echo 3. Start Train Dataset

set /p c=Enter Your Choise:

if %c%==1 (
    python handTrack.py
) else if %c%==1 (
    python collectData.py
) else if %c%==1 (
    python trainDataset.py
) else (
    echo You've Entered the Wrong Input.
)

@pause