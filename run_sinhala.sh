#!/bin/bash
# File: run_mining.sh

# Create logs directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/muss_mining_${TIMESTAMP}.log"

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"

# Run with nohup
echo "Starting mining process at $(date)" > $LOG_FILE
echo "---------------------------------------" >> $LOG_FILE
nohup python scripts/mine_sinhala.py >> $LOG_FILE 2>&1 &

# Save PID
echo $! > logs/muss_mining_${TIMESTAMP}.pid
PID=$(cat logs/muss_mining_${TIMESTAMP}.pid)

echo "Process started with PID $PID"
echo "Log file: $LOG_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop process: kill $PID"