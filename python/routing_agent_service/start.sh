#!/bin/bash
# start.sh - Improved wrapper that properly handles port conflicts

echo "Starting Flask app wrapper..."
echo "Wrapper PID: $$"

# Function to kill any existing processes on port 8080
cleanup_port() {
    echo "Checking for processes on port 8080..."
    local pids=$(lsof -ti:8080 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "Found processes on port 8080: $pids"
        echo "Killing processes: $pids"
        kill -TERM $pids 2>/dev/null
        sleep 2
        # Force kill if still running
        kill -KILL $pids 2>/dev/null
        sleep 1
    fi
}

# Function to handle signals
cleanup() {
    echo "Received signal, shutting down..."
    if [ ! -z "$FLASK_PID" ]; then
        echo "Killing Flask PID: $FLASK_PID"
        kill -TERM $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
    cleanup_port
    echo "Wrapper exiting"
    exit 0
}

# Function to restart Flask
restart_flask() {
    echo "Restarting Flask..."
    if [ ! -z "$FLASK_PID" ]; then
        echo "Stopping current Flask PID: $FLASK_PID"
        kill -TERM $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
    cleanup_port
    sleep 2
    start_flask
}

# Function to start Flask
start_flask() {
    echo "Starting Flask application..."
    cd /app
    
    # Clean up any leftover processes first
    cleanup_port
    
    # Wait a moment for port to be released
    sleep 1
    
    python routing_agent_service.py &
    FLASK_PID=$!
    echo "Flask started with PID: $FLASK_PID"
    
    # Give Flask a moment to start up and verify it's running
    sleep 3
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "ERROR: Flask failed to start properly"
        return 1
    fi
    
    echo "Flask is running successfully on PID: $FLASK_PID"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT
trap restart_flask SIGUSR1

# Install lsof if not available (add to Dockerfile would be better)
if ! command -v lsof &> /dev/null; then
    echo "lsof not found, installing..."
    apt-get update && apt-get install -y lsof
fi

mkdir -p /app/logs

# Initial cleanup
cleanup_port
sleep 2

start_flask
while true; do
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "Flask process died, restarting..."
        sleep 2
        start_flask
    fi
    if [ -f "/app/.restart_trigger" ]; then
        echo "Found restart trigger file, restarting Flask..."
        rm -f /app/.restart_trigger
        restart_flask
    fi
    sleep 5
done