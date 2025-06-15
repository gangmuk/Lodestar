#!/bin/bash
# start.sh - Simple wrapper that keeps running even if Flask dies

echo "Starting Flask app wrapper..."
echo "Wrapper PID: $$"

# Function to handle signals
cleanup() {
    echo "Received signal, shutting down..."
    if [ ! -z "$FLASK_PID" ]; then
        echo "Killing Flask PID: $FLASK_PID"
        kill -TERM $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
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
    start_flask
}

# Function to start Flask
start_flask() {
    echo "Starting Flask application..."
    cd /app
    python routing_agent_service.py &
    FLASK_PID=$!
    echo "Flask started with PID: $FLASK_PID"
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT
trap restart_flask SIGUSR1

mkdir -p /app/logs

start_flask
while true; do
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "Flask process died, restarting..."
        start_flask
    fi
    if [ -f "/app/.restart_trigger" ]; then
        echo "Found restart trigger file, restarting Flask..."
        rm -f /app/.restart_trigger
        restart_flask
    fi
    sleep 5
done