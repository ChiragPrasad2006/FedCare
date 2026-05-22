#!/bin/bash
# Start FedCare servers locally (for development)

set -e

echo "========================================"
echo "FedCare - Starting Servers Locally"
echo "========================================"
echo ""

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create logs directory
mkdir -p logs

echo ""
echo "Starting servers..."
echo ""

# Start main server in background
echo "Starting Main Server on port 5000..."
$PYTHON main_server/app.py > logs/main_server.log 2>&1 &
MAIN_PID=$!
echo "  PID: $MAIN_PID"

# Wait for main server to start
sleep 2

# Start hospital servers
for i in 1 2 3; do
    port=$((5000 + i))
    hospital_id="hospital_$i"
    echo "Starting Hospital Server $i on port $port..."
    HOSPITAL_ID=$hospital_id PORT=$port $PYTHON hospital_server/app.py > logs/hospital_server_$i.log 2>&1 &
    HOSPITAL_PIDS="$HOSPITAL_PIDS $!"
    echo "  PID: $!"
done

echo ""
echo "========================================"
echo "Servers started successfully!"
echo "========================================"
echo ""
echo "Main Server: http://localhost:5000"
echo "Hospital 1:  http://localhost:5001"
echo "Hospital 2:  http://localhost:5002"
echo "Hospital 3:  http://localhost:5003"
echo ""
echo "To view logs:"
echo "  tail -f logs/main_server.log"
echo "  tail -f logs/hospital_server_*.log"
echo ""
echo "To stop all servers, run:"
echo "  kill $MAIN_PID $HOSPITAL_PIDS"
echo ""
echo "Or run the orchestrator in another terminal:"
echo "  python orchestrator.py"
echo ""

# Wait for all processes
wait
