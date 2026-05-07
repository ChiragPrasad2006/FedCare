#!/bin/bash
# Start FedCare with Docker Compose

set -e

echo "========================================"
echo "FedCare - Starting with Docker Compose"
echo "========================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

echo "Building Docker images..."
docker-compose -f docker/docker-compose.yml build

echo ""
echo "Starting containers..."
docker-compose -f docker/docker-compose.yml up -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

# Check if services are running
echo ""
echo "Checking service status..."

# Check main server
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✓ Main Server: Running (http://localhost:5000)"
else
    echo "✗ Main Server: Not responding"
fi

# Check hospital servers
for port in 5001 5002 5003; do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✓ Hospital Server: Running (http://localhost:$port)"
    else
        echo "✗ Hospital Server: Not responding on port $port"
    fi
done

echo ""
echo "========================================"
echo "Services started successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run examples: python examples.py"
echo "2. Run orchestrator: python orchestrator.py"
echo "3. View logs: docker-compose -f docker/docker-compose.yml logs -f"
echo "4. Stop services: docker-compose -f docker/docker-compose.yml down"
echo ""
