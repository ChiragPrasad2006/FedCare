#!/bin/bash
# Deploy FedCare to Kubernetes

set -e

echo "========================================"
echo "FedCare - Deploying to Kubernetes"
echo "========================================"
echo ""

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed"
    echo "Please install kubectl from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

echo "Building Docker images..."
docker build -t fedcare-main-server:latest -f docker/Dockerfile.main_server .
docker build -t fedcare-hospital-server:latest -f docker/Dockerfile.hospital_server .

echo ""
echo "Checking Kubernetes cluster..."
kubectl cluster-info

echo ""
echo "Loading images into Kubernetes..."
# For minikube
if command -v minikube &> /dev/null; then
    echo "Loading into minikube..."
    minikube image load fedcare-main-server:latest
    minikube image load fedcare-hospital-server:latest
fi

echo ""
echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/deployment.yml

echo ""
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=main-server -n fedcare --timeout=300s 2>/dev/null || true
sleep 5

echo ""
echo "Checking deployment status..."
kubectl get pods -n fedcare
kubectl get svc -n fedcare

echo ""
echo "========================================"
echo "Deployment completed!"
echo "========================================"
echo ""
echo "Port forwarding commands:"
echo "  kubectl port-forward -n fedcare svc/main-server 5000:5000"
echo "  kubectl port-forward -n fedcare svc/hospital-server 5001:5001"
echo ""
echo "View logs:"
echo "  kubectl logs -n fedcare deployment/main-server"
echo "  kubectl logs -n fedcare deployment/hospital-server"
echo ""
echo "Delete deployment:"
echo "  kubectl delete namespace fedcare"
echo ""
