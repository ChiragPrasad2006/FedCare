# Local Kubernetes Deployment Guide

This is the recommended setup for a college project if you want to avoid cloud costs.

It runs FedCare on **your own laptop or desktop** using local Kubernetes resources:

- `1` main server
- `2` hospital servers
- `0` cloud hosting cost

## Why use local Kubernetes

Benefits:

- no Google Cloud compute bill
- no always-on hosted service charges
- easier to control CPU and RAM usage
- still lets you present a real distributed architecture

Tradeoff:

- the model uses your system resources, not Google Cloud resources
- if your laptop has limited RAM, TensorFlow can still feel heavy

## Recommended local options

Use one of these:

- `Docker Desktop` with Kubernetes enabled
- `Minikube`

For most students, `Minikube` is easier to reason about.

## Architecture used locally

The Kubernetes manifest in [deployment.yml](C:/Users/chira/Documents/GitHub/FedCare/kubernetes/deployment.yml:1) creates:

- `main-server`
- `hospital-server-1`
- `hospital-server-2`

Each runs as its own deployment with its own internal Kubernetes service.

## Prerequisites

Install:

1. `Docker Desktop` or Docker Engine
2. `kubectl`
3. `Minikube` if you are not using Docker Desktop Kubernetes

Optional but helpful:

- at least `8 GB` RAM on your system

## Option A: Run with Minikube

### Step 1: Start Minikube

```bash
minikube start --cpus=4 --memory=8192 --driver=docker
```

If your machine is smaller, try:

```bash
minikube start --cpus=2 --memory=4096 --driver=docker
```

That smaller setup may still work for a light presentation demo, but model startup can be slower.

### Step 2: Point Docker to Minikube

This makes Docker build images directly inside Minikube so Kubernetes can use them without pushing to a registry.

On PowerShell:

```powershell
minikube -p minikube docker-env --shell powershell | Invoke-Expression
```

On bash:

```bash
eval $(minikube docker-env)
```

### Step 3: Build the images

From the project root:

```bash
docker build -t fedcare-main-server:latest -f docker/Dockerfile.main_server .
docker build -t fedcare-hospital-server:latest -f docker/Dockerfile.hospital_server .
```

### Step 4: Deploy to Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yml
```

### Step 5: Verify pods

```bash
kubectl get pods -n fedcare
kubectl get svc -n fedcare
```

You should see:

- `main-server`
- `hospital-server-1`
- `hospital-server-2`

### Step 6: Open the dashboards with port forwarding

Use three terminals.

Terminal 1:

```bash
kubectl port-forward -n fedcare svc/main-server 5000:5000
```

Terminal 2:

```bash
kubectl port-forward -n fedcare svc/hospital-server-1 5001:5001
```

Terminal 3:

```bash
kubectl port-forward -n fedcare svc/hospital-server-2 5002:5001
```

Then open:

- `http://localhost:5000/`
- `http://localhost:5001/`
- `http://localhost:5002/`

## Option B: Run with Docker Desktop Kubernetes

### Step 1: Enable Kubernetes

In Docker Desktop:

1. Open `Settings`
2. Go to `Kubernetes`
3. Enable Kubernetes
4. Wait until the cluster is ready

### Step 2: Build images locally

From the project root:

```bash
docker build -t fedcare-main-server:latest -f docker/Dockerfile.main_server .
docker build -t fedcare-hospital-server:latest -f docker/Dockerfile.hospital_server .
```

### Step 3: Deploy

```bash
kubectl apply -f kubernetes/deployment.yml
```

### Step 4: Port-forward the services

```bash
kubectl port-forward -n fedcare svc/main-server 5000:5000
kubectl port-forward -n fedcare svc/hospital-server-1 5001:5001
kubectl port-forward -n fedcare svc/hospital-server-2 5002:5001
```

or 

``bash
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up --build
```

Open:

- `http://localhost:5000/`
- `http://localhost:5001/`
- `http://localhost:5002/`

## Presentation flow

Use this order during your demo:

1. Open the main dashboard on `localhost:5000`
2. Click `Initialize System`
3. Open hospital 1 on `localhost:5001`
4. Set hospital ID to `hospital_1`
5. Click `Configure Hospital`
6. Click `Generate Demo Records`
7. Click `Load Demo Training Data`
8. Open hospital 2 on `localhost:5002`
9. Set hospital ID to `hospital_2`
10. Click `Configure Hospital`
11. Click `Generate Demo Records`
12. Click `Load Demo Training Data`
13. On both hospital pages, click `Sync and Train`
14. Return to the main dashboard and click `Aggregate Updates`

## Useful commands

Check deployments:

```bash
kubectl get deployments -n fedcare
```

Check pods:

```bash
kubectl get pods -n fedcare
```

Check logs:

```bash
kubectl logs -n fedcare deployment/main-server
kubectl logs -n fedcare deployment/hospital-server-1
kubectl logs -n fedcare deployment/hospital-server-2
```

Restart a deployment:

```bash
kubectl rollout restart deployment/main-server -n fedcare
kubectl rollout restart deployment/hospital-server-1 -n fedcare
kubectl rollout restart deployment/hospital-server-2 -n fedcare
```

Delete everything:

```bash
kubectl delete namespace fedcare
```

## Resource advice for a student laptop

Current manifest limits:

- main server: `1 CPU`, `1 GiB RAM`
- each hospital server: `0.5 CPU`, `512 MiB RAM`

If your system struggles:

1. Lower TensorFlow workload during demos
2. Keep only `2` hospital servers
3. Use the dashboard demo buttons instead of large datasets
4. Close heavy background apps before starting Minikube or Docker Desktop

## Important note

This local Kubernetes setup is the best choice if:

- you do not want to add a billing method for cloud hosting
- you want full control over cost
- the project is mainly for demonstration, presentation, and architecture explanation
