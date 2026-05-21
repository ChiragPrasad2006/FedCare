# Google Cloud Deployment Guide

This guide is the simplest way to present FedCare on Google Cloud with:

- `1` main server
- `2` hospital servers
- separate web dashboards for each service
- hospital nodes sending updates to the main server

## Cost warning

Google Cloud is not the best default choice if you want to guarantee zero spending.

Important points:

- Google Cloud commonly requires a billing account and payment method before you can continue
- Cloud Run has an always-free tier, but that does not guarantee your workload will stay free
- TensorFlow-based services can exceed free usage if they stay up too long or use enough CPU and RAM

For a college project with tight budget constraints, the safer option is:

- run FedCare locally with Kubernetes on your own machine
- use [LOCAL_KUBERNETES_DEPLOYMENT.md](C:/Users/chira/Documents/GitHub/FedCare/docs/LOCAL_KUBERNETES_DEPLOYMENT.md:1)

## Recommended demo architecture

Use **three Cloud Run services** in the same Google Cloud project and region:

- `fedcare-main`
- `fedcare-hospital-1`
- `fedcare-hospital-2`

Why this is the best fit for a presentation:

- each service gets its own HTTPS URL
- no VM management is needed
- the Flask apps in this repo can run directly in containers
- it is easier to explain than GKE during a project demo

## Before you deploy

1. Open `https://console.cloud.google.com/`
2. Create or select a project
3. Enable these APIs:
- Cloud Run Admin API
- Artifact Registry API
- Cloud Build API

## Repo changes that help Cloud Run

This repo now supports:

- the Cloud Run `PORT` environment variable
- `MAIN_SERVER_URL` for hospital nodes, so they can call the main server over HTTPS
- dashboard pages on `/` for the main server and each hospital server

## Step 1: Build container images

From the project root:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1

gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fedcare/main-server:latest -f docker/Dockerfile.main_server .
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fedcare/hospital-server:latest -f docker/Dockerfile.hospital_server .
```

If the `fedcare` Artifact Registry repository does not exist yet, create it first:

```bash
gcloud artifacts repositories create fedcare \
  --repository-format=docker \
  --location=us-central1
```

## Step 2: Deploy the main server

Deploy the main server first because the hospital services need its HTTPS URL:

```bash
gcloud run deploy fedcare-main \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fedcare/main-server:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000 \
  --set-env-vars MAIN_SERVER_HOST=0.0.0.0,MAIN_SERVER_PORT=5000,NUM_HOSPITALS=2,NUM_ROUNDS=5
```

After deployment, copy the service URL. It will look similar to:

```text
https://fedcare-main-xxxxxxxxxx-uc.a.run.app
```

## Step 3: Deploy hospital server 1

Replace `MAIN_URL` below with the real main server URL:

```bash
gcloud run deploy fedcare-hospital-1 \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fedcare/hospital-server:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5001 \
  --set-env-vars HOSPITAL_PORT=5001,MAIN_SERVER_URL=MAIN_URL
```

## Step 4: Deploy hospital server 2

```bash
gcloud run deploy fedcare-hospital-2 \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/fedcare/hospital-server:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5001 \
  --set-env-vars HOSPITAL_PORT=5001,MAIN_SERVER_URL=MAIN_URL
```

## Step 5: Open the dashboards

Open these URLs in the browser:

- `https://...fedcare-main.../`
- `https://...fedcare-hospital-1.../`
- `https://...fedcare-hospital-2.../`

## Step 6: Demo flow for your presentation

Use this order so the audience can follow the architecture clearly:

1. Open the main server dashboard and click `Initialize System`.
2. Open hospital server 1.
3. Set hospital ID to `hospital_1`.
4. Click `Configure Hospital`.
5. Click `Generate Demo Records`.
6. Click `Load Demo Training Data`.
7. Repeat the same steps on hospital server 2 with hospital ID `hospital_2`.
8. On each hospital page, click `Sync and Train`.
9. Return to the main server page and click `Aggregate Updates`.

What the audience will see:

- hospitals keep patient records locally
- hospitals anonymize records before presentation-friendly summaries are shown centrally
- hospitals send model updates to the main server
- the main server shows training accuracy, connected hospitals, and privacy-removal quality

## Using the Google Cloud Console instead of CLI

You can also deploy from the Console:

1. Go to `Cloud Run`
2. Click `Create Service`
3. Choose `Deploy one revision from an existing container image`
4. Select the image from Artifact Registry
5. Set the container port:
- `5000` for main server
- `5001` for hospital servers
6. In `Variables & Secrets`, add environment variables
7. Allow unauthenticated access for the demo setup
8. Create the service

Environment variables to set in the Console:

### Main server

```text
MAIN_SERVER_HOST=0.0.0.0
MAIN_SERVER_PORT=5000
NUM_HOSPITALS=2
NUM_ROUNDS=5
```

### Hospital server 1 and 2

```text
HOSPITAL_PORT=5001
MAIN_SERVER_URL=https://fedcare-main-xxxxxxxxxx-uc.a.run.app
```

## Important note about security

For a class or project presentation, `--allow-unauthenticated` is the easiest path.

For a stronger production setup:

- make the main server private
- use Cloud Run service-to-service authentication
- grant hospital service accounts permission to invoke the main service
- update the hospital code to send Google-signed ID tokens on requests

The current repo is now ready for the simple public-demo flow. The IAM-secured inter-service flow would need a small follow-up change for authenticated HTTP requests.

## Health endpoints

After deployment you can verify:

```bash
curl https://YOUR_MAIN_URL/health
curl https://YOUR_HOSPITAL_1_URL/health
curl https://YOUR_HOSPITAL_2_URL/health
```

## If you want Kubernetes later

The repo already includes:

- `docker/`
- `kubernetes/deployment.yml`

So you can also move to GKE later, but Cloud Run is the cleaner option for the current presentation goal.
