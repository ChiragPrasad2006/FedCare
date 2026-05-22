# Hospital UI - Patient Data Retrieval Guide

## Overview

The hospital server UI has been enhanced with a **"Retrieve Patient Data"** feature that allows you to view and browse anonymized patient records received from other hospitals. This is useful for:

- **Presentation purposes**: Review and present data to stakeholders
- **Quality assurance**: Verify data anonymization 
- **Data exploration**: Understand patterns in received data
- **Testing**: Validate the federated learning data flow

## How to Use

### 1. Generate Test Data

First, generate test data for the hospital servers:

```bash
python generate_test_data.py
```

This creates MNIST-like data split across 3 hospitals (2000 samples each).

### 2. Start the Servers

In separate terminals:

**Terminal 1 - Main Server:**
```bash
python main_server/app.py
```

**Terminal 2 - Hospital 1:**
```bash
HOSPITAL_ID=hospital_1 PORT=5001 python hospital_server/app.py
```

**Terminal 3 - Hospital 2:**
```bash
HOSPITAL_ID=hospital_2 PORT=5002 python hospital_server/app.py
```

**Terminal 4 - Hospital 3:**
```bash
HOSPITAL_ID=hospital_3 PORT=5003 python hospital_server/app.py
```

### 3. Access Hospital Dashboard

Visit the hospital dashboards:
- Hospital 1: http://localhost:5001
- Hospital 2: http://localhost:5002
- Hospital 3: http://localhost:5003

### 4. Upload Patient Records

From Hospital 1 dashboard:
1. Create a test CSV/JSON file with patient data (see examples below)
2. Select "hospital_2" as destination
3. Click "Upload and route" button
4. Records are anonymized by main server and sent to hospital_2

### 5. Retrieve Patient Data

From Hospital 2 dashboard:
1. Scroll down to "Browse" section
2. Click "📥 Retrieve Patient Data" button
3. View received records from hospital_1 in a formatted table
4. Browse through all batches and individual records

## Example Patient Data Format

### JSON Format
```json
{
  "records": [
    {
      "patient_id": "P001",
      "age": 45,
      "diagnosis": "Type 2 Diabetes",
      "blood_pressure": "120/80"
    },
    {
      "patient_id": "P002",
      "age": 62,
      "diagnosis": "Hypertension",
      "blood_pressure": "140/90"
    }
  ]
}
```

### CSV Format
```csv
patient_id,age,diagnosis,blood_pressure
P001,45,Type 2 Diabetes,120/80
P002,62,Hypertension,140/90
```

## Features

### Dashboard View
- **Hospital ID**: Currently logged-in hospital identifier
- **MNIST samples ready**: Number of training samples loaded
- **Received processed records**: Count of record batches received

### Upload Section
- Select destination hospital
- Upload JSON or CSV files
- View preview of processed records
- See anonymization metrics (removal accuracy)

### Retrieved Data Section
- **Batch Header**: Shows source hospital, record count, anonymization metrics
- **Record Detail**: View individual records in formatted JSON
- **Non-consuming**: Data retrieval doesn't remove records (can view multiple times)

## API Endpoints

### Hospital Server Endpoints

**Retrieve patient data (non-consuming):**
```
POST /retrieve_patient_data
Response: {
  "hospital_id": "hospital_2",
  "batch_count": 2,
  "record_count": 50,
  "batches": [...],
  "timestamp": "2026-05-22T10:30:00.000Z"
}
```

**Upload patient records:**
```
POST /upload_patient_records
Body: form-data with destination_hospital_id and records_file
```

**Get dashboard data:**
```
GET /dashboard_data
Response: All dashboard statistics and history
```

## Notes

- **Privacy**: Patient records are anonymized by the main server before delivery
- **Non-consuming retrieval**: Using the UI to view data doesn't clear it from the inbox
- **Batch organization**: Records are grouped by source hospital and timestamp
- **Multiple views**: You can retrieve the same data multiple times
- **Testing**: The synthetic data generator is useful for testing without real patient data

## Troubleshooting

### "No patient data available"
- Upload records from another hospital first
- Check that the main server is running
- Ensure records were successfully routed

### "Failed to retrieve data"
- Verify hospital server is running
- Check browser console for error details
- Ensure main server endpoint is accessible

### Data not appearing
- Records take a moment to appear after upload
- Try clicking "Retrieve Patient Data" again
- Check that records were routed to the correct hospital
