# Medical Data Sources

FedCare now uses MedMNIST for proper medical federated training.

- MedMNIST catalog: https://medmnist.com/
- MedMNIST GitHub: https://github.com/MedMNIST/MedMNIST

Common datasets in MedMNIST:

- `pathmnist` (Pathology images)
- `chestmnist` (Chest X-Rays)
- `dermaamnist` (Dermatology images)
- `octmnist` (Retinal OCT images)
- `pneumoniamnist` (Pneumonia Chest X-Rays)

For the FedCare hospital UI in this repo:

- Local model training already uses Medical Data automatically on hospital startup via the `medmnist` library.
- The upload flow in the dashboard is for patient-record files (CSV/JSON), not raw image archives.
- Sample patient uploads are available in:
  - `data_simulation/patient_records_hospital_1.json`
  - `data_simulation/patient_records_hospital_2.json`
