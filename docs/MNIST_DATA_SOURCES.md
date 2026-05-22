# MNIST Data Sources

Use the official MNIST sources below if you want the original files:

- TensorFlow MNIST catalog: https://www.tensorflow.org/datasets/catalog/mnist
- Original MNIST homepage: http://yann.lecun.com/exdb/mnist/

Common files from the original MNIST dataset:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

For the FedCare hospital UI in this repo:

- Local model training already uses MNIST automatically on hospital startup.
- The upload flow is for patient-record files, not raw MNIST image archives.
- Sample patient uploads are available in:
  - `data_simulation/patient_records_hospital_1.json`
  - `data_simulation/patient_records_hospital_2.json`
