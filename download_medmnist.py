import os
import medmnist

data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_root, exist_ok=True)

info = medmnist.INFO['pathmnist']
DataClass = getattr(medmnist, info['python_class'])

print(f"Downloading PathMNIST to {data_root}...")
DataClass(split='train', download=True, root=data_root)
DataClass(split='test', download=True, root=data_root)
print("Done!")

