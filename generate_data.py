import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import *
import sys

# Try to parse the dataset argument
dataset_name = sys.argv[1]

# Try to parse the sample size argument
try:
    sample_size = int(sys.argv[2])
except ValueError as e:
    print("The input value for sample size is invalid:", e)
    sys.exit(1)


# Load Dataset
if dataset_name.lower() == "mnist":
    X, y = mnist.load_data()[0]

    # If it's the mnist dataset, change the integer into their string representation, e.g. 0 --> "Digit 0"
    y = [f"Digit {int(val)}" for val in y]

elif dataset_name.lower() in ["cifar", "cifar10"]:
    X, y = cifar10.load_data()[0]

elif dataset_name.lower() in ["fashion", "fashion_mnist", "fashionmnist"]:
    X, y = fashion_mnist.load_data()[0]

else:
    print("Dataset not found.")
    sys.exit(1)

print("Dataset loaded.")

# Flatten the array, and normalize it
X = X.reshape(X.shape[0], -1)/255.


# We will select the integer values to be the index
df = pd.DataFrame(X, index=y)

if sample_size > df.shape[0]:
    print("Sample size is too great.")
    sys.exit(1)

samples = df.sample(n=sample_size, random_state=1234)

samples.to_csv(f"{dataset_name}_{sample_size}_input.csv", index=False)
pd.DataFrame(samples.index).to_csv(f"{dataset_name}_{sample_size}_labels.csv", index=False)

print("CSV files created.")
