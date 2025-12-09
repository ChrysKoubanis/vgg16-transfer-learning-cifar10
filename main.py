import tensorflow as tf
from tensorflow.keras.datasets import cifar10

#Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Flatten the labels from (50000, 1) to (50000,)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Check if the data is loaded correctly
print("Training data (shape):", x_train.shape)   
print("Training labels (shape):", y_train.shape)
print("Test data (shape):", x_test.shape)
print("Test labels (shape):", y_test.shape)

import sys
import tensorflow as tf

print("--- ΕΛΕΓΧΟΣ ΠΕΡΙΒΑΛΛΟΝΤΟΣ ---")
print(f"Python που τρέχει τώρα: {sys.executable}")
print(f"TensorFlow version: {tf.__version__}")
print("----------------------------")