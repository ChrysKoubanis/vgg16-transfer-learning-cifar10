import os
import time
import warnings
import matplotlib.pyplot as plt

# Environment configuration to clean console logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score

# Data structure to track comparative metrics
final_results = {}
time_results = {}

# 1. Load Dataset
print("[+] Loading CIFAR-10 Dataset...")
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

y_train_full = y_train_full.ravel()
y_test_full = y_test_full.ravel()

# Set FULL_DATASET to False during testing/debugging
FULL_DATASET = True

if not FULL_DATASET:
    n_train, n_test = 15000, 2000
    x_train, y_train = x_train_full[:n_train], y_train_full[:n_train]
    x_test, y_test = x_test_full[:n_test], y_test_full[:n_test]
else:
    x_train, y_train = x_train_full, y_train_full
    x_test, y_test = x_test_full, y_test_full

# 2. VGG16 Transfer Learning Feature Extraction
print("\n[+] Preprocessing images for VGG16...")
x_train_vgg = preprocess_input(x_train)
x_test_vgg = preprocess_input(x_test)

print("[+] Loading pre-trained VGG16 model (ImageNet weights)...")
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')

print("[+] Extracting feature vectors...")
features_train = vgg_base.predict(x_train_vgg)
features_test = vgg_base.predict(x_test_vgg)

# 3. Feature Scaling & Dimensionality Reduction
print("\n[+] Standardizing feature vectors...")
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

print("[+] Applying PCA (retaining 90% variance)...")
pca = PCA(n_components=0.9)
pca.fit(features_train_scaled)

x_train_pca = pca.transform(features_train_scaled)
x_test_pca = pca.transform(features_test_scaled)
print(f"    Reduced feature space from {features_train.shape[1]} to {x_train_pca.shape[1]} dimensions.")

# 4. Support Vector Machine (SVM) Classifiers
print("\n[+] Training SVM Classifiers...")
kernels = ['rbf', 'poly']

for kernel_type in kernels:
    print(f"\n--- Training SVM ({kernel_type.upper()} kernel) ---")
    svm = SVC(kernel=kernel_type, C=10.0, degree=3, cache_size=1000)
    
    start_time = time.time()
    svm.fit(x_train_pca, y_train)
    training_time = time.time() - start_time
    
    test_acc = svm.score(x_test_pca, y_test) * 100
    print(f"Training Time:   {round(training_time, 2)} seconds")
    print(f"Testing Accuracy: {round(test_acc, 2)}%")

    final_results[f"SVM-{kernel_type.upper()}"] = test_acc
    time_results[f"SVM-{kernel_type.upper()}"] = training_time

# 5. Baseline Model Comparisons
print("\n[+] Training Baseline & Comparative Models...")

# k-Nearest Neighbors
for k in [1, 3]:
    print(f"\n--- Training {k}-Nearest Neighbors ---")
    knn = KNeighborsClassifier(n_neighbors=k)
    
    start_time = time.time()
    knn.fit(x_train_pca, y_train)
    test_acc = knn.score(x_test_pca, y_test) * 100
    duration = time.time() - start_time
    
    print(f"Execution Time:   {round(duration, 2)} seconds")
    print(f"Testing Accuracy: {round(test_acc, 2)}%")

    final_results[f"{k}-NN"] = test_acc
    time_results[f"{k}-NN"] = duration

# Nearest Class Centroid (NCC)
print("\n--- Training Nearest Class Centroid (NCC) ---")
centroid = NearestCentroid()

start_time = time.time()
centroid.fit(x_train_pca, y_train)
test_acc = centroid.score(x_test_pca, y_test) * 100
duration = time.time() - start_time

print(f"Execution Time:   {round(duration, 2)} seconds")
print(f"Testing Accuracy: {round(test_acc, 2)}%")

final_results["NCC"] = test_acc
time_results["NCC"] = duration

# Multi-Layer Perceptron (MLP) with Hinge Loss
print("\n--- Training MLP with Categorical Hinge Loss ---")
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

mlp = Sequential([
    Dense(128, input_dim=x_train_pca.shape[1], activation='relu'),
    Dense(10, activation='linear')
])

mlp.compile(loss='categorical_hinge', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

start_time = time.time()
mlp.fit(x_train_pca, y_train_onehot, epochs=20, batch_size=32, verbose=0)
duration = time.time() - start_time

_, acc = mlp.evaluate(x_test_pca, y_test_onehot, verbose=0)
test_acc = acc * 100

print(f"Execution Time:   {round(duration, 2)} seconds")
print(f"Testing Accuracy: {round(test_acc, 2)}%")

final_results["MLP"] = test_acc
time_results["MLP"] = duration

# 6. Visualization Generation
print("\n[+] Generating Results Visualization Plot...")
names = list(final_results.keys())
acc_values = list(final_results.values())
time_values = list(time_results.values())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy Plot
colors_acc = ['blue', 'blue', 'blue', 'green', 'green', 'orange']
ax1.bar(names, acc_values, color=colors_acc[:len(names)])
ax1.set_title('Testing Accuracy (%) - Higher is Better', fontweight='bold')
ax1.set_ylabel('Accuracy %')
ax1.set_ylim(0, 100)

for i, v in enumerate(acc_values):
    ax1.text(i, v + 1, f"{round(v, 1)}%", ha='center', fontweight='bold')

# Training/Execution Time Plot
colors_time = ['salmon'] * len(names)
ax2.bar(names, time_values, color=colors_time)
ax2.set_title('Training/Exec Time (seconds) - Lower is Better', fontweight='bold')
ax2.set_ylabel('Seconds')

for i, v in enumerate(time_values):
    ax2.text(i, v + 0.5, f"{round(v, 2)}s", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('Figure_1.png', dpi=300)
plt.show()