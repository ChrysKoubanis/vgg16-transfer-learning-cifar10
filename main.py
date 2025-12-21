import os
import warnings

#silence tf messages
# 0 = all messages (default)
# 1 = hide info
# 2 = hide info and warnings (This clears the CPU/AVX messages)
# 3 = hide info, warnings, and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Turn off the specific "oneDNN" message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


# Dictionary to store final results for the graphs
final_results = {}
time_results = {}

#Loading the CIFAR-10 dataset
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

# Flatten the labels from (50000, 1) to (50000,)
y_train_full = y_train_full.ravel()
y_test_full = y_test_full.ravel()

# Check if the data is loaded correctly
#print("Training data (shape):", x_train.shape)   
#print("Training labels (shape):", y_train.shape)
#print("Test data (shape):", x_test.shape)
#print("Test labels (shape):", y_test.shape)

# I will use transfer learning with VGG16 to extract features from the CIFAR-10 images

FULL_DATASET= True #Switch to False to use the subset


if not(FULL_DATASET):
    # Data slicing here to so that the program doesnt take too long to run
    n_train = 15000  # 15000 images to process instead of 50000 , we will change it later for the final run
    n_test = 2000 # 2000 test images instead of 10000 , we will change it later for the final run

    x_train = x_train_full[:n_train]
    y_train = y_train_full[:n_train]
    x_test = x_test_full[:n_test]
    y_test = y_test_full[:n_test]

else:
    x_train = x_train_full[:]
    y_train = y_train_full[:]
    x_test = x_test_full[:]
    y_test = y_test_full[:]


# VGG16 Feature Extraction

# 1) Preprocess the data (required for VGG16)
# VGG16 requires specific preprocessing of input images
# 'preprocess_input' converts RGB to BGR and zero-centers each color channel 
# with respect to the ImageNet dataset mean, helping the model converge faster
x_train_vgg = preprocess_input(x_train)
x_test_vgg = preprocess_input(x_test)

# 2) Load the VGG16 model
# weights='imagenet': We initialize the model with weights learned from ImageNet
# include_top=False: WE remove the final classification layer ,we only want the convolutional base (the "feature extractor")
# pooling='avg': Average the output to get a clean 1D vector (size 512) for each image
print(" Loading the VGG16 model...")
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')
    
# 3) Extract the features
# We pass our images through the network. The result is a vector of 512 numbers representing the high-level features of the image
print("Extracting the features ...")
features_train = model.predict(x_train_vgg)
features_test = model.predict(x_test_vgg)

#print("Features train shape: ",features_train.shape)
#print("Features test shape: " ,features_test.shape)



# Standardizing the features
# It scales the features to have a mean of 0 and a standard deviation of 1
# That ensures that all features are treated equally (because PCA chooses features with the highest range of values)


scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

#PCA 
print("Applying PCA...")

# PCA configuration:
# n_components=0.9 tells PCA to choose the minimum number of components needed to retain 90% of the information in the data
# We do this to reduce the number of inputs to the SVM to make it faster and less prone to overfitting
pca = PCA(n_components=0.9)
    
pca.fit(features_train_scaled) # Only fit on the training data because altering the testing data is cheating

# Transform applies the rules to change dimensions from 512 to 329 and must be applied to both training and testing data
x_train_pca = pca.transform(features_train_scaled) 
x_test_pca = pca.transform(features_test_scaled)

print("Features after PCA: ", x_train_pca.shape)
print("Dimensions reduced from ",features_train.shape[1] ,"to " ,x_train_pca.shape[1])  #from 512 to 329


# SVM CLASSIFICATION
print("\nSVM training...")

# We will test 
#kernels = ['linear','rbf', 'poly'] #linear , Radial Basis Function (Gaussian kernel) and polynomial
kernels = ['rbf', 'poly'] #I didnt use the linear kernel at the final run ,because it was projected to take over 15 hours to converge due to the non-linear nature of the data.


for kernel_type in kernels:
    print("\nTraining SVM with", kernel_type.upper(), "kernel...")
    
    # SVM initialization

    # C parameter controls the trade-off between maximizing the margin and minimizing classification errors.
    # C=1.0 as a balanced starting point
    # degree=3 (Only for the polynomial kernel) is the default for 'poly' 
    svm = SVC(kernel=kernel_type, C=10.0, degree=3, cache_size=1000)
    
    # Train and Measure Time
    start_time = time.time()
    svm.fit(x_train_pca, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    print("Training Time: ", round(training_time, 2), "seconds")
    
    # Evaluate Accuracy
    train_pred = svm.predict(x_train_pca)
    test_pred = svm.predict(x_test_pca)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print("Training Accuracy:", round(train_acc*100, 2), "%")
    print("Testing Accuracy: ", round(test_acc*100, 2), "%")

    acc = svm.score(x_test_pca, y_test)
    final_results[f"SVM-{kernel_type.upper()}"] = acc * 100
    
    time_results[f"SVM-{kernel_type.upper()}"] = training_time


# 1) First test (Subset: x,y train=5000 images, x,y test=1000 images): 
#   Linear: 7.44 seconds ,Training Accuracy: 97.08% Testing Accuracy: 54.4%   Overfitting
#   RBF: 2.72 seconds ,Training Accuracy: 84.62% Testing Accuracy: 59.7%     Best Performer. It balances learning and generalization better than Linear. 
#                                                                           This confirms that the image features have non-linear relationships.

#   Polynomial: 3.78 seconds ,Training Accuracy: 66.12% Testing Accuracy: 36.0%   Underfitting. The model failed to learn even the training data well.
#                                                                                   It likely needs hyperparameter tuning (higher degree or different coef0) or more data.


# 2) Second test (Subset: x,y train=15000 images, x,y test=2000 images): Also for the polynomial kernel i will increase the C from 1.0 to 10.0 
#                                                                           to allow fewer errors in training

#Linear: 3278.46 seconds (55 minutes) ,Training Accuracy: 79.17% Testing Accuracy: 59.85%   The massive training time proves that the CIFAR-10 features are not linearly
#                                                                                         separable.The optimizer had to iterate thousands of times to find a suboptimal boundary.

#   RBF: 53.56 seconds ,Training Accuracy: 98.17% Testing Accuracy: 63.05%     RBF handles the non-linear complexity of images efficiently.It achieved the highest accuracy
#                                                                                in a fraction of the time. This is the preferred model.

#   Polynomial: 65.0 seconds ,Training Accuracy: 93.77% Testing Accuracy: 60.15%   Faster than Linear but less accurate than RBF. It performed decently but 
#                                                                                   offers no advantage over RBF for this specific dataset.

#3) Second test (Subset: x,y train=15000 images, x,y test=2000 images): I changed C from 10 to 100 and i trained the SVM with only the RBF and Polynomial kernels.

#   RBF: 54.31 seconds ,Training Accuracy: 100.0% Testing Accuracy: 60.7%     Overfitting

#   Polynomial: 61.89 seconds ,Training Accuracy: 99.57% Testing Accuracy: 59.75%   Overfitting

# The reason for that huge overfitting gap is that if C is equal to 100 then the SVM is very strict and memorized every little detail and noise in the 15000 training set.


# We will keep C as 10 because it was the best performing model.


# Comparison with Other Models
print("\nComparison with Other Models")

# k-NN
for k in [1,3]:
    print("\nTraining " + str(k) + "-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=k)
    
    start_time = time.time()
    knn.fit(x_train_pca, y_train)
    
    acc = knn.score(x_test_pca, y_test)
    end_time = time.time()
    
    duration = end_time - start_time
    print("Total Time ("+str(k)+"-NN):", round(duration, 2), "seconds")
    print("Testing Accuracy:", round(acc * 100, 2), "%")

    final_results[f"{k}-NN"] = acc * 100
    time_results[f"{k}-NN"] = duration

# NCC
print("\nTraining Nearest Class Centroid...")
centroid = NearestCentroid()

start_time = time.time()
centroid.fit(x_train_pca, y_train)
acc = centroid.score(x_test_pca, y_test)
end_time = time.time()

duration = end_time - start_time
print("Total Time (Centroid):", round(duration, 2), "seconds")
print("Testing Accuracy:", round(acc * 100, 2), "%")

final_results["NCC"] = acc * 100
time_results["NCC"] = duration

# MLP with Hinge Loss
print("\nTraining MLP with Hinge Loss...")

y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

mlp = Sequential()
mlp.add(Dense(128, input_dim=x_train_pca.shape[1], activation='relu'))
mlp.add(Dense(10, activation='linear')) 

mlp.compile(loss='categorical_hinge', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

start_time = time.time()
mlp.fit(x_train_pca, y_train_onehot, epochs=20, batch_size=32, verbose=0)
end_time = time.time()

_, acc = mlp.evaluate(x_test_pca, y_test_onehot, verbose=0)
duration = end_time - start_time
print("Total Time (MLP):", round(duration, 2), "seconds")
print("Testing Accuracy:", round(acc * 100, 2), "%")

final_results["MLP"] = acc * 100
time_results["MLP"] = duration

# Subset size of 15000 and 2000
# 1-NN took 2.7 seconds ,Testing accuracy: 45.85%
# 3-NN took 0.22 seconds ,Testing accuracy: 48.55%
# NCC took 0.07 seconds ,Testing accuracy: 53.55%
# MLP with Hinge Loss took 14.69 seconds ,Testing Accuracy: 57.75 %

# Graphs
print("\nGenerating Comparison Graphs (Side-by-Side)")

names = list(final_results.keys())
acc_values = list(final_results.values())
time_values = list(time_results.values())

# Create a figure with 2 subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Graph 1: Accuracy ---
colors_acc = ['blue', 'blue', 'blue', 'green', 'green', 'orange', 'red'] # added extra blue since you have 3 kernels in loop
ax1.bar(names, acc_values, color=colors_acc[:len(names)])
ax1.set_title('Testing Accuracy (%) - Higher is Better')
ax1.set_ylabel('Accuracy %')
ax1.set_ylim(0, 100)
# Add numbers on bars
for i, v in enumerate(acc_values):
    ax1.text(i, v + 1, f"{round(v, 1)}%", ha='center', fontweight='bold')

# --- Graph 2: Time ---
colors_time = ['salmon'] * len(names) # Use reddish color for time
ax2.bar(names, time_values, color=colors_time)
ax2.set_title('Training/Exec Time (seconds) - Lower is Better')
ax2.set_ylabel('Seconds')
# Add numbers on bars
for i, v in enumerate(time_values):
    ax2.text(i, v, f"{round(v, 2)}s", ha='center', fontweight='bold')

plt.tight_layout()
plt.show() # This will pop up the window with the graph


# For the final run i used the whole database ,but i excluded the Linear kernel ,because based on the subset test,
# it was projected to take over 15 hours to converge due to the non-linear nature of the data.

#The output is Figure_1.png