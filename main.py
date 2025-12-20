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

# Data slicing here to so that the program doesnt take too long to run
n_train = 15000  # 15000 images to process instead of 50000 , we will change it later for the final run
n_test = 2000 # 2000 test images instead of 10000 , we will change it later for the final run

x_train = x_train_full[:n_train]
y_train = y_train_full[:n_train]
x_test = x_test_full[:n_test]
y_test = y_test_full[:n_test]

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
#kernels = ['linear', 'rbf', 'poly'] #linear , Radial Basis Function (Gaussian kernel) and polynomial
kernels = ['rbf', 'poly'] 
  
for kernel_type in kernels:
    print("\nTraining SVM with", kernel_type.upper(), "kernel...")
    
    # SVM initialization

    # C parameter controls the trade-off between maximizing the margin and minimizing classification errors.
    # C=1.0 as a balanced starting point
    # degree=3 (Only for the polynomial kernel) is the default for 'poly' 
    svm = SVC(kernel=kernel_type, C=100.0, degree=3, cache_size=1000)
    
    # 2. Train and Measure Time
    start_time = time.time()
    svm.fit(x_train_pca, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    print("Training Time (" + kernel_type + "):", round(training_time, 2), "seconds")
    
    # 3. Evaluate Accuracy
    train_pred = svm.predict(x_train_pca)
    test_pred = svm.predict(x_test_pca)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print("Training Accuracy:", round(train_acc*100, 2), "%")
    print("Testing Accuracy: ", round(test_acc*100, 2), "%")


# 1) First test (Subset: x,y train=5000 images, x,y test=1000 images): 
#   Linear: 7.44 seconds ,Training Accuracy: 97.08% Testing Accuracy: 54.4%   Overfitting
#   RBF: 2.72 seconds ,Training Accuracy: 84.62% Testing Accuracy: 59.7%     Best Performer. It balances learning and generalization better than Linear. 
#                                                                            This confirms that the image features have non-linear relationships.

#   Polynomial: 3.78 seconds ,Training Accuracy: 66.12% Testing Accuracy: 36.0%   Underfitting. The model failed to learn even the training data well.
#                                                                                 It likely needs hyperparameter tuning (higher degree or different coef0) or more data.


# 2) Second test (Subset: x,y train=15000 images, x,y test=2000 images): Also for the polynomial kernel i will increase the C from 1.0 to 10.0 
#                                                                        to allow fewer errors in training

#Linear: 3278.46 seconds (55 minutes) ,Training Accuracy: 79.17% Testing Accuracy: 59.85%   The massive training time proves that the CIFAR-10 features are not linearly
#                                                                                       separable.The optimizer had to iterate thousands of times to find a suboptimal boundary.

#   RBF: 53.56 seconds ,Training Accuracy: 98.17% Testing Accuracy: 63.05%     RBF handles the non-linear complexity of images efficiently.It achieved the highest accuracy
#                                                                              in a fraction of the time. This is the preferred model.

#   Polynomial: 65.0 seconds ,Training Accuracy: 93.77% Testing Accuracy: 60.15%   Faster than Linear but less accurate than RBF. It performed decently but 
#                                                                                 offers no advantage over RBF for this specific dataset.



