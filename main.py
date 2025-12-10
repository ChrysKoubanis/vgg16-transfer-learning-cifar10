import os

#silence tf messages
# 0 = all messages (default)
# 1 = hide info
# 2 = hide info and warnings (This clears the CPU/AVX messages)
# 3 = hide info, warnings, and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Turn off the specific "oneDNN" message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Loading the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten the labels from (50000, 1) to (50000,)
y_train = y_train.ravel()
y_test = y_test.ravel()

# Check if the data is loaded correctly
#print("Training data (shape):", x_train.shape)   
#print("Training labels (shape):", y_train.shape)
#print("Test data (shape):", x_test.shape)
#print("Test labels (shape):", y_test.shape)

# I will use transfer learning with VGG16 to extract features from the CIFAR-10 images


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
print("Loading the VGG16 model...")
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

