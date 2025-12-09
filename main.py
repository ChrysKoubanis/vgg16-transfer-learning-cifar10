import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

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

# --- Step 2: VGG16 Feature Extraction ---
print("\n--- Step 2: Processing with VGG16 ---")
    
# 1. Preprocess the data (required for VGG16)
# This scales pixel values specifically for this model
x_train_vgg = preprocess_input(x_train)
x_test_vgg = preprocess_input(x_test)

# 2. Load the VGG16 model
# weights='imagenet': Use pre-trained weights
# include_top=False: Remove the final classification layer
# pooling='avg': Average the output to get a clean 1D vector
print("Loading VGG16 model...")
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='avg')
    
# 3. Extract features
print("Extracting features (this may take a moment)...")
features_train = model.predict(x_train_vgg)
features_test = model.predict(x_test_vgg)

print(f"Features shape (Train): {features_train.shape}")
print(f"Features shape (Test): {features_test.shape}")