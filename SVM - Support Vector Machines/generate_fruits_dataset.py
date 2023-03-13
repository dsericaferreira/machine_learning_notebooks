# %% [markdown]
# 1. Collect Images: Collect images of various fruits from different sources such as stock photo websites, online marketplaces, and your own photos. Ensure that the images are of high quality, and the fruits are clearly visible.
# 
# 2. Label the Images: Label each image with the corresponding fruit name. It is important to ensure that the labels are accurate and consistent.
# 
# 3. Preprocess the Images: Resize the images to a standardized size, such as 224x224 or 256x256 pixels. Apply normalization and augmentation techniques to the images to increase the variety of the dataset and improve the performance of the SVM model.
# 
# 4. Split the Dataset: Split the dataset into training, validation, and testing sets. The training set will be used to train the SVM model, the validation set will be used to tune hyperparameters and evaluate the performance during training, and the testing set will be used to evaluate the final performance of the model.
# 
# 5. Balance the Dataset: Ensure that the dataset is balanced, i.e., each class has roughly the same number of examples. This prevents the SVM model from being biased towards a particular fruit.
# 
# 6. Save the Dataset: Save the preprocessed and balanced dataset in a format that can be easily loaded and used in a machine learning framework such as TensorFlow or PyTorch.

# %%
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# %%
fruit_dir = '/Users/Raposinha/Desktop/machine_learning_notebooks/SVM - Support Vector Machines/images/train/train'

# %%
# Define the image size
image_size = (224, 224)


# %%
# Define the list of fruit classes
fruit_classes = ['Banana', 'Corn', 'Papaya', 'Blueberry', 'Plum', \
                 'Pineapple', 'Raspberry', 'Strawberry', 'Peach']

# %%
# Initialize the empty lists for images and labels
images = []
labels = []

# %%
# Iterate over the fruit classes
for fruit_class in fruit_classes:
    # Define the path to the fruit class directory
    class_dir = os.path.join(fruit_dir, fruit_class)
    
    # Iterate over the images in the fruit class directory
    for file in os.listdir(class_dir):
        # Load the image and resize it to the specified size
        image = Image.open(os.path.join(class_dir, file)).resize(image_size)
        
        # Convert the image to a numpy array and add it to the list of images
        images.append(np.array(image))
        
        # Add the label of the fruit class to the list of labels
        labels.append(fruit_class)

# %%
# Convert the list of images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# %%

# Shuffle the images and labels
images, labels = shuffle(images, labels, random_state=42)


# %%
# Split the dataset into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# %%
# Convert the fruit class labels to numerical labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)


# %%
# Save the preprocessed and balanced dataset
np.savez('fruits_dataset.npz', train_images=train_images, val_images=val_images, test_images=test_images, train_labels=train_labels, val_labels=val_labels, test_labels=test_labels)



