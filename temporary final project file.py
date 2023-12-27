# Load the images directories
import os
import cv2
from imutils import paths
import matplotlib.pyplot as plt
from keras.src.optimizers import SGD, RMSprop, Nadam
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from keras.src.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

path = r"C:\Users\AAHILAHMED\OneDrive\Desktop\brain_tumor_dataset"

print(os.listdir(path))

image_paths = list(paths.list_images(path))
print(len(image_paths))

images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append(label)

# Plot an image
def plot_image(image):
    plt.imshow(image)

plot_image(images[0])

images = np.array(images) / 255.0
labels = np.array(labels)
images[0]

# Perform One-hot encoding
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

print(labels[0])

#Split the dataset
(train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size= 0.10, random_state= 42, stratify= labels)

# Build the Image Data Generator
train_generator = ImageDataGenerator(
    fill_mode='nearest',
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,  # Add shear augmentation
    zoom_range=0.2,   # Add zoom augmentation
)


import tensorflow as tf

from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a Sequential model


model = tf.keras.models.Sequential()


# Convolutional2D Layer 1
model.add(tf.keras.layers.Conv2D(filters=3,
                                 kernel_size=(3, 3),
                                 activation='relu',
                                 input_shape=(224, 224, 3),
                                 padding='same'))

# MaxPooling2D Layer 1
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional2D Layer 2
model.add(tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 activation='relu',
                                 padding='valid'))

# MaxPooling2D Layer 2
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional2D Layer 3
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 activation='relu',
                                 padding='valid'))

# MaxPooling2D Layer 3
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(tf.keras.layers.Reshape((1, 6, 6, 64)))

# ConvLSTM2D Layer 1
# model.add(tf.keras.layers.ConvLSTM2D(filters=64,
                                    #  kernel_size=(3, 3),
                                    #  activation='relu',
                                    #  padding='valid',
                                    #  **{'return_sequences': False}))

# Fully Connected Layer (Dense)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))

# Output Layer
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()
opt= Nadam(learning_rate=1)
model.compile(optimizer = opt,
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])
batch_size = 8
train_steps = len(train_X) // batch_size
validation_steps = len(test_X) // batch_size
epochs = 15

# Fit the model
history = model.fit_generator(train_generator.flow(train_X, train_Y, batch_size= batch_size),
                              steps_per_epoch= train_steps,
                              validation_data = (test_X, test_Y),
                              validation_steps= validation_steps,
                              epochs= epochs)

predictions = model.predict(test_X, batch_size= batch_size)
predictions = np.argmax(predictions, axis= 1)
actuals = np.argmax(test_Y, axis= 1)

# Print Classification report and Confusion matrix
print(classification_report(actuals, predictions, target_names= label_binarizer.classes_))

cm = confusion_matrix(actuals, predictions)
print(cm)
# Final accuracy of our model
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print("Accuracy: {:.4f}".format(accuracy))

