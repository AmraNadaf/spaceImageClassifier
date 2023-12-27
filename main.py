

from my_app_test import app  # Import your Flask app from your_app.py

if __name__ == '__main__':
    app.run(debug=True)










'''
#PROGRAM 1(Works)(time for trainig acceptable) [CNN]
import os
import numpy as np
from keras.src.layers import MaxPooling2D, Conv2D, Flatten, Dense
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# Setting the directory
directory1 = 'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\train_data\\planet'
directory2 = 'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\train_data\\nebulae'
directory3 = 'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\train_data\\star'

#initilise common prefix
common_prefix1 = 'planet_image_'
common_prefix2 = 'nebulae_image_'
common_prefix3 = 'star_image_'

# list all files
files1 = os.listdir(directory1)
files2 = os.listdir(directory2)
files3 = os.listdir(directory3)

# Counter for unique numbers
c1 = 1
c2 = 1
c3 = 1

# Function to make unique image names
def get_unique_filename(directory, common_prefix, counter):
    while True:
        new_filename = common_prefix + str(counter) + '.jpg'
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1

# Loop through the files and rename them
for filename in files1:
    if filename.endswith('.jpg'):
        new_filename = get_unique_filename(directory1, common_prefix1, c1)
        os.rename(os.path.join(directory1, filename), os.path.join(directory1, new_filename))
        c1 += 1

for filename in files2:
    if filename.endswith('.jpg'):
        new_filename = get_unique_filename(directory2, common_prefix2, c2)
        os.rename(os.path.join(directory2, filename), os.path.join(directory2, new_filename))
        c2 += 1

for filename in files3:
    if filename.endswith('.jpg'):
        new_filename = get_unique_filename(directory3, common_prefix3, c3)
        os.rename(os.path.join(directory3, filename), os.path.join(directory3, new_filename))
        c3 += 1

# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3]  # Adjust the range as needed
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\train_data',  # Replace with the actual path to the training data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\validation_data',  # Replace with the actual path to the validation data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Model Building
#tryin to fix the model error

model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Three classes: planet, star, nebula


# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

model.save('C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\trained_models\\nebulae_classification_model.h5')
# Evaluation and Testing
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")


#user input
class_labels = ['nebulae', 'planet', 'star']
# Function to classify an image
from PIL import Image

def classify_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = class_labels[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        return str(e)
while(True):
    # Get input image from the user
    image_path = input("Enter the path to the astronomical image you want to classify: ")

    # Perform classification
    predicted_class = classify_image(image_path)

    if "Error" in predicted_class:
        print(f"An error occurred: {predicted_class}")
    else:
        print(f"The image belongs to class: {predicted_class}")






#(Noiceeeeeee works with an average accuracy of 89 -91% and now its predicting nebulae also correctlty hehehe )

'''
