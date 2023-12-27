from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow import keras



train_data_dir = 'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\train_data'
validation_data_dir = 'C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\validation_data'

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3]
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# model architecture
model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Three classes: planet, star, nebula

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

#calucating accuracy
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('C:\\Users\\AAHILAHMED\\PycharmProjects\\pythonProject\\trained_models\\nebulae_classification_model.h5')
