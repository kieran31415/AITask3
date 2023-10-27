import streamlit as st
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras import optimizers
from keras import layers
import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np

# Define a Streamlit app
def main():
    st.title("Vehicle Classifier EDA and Training Controls")

    # EDA Section
    st.header("Exploratory Data Analysis (EDA)")
    
    # Display some sample images
    st.subheader("Sample Images")
    
    root_directory = "./data"
    def count_pictures(root_directory, vehicle):
        counter = 0
        for foldername, subfolders, filenames in os.walk(root_directory):
            for filename in filenames:
                if vehicle in filename:
                    counter += 1
        return counter

# Loop over all the vehicles
    vehicles = ['boat', 'bus', 'car', 'motorbike', 'plane']
    total_count = 0
    for vehicle in vehicles:
    # Add and print the amount of pictures of this type of vehicle
        count = count_pictures(root_directory, vehicle)
        st.text(f"Number of pictures of a {vehicle}: {count}")
        total_count += count
        count = 0
# Print the total amount of pictures
    st.text(f"Total amount of pictures: {total_count}")
    
    # Define a map to your GitHub repository
    map = 'https://raw.githubusercontent.com/kieran31415/AI/main/Homework/Task3/data'

    image_data = [
        {"path": f"{map}/testing_set/boat/boat.13.jpg", "title": "Boat"},
        {"path": f"{map}/testing_set/bus/bus.2.jpg", "title": "Bus"},
        {"path": f"{map}/testing_set/car/car.2.jpg", "title": "Car"},
        {"path": f"{map}/testing_set/motorbike/motorbike.0.jpg", "title": "Motorbike"},
        {"path": f"{map}/testing_set/plane/plane.26.jpg", "title": "Plane"}
    ]

    # Display images using Streamlit
    st.title("Image Gallery")

    for image_info in image_data:
        st.subheader(image_info["title"])
        st.image(image_info["path"])

    # Training Controls Section
    st.header("Training Controls")

    # Slider for the number of training epochs
    num_epochs = st.slider("Number of Epochs", min_value=1, max_value=50, value=20)

    # Button to start training
    if st.button("Start Training"):
        st.text(f"Training model for {num_epochs} epochs...")

        # Include your training code here, updating the model with the selected options
        base_url = "https://github.com/kieran31415/AI/tree/main/Homework/Task3/data/"

    # Create an ImageDataGenerator
        train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                           rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

    # Create a list of categories
        categories = ['boat', 'bus', 'car', 'motorbike', 'plane']

    # Create the training and validation sets using the fetched images
        train_val_datagen = ImageDataGenerator(
            validation_split=0.2,     # Split the data into training and validation sets with an 80/20 ratio.
            rescale=1./255,           # Rescale pixel values to the range [0, 1].
            shear_range=0.2,          # Apply shear transformations to augment the data.
            zoom_range=0.2,           # Apply zoom transformations to augment the data.
            horizontal_flip=True      # Apply horizontal flipping as an augmentation technique.
        )

# Create an ImageDataGenerator for the test data, only rescaling is applied.
        test_datagen = ImageDataGenerator(
            rescale=1./255  # Rescale pixel values to the range [0, 1] for test data.
        )

# Create a training data generator from the 'training_set' directory.
        training_set = train_val_datagen.flow_from_directory(
            'data/training_set',     # Directory containing the training data.
            subset='training',       # Use the training subset of data.
            target_size=(64, 64),    # Resize images to a 64x64 pixel size.
            batch_size=32,           # Set the batch size for training data.
            class_mode='categorical' # Categorical labels for classification.
        )

# Create a validation data generator from the 'training_set' directory.
        validation_set = train_val_datagen.flow_from_directory(
            'data/training_set',     # Directory containing the training data.
            subset='validation',     # Use the validation subset of data.
            target_size=(64, 64),    # Resize images to a 64x64 pixel size.
            batch_size=32,           # Set the batch size for validation data.
            class_mode='categorical' # Categorical labels for classification.
        )

# Create a test data generator from the 'testing_set' directory.
        test_set = test_datagen.flow_from_directory(
            'data/testing_set',      # Directory containing the test data.
            target_size=(64, 64),    # Resize images to a 64x64 pixel size.
            batch_size=32,           # Set the batch size for test data.
            class_mode='categorical' # Categorical labels for classification.
        )


        # For example, update the `num_epochs` and `use_regularization` in your training code
        NUM_CLASSES = 5

        # Create a sequential model with a list of layers
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="sigmoid"),
            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        # Compile and train your model as usual
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

        print(model.summary())

        # Once training is done, you can display the loss and accuracy plots as you did before.
        history = model.fit(training_set,
                            validation_data=validation_set,
                            epochs=num_epochs
                            )
        # Display the loss and accuracy plots (similar to your code)
        st.subheader("Training Progress")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the loss curves on the first subplot
        ax1.plot(history.history['loss'], label='training loss')
        ax1.plot(history.history['val_loss'], label='validation loss')
        ax1.set_title('Loss curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the accuracy curves on the second subplot
        ax2.plot(history.history['accuracy'], label='training accuracy')
        ax2.plot(history.history['val_accuracy'], label='validation accuracy')
        ax2.set_title('Accuracy curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Display the figure using Streamlit
        st.pyplot(fig)

if __name__ == "__main__":
    main()