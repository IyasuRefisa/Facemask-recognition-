import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

class FaceMaskModel:
    def __init__(self, input_shape=(299, 299, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Defines the CNN model architecture."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2),
            Dropout(0.25),
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        """Compiles the model with loss function and optimizer."""
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        print("Model compiled.")

    def summary(self):
        """Prints a summary of the model architecture."""
        self.model.summary()