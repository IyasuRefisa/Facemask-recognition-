import os
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreparation:
    def __init__(self, zip_path, extract_path="data/dataset"):
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.target_size = (299, 299)  # Customize for model requirements
        self.batch_size = 32

    def extract_data(self):
        """Extracts the dataset from a zip file."""
        if not os.path.exists(self.extract_path):
            with ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
            print("Data extracted to:", self.extract_path)
        else:
            print("Data already extracted.")

    def setup_data_generators(self):
        """Sets up training, validation, and test data generators."""
        train_path = os.path.join(self.extract_path, 'train')
        val_path = os.path.join(self.extract_path, 'val')
        test_path = os.path.join(self.extract_path, 'test')

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            shear_range=0.25,
            zoom_range=0.05,
            horizontal_flip=True,
            width_shift_range=0.25,
            height_shift_range=0.5,
            brightness_range=(0.55, 0.35)
        )
        test_datagen = ImageDataGenerator()

        train_gen = train_datagen.flow_from_directory(
            train_path, target_size=self.target_size, batch_size=self.batch_size, class_mode='categorical'
        )
        val_gen = train_datagen.flow_from_directory(
            val_path, target_size=self.target_size, batch_size=self.batch_size, class_mode='categorical'
        )
        test_gen = test_datagen.flow_from_directory(
            test_path, target_size=self.target_size, batch_size=self.batch_size, class_mode='categorical'
        )

        return train_gen, val_gen, test_gen