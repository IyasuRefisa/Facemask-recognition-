import tensorflow as tf
import matplotlib.pyplot as plt

class ModelTraining:
    def __init__(self, model, checkpoint_path="saved_models/model.h5"):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.callbacks = self._set_callbacks()

    def _set_callbacks(self):
        """Sets up early stopping and checkpoint callbacks."""
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1
        )
        return [es_callback, chkpt_callback]

    def train(self, train_gen, val_gen, epochs=20):
        """Trains the model with given data generators."""
        history = self.model.fit(
            train_gen, epochs=epochs, validation_data=val_gen, callbacks=self.callbacks
        )
        self._plot_training_history(history)

    def _plot_training_history(self, history):
        """Plots training and validation accuracy and loss."""
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title('Model Accuracy')
        plt.show()

    def evaluate(self, test_gen):
        """Evaluates the model on the test set."""
        test_loss, test_acc = self.model.evaluate(test_gen)
        print(f"Test Accuracy: {test_acc:.2%}")
