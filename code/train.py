import tensorflow as tf
from data import get_datasets
from model import build_model
import os

def train_model(epochs=10, batch_size=32):

    datasets = get_datasets()
    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]

    input_shape = X_train.shape[1:]
    num_classes = len(set(y_train)) 
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model_path = os.path.join(os.path.dirname(__file__), "best_model.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Model training finished.")

    best_model = tf.keras.models.load_model(model_path)
    print("Best model loaded for evaluation.")
    
    return best_model, history

if __name__ == "__main__":

    EPOCHS = 20
    BATCH_SIZE = 64

    trained_model, training_history = train_model(epochs=EPOCHS, batch_size=BATCH_SIZE)