import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.callbacks import EarlyStopping

from data_preprocessing import load_and_preprocess_data
from model import build_model
from experiment_logger import ExperimentLogger

def train_model():
    # Parameters
    epochs = 50
    batch_size = 32
    model_name = "ANN_Churn_Predictor"

    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(X_train.shape[1])

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
        class_weight=class_weights
    )
    
    # Save the trained model
    model.save("churn_prediction_model.keras")
    print("Model trained and saved as churn_prediction_model.keras")

    # Log the experiment
    logger = ExperimentLogger()
    final_metrics = {
        "accuracy": history.history['accuracy'][-1],
        "loss": history.history['loss'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1]
    }
    logger.log_experiment(model_name, epochs, batch_size, final_metrics)

    return history, model

if __name__ == "__main__":
    train_model()
