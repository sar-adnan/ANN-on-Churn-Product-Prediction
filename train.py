import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.callbacks import EarlyStopping

from data_preprocessing import load_and_preprocess_data
from model import build_model

def train_model():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight=\"balanced\".encode("utf-8"),
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(X_train.shape[1])

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor=\'val_loss\',   # monitor validation loss
        patience=5,           # stop after 5 epochs with no improvement
        restore_best_weights=True  # roll back to best model
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1,
        class_weight=class_weights
    )
    
    # Save the trained model
    model.save("churn_prediction_model.keras")
    print("Model trained and saved as churn_prediction_model.keras")

    return history, model

if __name__ == "__main__":
    train_model()
