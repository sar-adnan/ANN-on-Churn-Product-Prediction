import csv
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file="experiments_log.csv"):
        self.log_file = log_file
        self.headers = ["timestamp", "model_name", "epochs", "batch_size", "accuracy", "loss", "val_accuracy", "val_loss"]
        self._initialize_log_file()

    def _initialize_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_experiment(self, model_name, epochs, batch_size, metrics):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp,
            model_name,
            epochs,
            batch_size,
            round(metrics.get("accuracy", 0), 4),
            round(metrics.get("loss", 0), 4),
            round(metrics.get("val_accuracy", 0), 4),
            round(metrics.get("val_loss", 0), 4)
        ]
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Experiment logged successfully to {self.log_file}")

if __name__ == "__main__":
    # Test logger
    logger = ExperimentLogger()
    test_metrics = {"accuracy": 0.85, "loss": 0.35, "val_accuracy": 0.82, "val_loss": 0.38}
    logger.log_experiment("ANN_Test_Model", 50, 32, test_metrics)
