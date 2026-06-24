# 📊 Customer Churn Prediction using Artificial Neural Networks (ANN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project focuses on predicting **Customer Churn** for a telecommunications company using a deep learning approach. Customer churn occurs when customers stop doing business with a company, and predicting this is crucial for retention strategies. We implement an **Artificial Neural Network (ANN)** to classify whether a customer is likely to churn based on their usage patterns and demographic data.

### 💼 Business Impact

By accurately identifying at-risk customers, businesses can:
- Implement targeted retention campaigns.
- Improve customer satisfaction by addressing pain points.
- Reduce the high cost of acquiring new customers compared to retaining existing ones.

## 🏗️ Project Architecture

The project is designed with a **modular production-ready structure** to ensure scalability and maintainability:

- **`data_preprocessing.py`**: Automated pipeline for data cleaning, handling missing values, encoding categorical variables, and feature scaling.
- **`model.py`**: Definition of the ANN architecture, including dropout layers for regularization to prevent overfitting.
- **`train.py`**: Training script that incorporates class weighting (to handle imbalanced data) and early stopping.
- **`ANN.ipynb`**: Interactive notebook for demonstration and exploratory analysis.

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sar-adnan/ANN-on-Churn-Product-Prediction.git
   cd ANN-on-Churn-Product-Prediction
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

To train the model and generate the prediction results, run:
```bash
python train.py
```

## 📊 Dataset

The model is trained on the **Telco Customer Churn** dataset, which includes:
- **Demographics:** Gender, seniority, partners, and dependents.
- **Services:** Phone, multiple lines, internet, online security, backup, device protection, tech support, and streaming.
- **Account Info:** Tenure, contract type, payment method, paperless billing, monthly charges, and total charges.

## 📈 Model Performance

The ANN model utilizes:
- **Input Layer:** Matches the number of preprocessed features.
- **Hidden Layers:** Multiple dense layers with ReLU activation.
- **Regularization:** Dropout layers (0.3) to ensure robust generalization.
- **Output Layer:** Sigmoid activation for binary classification (Churn vs. No Churn).

## 📅 Roadmap (Curriculum Progress)

- [x] **Day 1:** Refactor notebook into modular Python scripts.
- [x] **Day 2:** Create comprehensive documentation and architecture overview.
- [ ] **Day 3:** Implement Data Validation (Pydantic).
- [ ] **Day 4:** Add Experiment Tracking.
- [ ] **Day 5:** Build Streamlit Inference Dashboard.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
