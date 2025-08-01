# 🧠 Customer Churn Prediction using Artificial Neural Networks

This project applies an Artificial Neural Network (ANN) to predict customer churn using a dataset from Kaggle. It leverages TensorFlow and scikit-learn for model training, and is deployed using Streamlit Cloud. Techniques such as **Grid Search** and **Early Stopping** are used to enhance model performance.

---

## 📌 Project Highlights

- Built with **TensorFlow 2.15.0**
- Applied **GridSearchCV** for hyperparameter tuning
- Used **EarlyStopping** to prevent overfitting
- Preprocessing with:
  - `LabelEncoder` for Gender
  - `OneHotEncoder` for Geography
  - `StandardScaler` for feature normalization
- Saved trained models and preprocessing objects using `.h5` and `.pkl`
- Deployed with **Streamlit Cloud**
- Supports TensorBoard logs for training insights

---

## 📁 Project Structure

```
Churn-Modelling-Using-ANN/
│
├── dataset/                        # Contains raw input data
├── logs/                           # Training and TensorBoard logs
├── regressionlogs/                 # Regression model logs (if applicable)
│
├── app.py                          # Streamlit web app
├── model.h5                        # Trained ANN classification model
├── regression_model.h5             # Optional regression model
├── scaler.pkl                      # Scaler used for feature normalization
├── label_encoder_gender.pkl        # Label encoder for Gender feature
├── onehot_encoder_geo.pkl          # One-hot encoder for Geography
│
├── requirements.txt                # List of required Python libraries
├── README.md                       # This file
│
├── prediction.ipynb               # Notebook for model predictions
├── research.ipynb                 # Research and experimentation
├── salaryregression.ipynb         # Regression analysis (optional)
└── hyperparametertunninggann.ipynb# Grid search and hyperparameter tuning
```

---

## 📊 Dataset Overview

Sourced from [Kaggle](https://www.kaggle.com/api/v1/datasets/download/shrutimechlearn/churn-modelling), this dataset contains:

- Customer demographics (Age, Gender, Geography)
- Account information (Credit Score, Balance, Products, etc.)
- Whether the customer exited (target variable: `Exited`)

---

## 🔍 Model Optimization

### ✅ Grid Search
Used `GridSearchCV` to tune:

- Optimizer (`adam`, `rmsprop`)
- Batch size
- Number of epochs
- Number of hidden units

### ✅ Early Stopping
Configured with patience and monitored `val_loss` to avoid overfitting.

---

## ⚙️ Installation & Running the App

### 🔧 Requirements

```txt
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
```

Install them using:

```bash
pip install -r requirements.txt
```

### ▶️ Run Locally

```bash
streamlit run app.py
```

### 📈 View TensorBoard Logs

```bash
tensorboard --logdir=logs/
```

---


## 📌 Example Use Case

> A bank can use this solution to automatically flag customers likely to leave, allowing retention teams to act preemptively.

---
