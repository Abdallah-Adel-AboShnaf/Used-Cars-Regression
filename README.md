# 🚗 Used Car Price Predictor

A machine learning web application that predicts **used car prices** based on vehicle specifications.  
The project follows a complete **data preprocessing → feature engineering → model training → deployment** pipeline and is deployed using **Streamlit**.

---

## 📌 Project Overview

This project aims to estimate the price of a used car using historical car listing data.  
It replicates the full data cleaning and feature engineering workflow from the Jupyter Notebook and deploys the trained model in an interactive web interface.

- **Model Used:** Decision Tree Regressor  
- **Frontend:** Streamlit  
- **Language:** Python  

---

## 🧠 Machine Learning Pipeline

### 1. Data Cleaning
- Removed irrelevant columns (e.g. `price_drop`, `seller_rating`)
- Removed duplicate records
- Handled missing values:
  - Numerical → Median
  - Categorical → Most frequent value

### 2. Feature Engineering
New features created:
- `is_turbo` (from engine description)
- `is_automatic` (from transmission)
- Drivetrain flags: `is_AWD`, `is_FWD`, `is_RWD`
- `total_usage_score` (one owner + personal use)

### 3. Encoding & Scaling
- **Categorical Encoding:** Label Encoding
- **Scaling:** StandardScaler

### 4. Model Training
- Algorithm: **DecisionTreeRegressor**
- Training performed on scaled features
- Random state fixed for reproducibility

---

## 🖥️ Web Application (Streamlit)

The Streamlit app allows users to:
- Select car specifications from dropdowns
- Enter numerical details (year, mileage, ratings)
- Predict car price instantly

### Key Features
- Sidebar-based input form
- Automatic handling of unseen categorical values
- Real-time prediction output
- Cached data loading for performance

---

```md
## 🗂️ Project Structure

├── data/
│ └── cars.csv
├── models/
│ └── decision_tree_model.pkl
├── Used Cars Regression.ipynb
├── GUI.py
└── README.md
```


---

## ▶️ How to Run the Project

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn
```

### 2. Run the App
```bash
streamlit run GUI.py
```

## 📊 Model Output

Displays estimated used car price in USD

Handles missing or unseen values safely

Uses the same preprocessing logic as training
